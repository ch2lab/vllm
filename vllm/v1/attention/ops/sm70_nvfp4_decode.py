# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM70 (V100) NVFP4 paged DECODE attention with in-kernel dequantization.

Retires the fp16 shadow decode path (dequant-to-shadow + fp16 FA2) for
``kv_cache_dtype=nvfp4`` on sm_70. A two-kernel split-KV decode reads the
packed NVFP4 paged KV cache directly, dequantizes K/V in registers, and
runs flash-style online-softmax attention. No fp16 KV materialization, so
no persistent shadow buffer (the 4 GiB/GPU one) and no per-step dequant.

Performance design (V100: 80 SMs, no tensor cores usable from Triton):
- GQA grouping: one program handles the whole qo-head group of one kv head,
  so each KV byte is read once per kv head, not once per qo head.
- Split-KV: the request's KV blocks are partitioned across ``splits``
  programs; a tiny combine kernel merges the partial online-softmax states.
  ``splits`` is chosen so the grid fills the SMs (~80 programs), which
  matters for small batches and long context.
- The only persistent allocation is a ~0.6 MB partial-state workspace
  (fixed slot count independent of batch/split choice); the per-request KV
  lengths live in a 4-byte-each device buffer.

Dequant semantics and the LBHNC page layout are identical to
``sm70_nvfp4_dequant.py`` (bit-exact inverse of the SM70 store kernel).

CUDA-graph safe: both launches use a fixed grid for the captured graph
(``(num_reqs, num_kv_heads, splits)`` and ``(num_reqs, num_kv_heads)``);
per-request KV lengths are read from a device buffer at runtime and bound
the (runtime) block loops, so replays with different kv_lens stay
launch-invariant.

Sliding window (``window_left >= 0``) matches the FlashInfer FA2 mask: the
query attends to KV tokens ``[max(0, kv_len - 1 - window_left), kv_len)``.
"""

import torch

from vllm.triton_utils import tl, triton

from vllm.v1.attention.ops.sm70_nvfp4_dequant import (
    _nvfp4_e2m1_decode,
    _nvfp4_e4m3_decode,
)


@triton.jit
def _sm70_nvfp4_decode_split_kernel(
    q_ptr,  # fp16 [R, HQ, D]
    kv_ptr,  # uint8 [B, 2H, N, F] (opaque page bytes, LBHNC)
    bt_ptr,  # int32 [R, bt_cols]
    kvlens_ptr,  # int32 [R] (device) per-request KV lengths
    ws_ptr,  # f32 [SLOTS, RATIO, D+2] partial states (acc[D] | m | l)
    k_scale,  # f32 layer _k_scale_float
    v_scale,  # f32 layer _v_scale_float
    sm_scale,  # f32 softmax scale
    page,  # i64 block stride in bytes
    side_off,  # i64 V side base offset from K side base (H * N * F)
    bt_cols,  # i32 block_table row width
    window,  # i32 sliding window_left, < 0 = off
    splits,  # i32 split count for this launch
    H: tl.constexpr,  # num kv heads
    HQ: tl.constexpr,  # num qo heads
    D: tl.constexpr,  # head dim
    N: tl.constexpr,  # block size (tokens per page)
):
    """One program per (request, kv head, KV split); q_len = 1.

    Processes the qo-head group of this kv head over its chunk of the
    request's KV blocks and writes the partial online-softmax state
    (``[RATIO, D+2]``: output acc, running max m, running sum l) to the
    workspace slot ``[(r * H + h) * splits + s]``. Empty splits write
    ``m = -inf, l = 0, acc = 0``.
    """
    DD: tl.constexpr = D // 2  # data bytes per token row
    SD: tl.constexpr = D // 16  # scale bytes per token row
    RATIO: tl.constexpr = HQ // H
    RP: tl.constexpr = triton.next_power_of_2(RATIO)  # padded group

    r = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    slot = (r * H + h) * splits + sp

    d = tl.arange(0, D)
    dd = tl.arange(0, DD)
    sd = tl.arange(0, SD)
    n = tl.arange(0, N)
    i = tl.arange(0, RP)  # qo head within the group (padded)
    i_valid = i < RATIO

    kv_len = tl.load(kvlens_ptr + r)
    # Partial state for this slot: acc [RP, D], m [RP], l [RP].
    acc = tl.zeros((RP, D), dtype=tl.float32)
    m_i = tl.full((RP,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((RP,), dtype=tl.float32)

    if kv_len > 0:
        # Sliding window: attend to [lo, kv_len), FA2-compatible bounds.
        lo = kv_len - 1 - window
        lo = tl.maximum(tl.where(window >= 0, lo, 0), 0)
        b_lo = lo // N
        nb_hi = (kv_len + N - 1) // N
        # Contiguous partition of [b_lo, nb_hi) into `splits` chunks.
        span = nb_hi - b_lo
        b0 = b_lo + (span * sp) // splits
        b1 = b_lo + (span * (sp + 1)) // splits
        if b0 < b1:
            qg = tl.load(
                q_ptr + (r * HQ + h * RATIO).to(tl.int64) * D
                + i[:, None].to(tl.int64) * D
                + d[None, :],
                mask=i_valid[:, None],
                other=0.0,
            ).to(tl.float32)  # [RP, D]
            page64 = page.to(tl.int64)
            h64 = h.to(tl.int64)
            sf_head_base = H * (N * DD) + h64 * (N * SD)
            idx_d = tl.broadcast_to((d // 2)[None, :], (N, D))
            idx_sf = tl.broadcast_to((d // 16)[None, :], (N, D))

            for b in range(b0, b1):
                bid = tl.load(bt_ptr + r * bt_cols + b).to(tl.int64)
                data_base = bid * page64 + h64 * (N * DD)
                sf_base = sf_head_base + bid * page64

                # K side: dequant [N, D] in registers. Each token row is a
                # contiguous DD/SD-byte load, then a per-dim gather (the
                # 2-D interleave-indexed load form is miscompiled on sm_70,
                # see sm70_nvfp4_dequant.py).
                raw = tl.load(
                    kv_ptr + data_base + n.to(tl.int64)[:, None] * DD + dd[None, :]
                )  # [N, DD] u8
                nib = tl.where(
                    (d % 2)[None, :] == 1,
                    tl.gather(raw, idx_d, 1) >> 4,
                    tl.gather(raw, idx_d, 1),
                ) & 0x0F  # [N, D]
                vals = _nvfp4_e2m1_decode(nib).to(tl.float32)  # [N, D]
                sf = _nvfp4_e4m3_decode(
                    tl.load(
                        kv_ptr + sf_base + n.to(tl.int64)[:, None] * SD + sd[None, :]
                    )
                ).to(tl.float32) * k_scale  # [N, SD]
                K = (vals * tl.gather(sf, idx_sf, 1)).to(tl.float16)  # [N, D]

                # scores [RP, N]; mask tokens outside [lo, kv_len).
                scores = tl.sum(
                    K.to(tl.float32)[None, :, :] * qg[:, None, :], axis=2
                ) * sm_scale
                tok = b * N + n
                scores = tl.where(
                    (tok[None, :] >= lo) & (tok[None, :] < kv_len),
                    scores,
                    float("-inf"),
                )
                m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                p = tl.exp(scores - m_new[:, None])  # [RP, N]
                alpha = tl.exp(m_i - m_new)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]

                # V side: dequant [N, D], accumulate p @ V per qo head.
                raw_v = tl.load(
                    kv_ptr + data_base + side_off + n.to(tl.int64)[:, None] * DD + dd[None, :]
                )  # [N, DD] u8
                nib_v = tl.where(
                    (d % 2)[None, :] == 1,
                    tl.gather(raw_v, idx_d, 1) >> 4,
                    tl.gather(raw_v, idx_d, 1),
                ) & 0x0F  # [N, D]
                vals_v = _nvfp4_e2m1_decode(nib_v).to(tl.float32)  # [N, D]
                sf_v = _nvfp4_e4m3_decode(
                    tl.load(
                        kv_ptr + sf_base + side_off + n.to(tl.int64)[:, None] * SD + sd[None, :]
                    )
                ).to(tl.float32) * v_scale  # [N, SD]
                V = (vals_v * tl.gather(sf_v, idx_sf, 1)).to(tl.float16)  # [N, D]

                acc += tl.sum(
                    p[:, :, None] * V.to(tl.float32)[None, :, :], axis=1
                )
                m_i = m_new

    # Store partial state: slot [RP, D+2] (acc | m | l).
    ws_base = (slot.to(tl.int64) * (RP * (D + 2)))
    tl.store(ws_ptr + ws_base + i[:, None] * (D + 2) + d[None, :], acc)
    tl.store(ws_ptr + ws_base + i * (D + 2) + D, m_i)
    tl.store(ws_ptr + ws_base + i * (D + 2) + (D + 1), l_i)


@triton.jit
def _sm70_nvfp4_decode_combine_kernel(
    ws_ptr,  # f32 [SLOTS, RATIO, D+2] partial states
    out_ptr,  # fp16 [R, HQ, D]
    splits: tl.constexpr,  # split count
    H: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
):
    """Merge the split-KV partial states into the final output.

    One program per (request, kv head). A request whose KV is empty (all
    partials ``m = -inf``) gets zero rows.
    """
    RATIO: tl.constexpr = HQ // H
    RP: tl.constexpr = triton.next_power_of_2(RATIO)  # padded group

    r = tl.program_id(0)
    h = tl.program_id(1)

    d = tl.arange(0, D)
    i = tl.arange(0, RP)
    i_valid = i < RATIO
    SP: tl.constexpr = triton.next_power_of_2(splits)
    si = tl.arange(0, SP)
    si_valid = si < splits

    # Gather all split partials: [SP, RP, :]; padded split indices are
    # masked out (m = -inf => zero combine weight).
    slot = (
        (r * H + h).to(tl.int64) * (splits * (RP * (D + 2)))
        + si[:, None].to(tl.int64) * (RP * (D + 2))
        + i[None, :].to(tl.int64) * (D + 2)
    )  # [SP, RP]
    slot64 = slot.to(tl.int64)
    part = tl.load(
        ws_ptr + slot64[:, :, None] + d[None, None, :],
        mask=si_valid[:, None, None],
        other=0.0,
    )  # [SP, RP, D]
    m = tl.load(
        ws_ptr + slot64 + D, mask=si_valid[:, None], other=float("-inf")
    )  # [SP, RP]
    l = tl.load(
        ws_ptr + slot64 + (D + 1), mask=si_valid[:, None], other=0.0
    )  # [SP, RP]

    m_max = tl.max(m, axis=0)  # [RP]
    # Request with no KV: all m == -inf -> zero output.
    empty = m_max == float("-inf")
    m_safe = tl.where(empty, 0.0, m_max)
    w = tl.exp(m - m_safe[None, :])  # [splits, RP]
    denom = tl.sum(w * l, axis=0)  # [RP]
    denom = tl.where(denom > 0, denom, 1.0)
    out = tl.sum(
        w[:, :, None] * part, axis=0
    ) / denom[:, None]  # [RP, D]
    out = tl.where(empty[:, None], 0.0, out)

    qo0 = (r * HQ + h * RATIO).to(tl.int64) * D
    tl.store(
        out_ptr + qo0 + i[:, None] * D + d[None, :],
        out.to(tl.float16),
        mask=i_valid[:, None],
    )


def sm70_nvfp4_paged_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    out: torch.Tensor,
    k_scale: float,
    v_scale: float,
    sm_scale: float,
    window_left: int,
    splits: int,
    workspace: torch.Tensor,
) -> None:
    """In-kernel-dequant NVFP4 paged decode (q_len = 1 per request).

    Args:
        q: fp16 ``[num_reqs, num_qo_heads, D]`` queries (contiguous).
        kv_cache: raw uint8 layer cache ``[B, 2*H_kv, N, F]`` (LBHNC).
        block_table: int32 ``[num_reqs, max_blocks]`` (device).
        kv_lens: int32 ``[num_reqs]`` **device** per-request KV lengths.
        out: fp16 ``[num_reqs, num_qo_heads, D]`` (contiguous); written in
            place. Requests with ``kv_len == 0`` get zero rows.
        k_scale, v_scale: the layer's ``_k_scale_float`` / ``_v_scale_float``.
        sm_scale: softmax scale from the layer.
        window_left: sliding window size in tokens, negative = no window
            (FA2-compatible: attends to the last ``window_left + 1`` tokens).
        splits: KV split count (fixed per launch; grid dim).
        workspace: f32 buffer with at least
            ``splits * num_reqs * H_kv * next_power_of_2(HQ // H) * (D + 2)``
            elements; the per-(request, kv head, split) partial state,
            reused across layers/steps.
    """
    assert q.dtype == torch.float16 and out.dtype == torch.float16
    assert kv_cache.dtype == torch.uint8 and block_table.dtype == torch.int32
    _, two_h, N, F = kv_cache.shape
    H = two_h // 2
    D = 2 * (F * 8 // 9)
    R, HQ = q.shape[0], q.shape[1]
    assert HQ % H == 0, f"GQA requires {HQ} qo heads over {H} kv heads"
    RP = triton.next_power_of_2(HQ // H)
    assert R * H * splits * RP * (D + 2) <= workspace.numel()
    _sm70_nvfp4_decode_split_kernel[(R, H, splits)](
        q,
        kv_cache,
        block_table,
        kv_lens,
        workspace,
        k_scale,
        v_scale,
        sm_scale,
        page=kv_cache.stride(0),
        side_off=H * N * F,
        bt_cols=block_table.shape[1],
        window=window_left,
        splits=splits,
        H=H,
        HQ=HQ,
        D=D,
        N=N,
        num_warps=4,
    )
    _sm70_nvfp4_decode_combine_kernel[(R, H)](
        workspace,
        out,
        splits=splits,
        H=H,
        HQ=HQ,
        D=D,
        num_warps=4,
    )
