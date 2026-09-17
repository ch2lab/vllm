# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM70 (V100) NVFP4 KV-cache side dequant: paged NVFP4 -> dense FP16 shadow.

The FlashInfer FA2 paged reader cannot serve NVFP4 KV on sm_70: its 1-byte
in-loop dequant path requires ``ldmatrix`` (sm_75+), and the native decode
kernel has no fp4 path at all. The SM120-verified NVFP4-KV architecture
instead dequantizes the cache sides to 16-bit and runs mature 16-bit
attention kernels; on SM70 we dequantize to FP16 into a shadow paged buffer
(identity block table) and feed it to the V100-verified fp16 paged FA2
prefill/decode.

Dequant semantics (exact inverse of the SM70 store kernel
``csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu::
reshape_and_cache_nvfp4_kernel_sm70``):

    store:    sf_code = e4m3(global_scale * amax16 / 6), global_scale = 1/scale
              q       = round_e2m1(x * global_scale / float(sf_code))
    dequant:  x_hat   = E2M1_LUT[q] * float(sf_code) * scale

E2M1 LUT {0, .5, 1, 1.5, 2, 3, 4, 6} with sign in nibble bit 3; the scale
factor covers 16 consecutive dims. Low nibble = even dim, high = odd.

Physical layout (LBHNC / head-major block interior, one raw page
``[K_data | K_scale | V_data | V_scale]``):

    data[b, h, n, g]   = side_base + b*page + h*(N*data_dim) + n*data_dim + g*8
    scale[b, h, n, g]  = side_base + H*N*data_dim
                          + b*page + h*(N*scale_dim) + n*scale_dim + g

where ``g`` indexes the 16-dim groups, ``N`` the block size, and the V side
base is ``H*N*F`` past the K side base. On SM70 both the K and V scales are
stored linearly (the store kernel disables the trtllm-gen V-scale swizzle
for compute capability major == 7).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _nvfp4_e2m1_decode(x):
    """Decode fp4 e2m1 nibbles to float (vectorized over a tensor of u8)."""
    i = x & 7
    m = tl.where(
        i == 1,
        0.5,
        tl.where(
            i == 2,
            1.0,
            tl.where(
                i == 3,
                1.5,
                tl.where(
                    i == 4,
                    2.0,
                    tl.where(i == 5, 3.0, tl.where(i == 6, 4.0, tl.where(i == 7, 6.0, 0.0))),
                ),
            ),
        ),
    )
    return tl.where((x & 8) != 0, -m, m)


@triton.jit
def _nvfp4_e4m3_decode(x):
    """Decode one fp8 e4m3 scale byte to float (scalar or tensor of u8)."""
    sgn = (x & 0x80) != 0
    e = (x >> 3) & 0xF
    m = x & 7
    ef = e.to(tl.float32)
    mf = m.to(tl.float32)
    val = tl.where(e == 0, mf * 0.001953125, (1.0 + mf * 0.125) * tl.math.exp2(ef - 7.0))
    return tl.where(sgn, -val, val)


@triton.jit
def _sm70_nvfp4_dequant_side_kernel(
    kv_ptr,  # uint8 raw [B, 2H, N, F] (opaque page bytes)
    bt_ptr,  # int32 [num_reqs, max_blocks]
    sreq_ptr,  # int32 [B_s] shadow block -> request id
    st0_ptr,  # int32 [B_s] shadow block -> first token index
    out_ptr,  # fp16 [B_s, H, N, 2D] (K = [:D], V = [D:])
    side,  # 0 = K, 1 = V (runtime)
    scale,  # f32 scalar: layer _k_scale_float / _v_scale_float
    page,  # i64 block stride in bytes
    side_off,  # i64 V side base offset from K side base (H * N * F)
    bt_cols,  # i32 block_table row width
    H: tl.constexpr,  # num kv heads
    D: tl.constexpr,  # head dim
    N: tl.constexpr,  # block size
):
    """One program per (shadow block, kv head, token in block).

    Reads one contiguous ``D//2``-byte data row (16 nibble groups) and the
    16 e4m3 scale bytes, writes one contiguous 2D-wide fp16 row.
    """
    DD: tl.constexpr = D // 2  # data bytes per token row
    SD: tl.constexpr = D // 16  # scale bytes per token row
    G: tl.constexpr = D // 16  # 16-dim scale groups per head

    pid_b = tl.program_id(0)  # shadow block
    pid_h = tl.program_id(1)  # kv head
    n = tl.program_id(2)  # token within the shadow block

    r = tl.load(sreq_ptr + pid_b)
    t0 = tl.load(st0_ptr + pid_b)
    b = tl.load(bt_ptr + r * bt_cols + t0 // N).to(tl.int64)

    side_base = (side * side_off).to(tl.int64)
    page64 = page.to(tl.int64)
    data_base = (
        side_base
        + b * page64
        + pid_h.to(tl.int64) * (N * DD)
        + n.to(tl.int64) * DD
    )
    raw = tl.load(kv_ptr + data_base + tl.arange(0, DD))  # [DD] u8
    scale_base = (
        side_base
        + H * (N * DD)
        + b * page64
        + pid_h.to(tl.int64) * (N * SD)
        + n.to(tl.int64) * SD
    )
    sf = _nvfp4_e4m3_decode(
        tl.load(kv_ptr + scale_base + tl.arange(0, SD))
    ).to(tl.float32) * scale  # [G]

    # Dim d lives in byte d//2 (low nibble = even d, high = odd); its scale
    # group is d//16. (Explicit per-dim gather; the join/trans interleave
    # form is miscompiled on sm_70.)
    d16 = tl.arange(0, D)
    nib = tl.where(
        (d16 % 2) == 1, tl.gather(raw, d16 // 2, 0) >> 4,
        tl.gather(raw, d16 // 2, 0),
    ) & 0x0F
    vals = _nvfp4_e2m1_decode(nib).to(tl.float32)
    sf2 = tl.gather(sf, d16 // 16, 0)
    out16 = vals * sf2

    out_off = (
        pid_b.to(tl.int64) * (H * N * 2 * D)
        + pid_h * (N * 2 * D)
        + n * (2 * D)
        + side * D
        + d16
    )
    tl.store(out_ptr + out_off, out16.to(tl.float16))


def _shadow_block_maps(
    B_s: int, kv_lens: torch.Tensor, N: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-shadow-block (request id, first token index) maps, cached per shape.

    Shadow blocks are allotted contiguously per request: request r owns
    ``ceil(kv_len_r / N)`` blocks starting at ``indptr[r]`` (identity block
    table, see ``sm70_shadow_plan_tensors``).
    """
    if not hasattr(_shadow_block_maps, "_cache"):
        _shadow_block_maps._cache = {}
    key = (B_s, tuple(kv_lens.tolist()), N)
    hit = _shadow_block_maps._cache.get(key)
    if hit is not None and hit[0].device == device:
        return hit
    if len(_shadow_block_maps._cache) > 64:
        _shadow_block_maps._cache.clear()
    num_reqs = kv_lens.shape[0]
    sreq = torch.zeros(B_s, dtype=torch.int32)
    st0 = torch.zeros(B_s, dtype=torch.int32)
    base = 0
    for r in range(num_reqs):
        nb = (int(kv_lens[r].item()) + N - 1) // N
        if nb > 0:
            sreq[base : base + nb] = r
            st0[base : base + nb] = torch.arange(nb, dtype=torch.int32) * N
        base += nb
    entry = (sreq.to(device), st0.to(device))
    _shadow_block_maps._cache = {key: entry}
    return entry


def sm70_nvfp4_paged_dequant(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    out: torch.Tensor,
    k_scale: float,
    v_scale: float,
) -> None:
    """Dequantize paged NVFP4 KV into an fp16 shadow paged buffer.

    Args:
        kv_cache: raw uint8 layer cache ``[B, 2*H_kv, N, F]`` (F = D/2 + D/16),
            head-major block interior (LBHNC).
        block_table: int32 ``[num_reqs, max_blocks]`` (device), the primary
            nvfp4 cache block table for this step.
        kv_lens: int32 ``[num_reqs]`` per-request KV lengths.
        out: fp16 shadow buffer ``[B_s, H_kv, N, 2*D]`` (K in ``[0:D]``,
            V in ``[D:2D]``); ``B_s`` must equal the total shadow block
            allotment of ``sm70_shadow_plan_tensors(kv_lens, N)``.
        k_scale, v_scale: the layer's ``_k_scale_float`` / ``_v_scale_float``
            (the same floats the store kernel received).
    """
    _, two_h, N, F = kv_cache.shape
    H = two_h // 2
    D = 2 * (F * 8 // 9)
    page = kv_cache.stride(0)

    B_s = out.shape[0]
    sreq, st0 = _shadow_block_maps(B_s, kv_lens, N, kv_cache.device)

    grid = (B_s, H, N)
    for side, scale in ((0, k_scale), (1, v_scale)):
        _sm70_nvfp4_dequant_side_kernel[grid](
            kv_cache,
            block_table,
            sreq,
            st0,
            out,
            side=side,
            scale=scale,
            page=page,
            side_off=H * N * F if side else 0,
            bt_cols=block_table.shape[1],
            H=H,
            D=D,
            N=N,
            num_warps=4,
        )


@triton.jit
def _sm70_nvfp4_stage_nhd_side_kernel(
    kv_ptr,  # uint8 raw [B, 2H, N, F] (opaque page bytes)
    bt_ptr,  # int32 [M] the request's block ids in token order
    out_ptr,  # fp16 [M, N, H, D] (NHD: block, token, head, dim)
    side,  # 0 = K, 1 = V (runtime)
    scale,  # f32 scalar: layer _k_scale_float / _v_scale_float
    page,  # i64 block stride in bytes
    side_off,  # i64 V side base offset from K side base (H * N * F)
    H: tl.constexpr,  # num kv heads
    D: tl.constexpr,  # head dim
    N: tl.constexpr,  # block size
):
    """NHD staging variant of the side kernel: one program per
    (request block, kv head, token in block), writing ``out[b, n, h, :]``.

    Same raw-byte decode as the shadow dequant; only the output addressing
    changes (HND shadow -> NHD workspace for the splitd paged kernel).
    """
    DD: tl.constexpr = D // 2  # data bytes per token row
    SD: tl.constexpr = D // 16  # scale bytes per token row

    pid_b = tl.program_id(0)  # request block
    pid_h = tl.program_id(1)  # kv head
    n = tl.program_id(2)  # token within the block

    b = tl.load(bt_ptr + pid_b).to(tl.int64)

    side_base = (side * side_off).to(tl.int64)
    page64 = page.to(tl.int64)
    data_base = (
        side_base
        + b * page64
        + pid_h.to(tl.int64) * (N * DD)
        + n.to(tl.int64) * DD
    )
    raw = tl.load(kv_ptr + data_base + tl.arange(0, DD))  # [DD] u8
    scale_base = (
        side_base
        + H * (N * DD)
        + b * page64
        + pid_h.to(tl.int64) * (N * SD)
        + n.to(tl.int64) * SD
    )
    sf = _nvfp4_e4m3_decode(
        tl.load(kv_ptr + scale_base + tl.arange(0, SD))
    ).to(tl.float32) * scale  # [G]

    d16 = tl.arange(0, D)
    nib = tl.where(
        (d16 % 2) == 1, tl.gather(raw, d16 // 2, 0) >> 4,
        tl.gather(raw, d16 // 2, 0),
    ) & 0x0F
    vals = _nvfp4_e2m1_decode(nib).to(tl.float32)
    sf2 = tl.gather(sf, d16 // 16, 0)
    out16 = vals * sf2

    out_off = (
        pid_b.to(tl.int64) * (N * H * D)
        + n.to(tl.int64) * (H * D)
        + pid_h.to(tl.int64) * D
        + d16
    )
    tl.store(out_ptr + out_off, out16.to(tl.float16))


def sm70_nvfp4_paged_stage_nhd(
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    kv_len: int,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    k_scale: float,
    v_scale: float,
) -> None:
    """Stage one request's NVFP4 paged KV into NHD fp16 workspaces.

    Feeds the SM70 splitd paged prefill kernel: K and V as separate
    ``[M, N, H_kv, D]`` fp16 tensors (NHD) plus the request's block-table row.

    Args:
        kv_cache: raw uint8 layer cache ``[B, 2*H_kv, N, F]`` (LBHNC).
        block_table_row: int32 ``[M]`` (device) — the request's physical
            block ids in token order (a row of its block table).
        kv_len: live token count; ``M = k_out.size(0)`` must satisfy
            ``M * N >= kv_len`` (the splitd kernel requires kv_len % 32 == 0
            and N == 16, so production calls have no partial blocks).
        k_out: fp16 ``[M, N, H_kv, D]`` preallocated K workspace.
        v_out: fp16 ``[M, N, H_kv, D]`` preallocated V workspace.
        k_scale, v_scale: the layer's ``_k_scale_float`` / ``_v_scale_float``.
    """
    _, two_h, N, F = kv_cache.shape
    H = two_h // 2
    D = 2 * (F * 8 // 9)
    page = kv_cache.stride(0)
    M = k_out.shape[0]
    assert M * N >= kv_len, (
        f"stage workspace smaller than kv_len: M={M} N={N} "
        f"M*N={M * N} kv_len={kv_len}"
    )
    # Stage only the blocks that hold real tokens; any padded workspace
    # rows (M > nb) keep their pre-allocated (zero) values.
    nb = (kv_len + N - 1) // N
    assert nb <= M, "stage workspace has fewer blocks than kv_len needs"
    grid = (nb, H, N)
    for out, side, scale in ((k_out, 0, k_scale), (v_out, 1, v_scale)):
        _sm70_nvfp4_stage_nhd_side_kernel[grid](
            kv_cache,
            block_table_row,
            out,
            side=side,
            scale=scale,
            page=page,
            side_off=H * N * F if side else 0,
            H=H,
            D=D,
            N=N,
            num_warps=4,
        )


# CUDA-graph decode: blocks per program in the fixed-grid graph kernel.
_GRAPH_BLKS = 16


@triton.jit
def _sm70_nvfp4_dequant_graph_kernel(
    kv_ptr,  # uint8 raw [B, 2H, N, F]
    bt_ptr,  # int32 [num_reqs, max_blocks]
    kvlens_ptr,  # int32 [num_reqs] device KV lengths
    out_ptr,  # fp16 [num_reqs*max_blocks, H, N, 2D]
    k_scale,  # f32 scalar: layer _k_scale_float
    v_scale,  # f32 scalar: layer _v_scale_float
    page,  # i64 block stride in bytes
    side_off,  # i64 V side base offset from K side base (H * N * F)
    bt_cols,  # i32 block_table row width
    max_blocks,  # i32 uniform per-request shadow block allotment
    num_chunks,  # i32 ceil(max_blocks / BLKS)
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    BLKS: tl.constexpr,
):
    """CUDA-graph-safe variant: uniform per-request block allotment.

    One program per (request, block chunk, kv head, token in block) with a
    fixed grid for the captured graph. Shadow block ``s = r*max_blocks +
    b_local``; blocks at or past the request's live kv_len are skipped
    (kv_len changes across replays, so the bound must be read on device).
    K and V are dequantized in the same launch.
    """
    DD: tl.constexpr = D // 2
    SD: tl.constexpr = D // 16

    pid_c = tl.program_id(0)
    r = pid_c // num_chunks
    b0 = (pid_c % num_chunks) * BLKS
    pid_h = tl.program_id(1)
    n = tl.program_id(2)

    kv_len_r = tl.load(kvlens_ptr + r)
    nb = (kv_len_r + N - 1) // N
    if b0 >= nb:
        return

    page64 = page.to(tl.int64)
    d16 = tl.arange(0, D)
    row_stride = H * N * 2 * D

    for side in tl.static_range(2):
        scale = k_scale if side == 0 else v_scale
        side_base = (side * side_off).to(tl.int64)
        out_row = (
            (r * max_blocks).to(tl.int64) * row_stride
            + pid_h * (N * 2 * D)
            + n * (2 * D)
            + side * D
        )
        for bi in range(BLKS):
            b_local = b0 + bi
            active = b_local < nb
            # Inactive iterations clamp to block 0 (always in range) so the
            # loads stay valid; their stores are masked out.
            b = tl.where(
                active, tl.load(bt_ptr + r * bt_cols + b_local).to(tl.int64), 0
            )
            data_base = (
                side_base
                + b * page64
                + pid_h.to(tl.int64) * (N * DD)
                + n.to(tl.int64) * DD
            )
            raw = tl.load(kv_ptr + data_base + tl.arange(0, DD))
            scale_base = (
                side_base
                + H * (N * DD)
                + b * page64
                + pid_h.to(tl.int64) * (N * SD)
                + n.to(tl.int64) * SD
            )
            sf = _nvfp4_e4m3_decode(
                tl.load(kv_ptr + scale_base + tl.arange(0, SD))
            ).to(tl.float32) * scale
            nib = tl.where(
                (d16 % 2) == 1, tl.gather(raw, d16 // 2, 0) >> 4,
                tl.gather(raw, d16 // 2, 0),
            ) & 0x0F
            vals = _nvfp4_e2m1_decode(nib).to(tl.float32)
            sf2 = tl.gather(sf, d16 // 16, 0)
            out16 = (vals * sf2).to(tl.float16)
            tl.store(
                out_ptr + out_row + b_local.to(tl.int64) * row_stride + d16,
                out16,
                mask=active,
            )


def sm70_nvfp4_graph_dequant(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    out: torch.Tensor,
    k_scale: float,
    v_scale: float,
    num_reqs: int,
    max_blocks: int,
) -> None:
    """Graph-capture-safe NVFP4 -> fp16 shadow dequant (uniform allotment).

    Fixed grid ``(num_reqs * ceil(max_blocks/BLKS), H, N)`` so the launch is
    identical across replays of the same captured graph; per-request live
    length bounds the work via the device kv-lens buffer.

    Args:
        kv_cache: raw uint8 layer cache ``[B, 2*H_kv, N, F]`` (LBHNC).
        block_table: int32 ``[num_reqs, max_blocks]`` device block table.
        kv_lens: int32 ``[num_reqs]`` **device** per-request KV lengths.
        out: fp16 shadow buffer ``[num_reqs*max_blocks, H_kv, N, 2*D]``;
            request r owns shadow blocks ``[r*max_blocks, (r+1)*max_blocks)``.
        k_scale, v_scale: the layer's ``_k_scale_float`` / ``_v_scale_float``.
        num_reqs: number of requests (captured batch size, incl. padding).
        max_blocks: uniform per-request shadow block allotment.
    """
    _, two_h, N, F = kv_cache.shape
    H = two_h // 2
    D = 2 * (F * 8 // 9)
    page = kv_cache.stride(0)
    num_chunks = (max_blocks + _GRAPH_BLKS - 1) // _GRAPH_BLKS
    grid = (num_reqs * num_chunks, H, N)
    _sm70_nvfp4_dequant_graph_kernel[grid](
        kv_cache,
        block_table,
        kv_lens,
        out,
        k_scale=k_scale,
        v_scale=v_scale,
        page=page,
        side_off=H * N * F,
        bt_cols=block_table.shape[1],
        max_blocks=max_blocks,
        num_chunks=num_chunks,
        H=H,
        D=D,
        N=N,
        BLKS=_GRAPH_BLKS,
        num_warps=4,
    )


def sm70_shadow_plan_tensors(
    kv_lens: torch.Tensor, N: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Wrapper plan tensors for the fp16 shadow paged buffer.

    Returns ``(paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
    seq_lens, num_shadow_blocks)``; all tensors are CPU int32, matching the
    block allotment of ``sm70_nvfp4_paged_dequant`` (contiguous per-request
    identity block table).
    """
    kv_lens = kv_lens.to(dtype=torch.int32)
    num_reqs = kv_lens.shape[0]
    blocks_per_req = (kv_lens + N - 1) // N
    indptr = torch.zeros(num_reqs + 1, dtype=torch.int32)
    torch.cumsum(blocks_per_req, dim=0, out=indptr[1:])
    B_s = int(indptr[-1].item())
    indices = torch.arange(B_s, dtype=torch.int32)
    last_page_len = kv_lens % N
    last_page_len = torch.where(last_page_len == 0, N, last_page_len)
    return indptr, indices, last_page_len, kv_lens, B_s
