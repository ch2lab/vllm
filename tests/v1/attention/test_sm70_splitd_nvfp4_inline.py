# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the VLLM project

"""A/B + oracle test for the SM70 NVFP4 in-kernel-dequant splitd paged op.

Compares the new ``sm70_d256_splitd_n32_nvfp4_paged_fwd`` (kernel reads the
packed NVFP4 cache directly, decoding in-kernel) against the production
staged path (Triton ``sm70_nvfp4_paged_stage_nhd`` -> fp16 NHD workspaces ->
``sm70_d256_splitd_n32_paged_fwd``):

1. A == B: identical outputs (both decode the same bytes to the same fp16
   K/V, so the GEMMs consume identical inputs and match bit-for-bit).
2. FP64 oracle over the Python-decoded KV catches layout/scale bugs that
   would be common to both compared paths.
"""

import pytest
import torch

from vllm.v1.attention.ops.sm70_nvfp4_dequant import sm70_nvfp4_paged_stage_nhd


def _load_ops():
    import vllm.vllm_flash_attn  # noqa: F401 (loads the C library)

    paged = torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_paged_fwd
    nvfp4 = torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_nvfp4_paged_fwd
    return paged, nvfp4


# ---------------------------------------------------------------------------
# Python mirror of the NVFP4 dequant (same semantics as the in-kernel and
# Triton decodes): x_hat = e2m1(nib) * e4m3(scale_byte) * layer_scale.

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e2m1(nib: torch.Tensor) -> torch.Tensor:
    """Decode e2m1 nibbles (u8, low 3 bits = mantissa, bit 3 = sign)."""
    idx = (nib & 7).long()
    mag = _E2M1.to(nib.device)[idx]
    sign = torch.where((nib & 8) != 0, -1.0, 1.0)
    return mag * sign


def _e4m3(b: torch.Tensor) -> torch.Tensor:
    """Decode fp8 e4m3 scale bytes (u8)."""
    ef = ((b >> 3) & 0xF).float()
    mf = (b & 7).float()
    val = torch.where(
        ef == 0, mf * 0.001953125, (1.0 + mf * 0.125) * torch.exp2(ef - 7.0)
    )
    sign = torch.where((b & 0x80) != 0, -1.0, 1.0)
    return val * sign


def _packed_offsets(nb, h_kv, N, F, SD, side, device):
    """Flat byte offsets of one token's data/scale rows in the packed layout.

    The physical cache is NOT contiguous per (token) row: each block
    (page = 2*H*N*F bytes) holds, per side, all data then all scale:
      data  [b, h, n, f] = b*page + s*(H*N*F) + h*(N*DD) + n*DD + f
      scale [b, h, n, g] = b*page + s*(H*N*F) + H*(N*DD) + h*(N*SD) + n*SD + g
    """
    DD = F - SD
    H = h_kv
    page = 2 * H * N * F
    L = nb * N
    tok = torch.arange(L, device=device)
    b = tok // N
    n = tok % N
    data_off = b * page + side * (H * N * F) + n * DD
    scale_off = b * page + side * (H * N * F) + H * (N * DD) + n * SD
    return DD, page, data_off, scale_off


def _build_cache(nb, h_kv, N, F, SD, device, seed):
    """Random packed NVFP4 cache [nb, 2H, N, F] (flat buffer, packed layout)."""
    H = h_kv
    page = 2 * H * N * F
    g = torch.Generator().manual_seed(seed)
    flat = torch.randint(
        0, 256, (nb * page,), generator=g, dtype=torch.uint8
    ).to(device)
    return flat.view(nb, 2 * H, N, F)


def _decode_side(cache, h_kv, N, F, SD, side, scale):
    """Decode one side of the cache to fp16 [L, h_kv, D] (Python mirror of
    the in-kernel / Triton dequant over the true packed layout)."""
    DD, page, data_off, scale_off = _packed_offsets(
        cache.size(0), h_kv, N, F, SD, side, cache.device
    )
    D = DD * 2
    assert D // 16 == SD
    flat = cache.reshape(-1)
    L = cache.size(0) * N
    out = torch.empty(L, h_kv, D, dtype=torch.float16, device=cache.device)
    d_arange = torch.arange(DD, device=cache.device)
    s_arange = torch.arange(SD, device=cache.device)
    for h in range(h_kv):
        data = flat[
            (data_off + h * (N * DD)).view(-1, 1) + d_arange.view(1, -1)
        ]
        lo = data & 0x0F
        hi = data >> 4
        nib = torch.stack([lo, hi], dim=-1).reshape(L, D)
        sf = _e4m3(
            flat[(scale_off + h * (N * SD)).view(-1, 1) + s_arange.view(1, -1)]
        )
        val = _e2m1(nib) * sf.repeat_interleave(D // 16, dim=1) * scale
        out[:, h].copy_(val.half())
    return out


def _ref_causal_gqa(q64, k64, v64, scale):
    """FP64 GQA reference: q [1, T, Hq, D], k/v [L, Hkv, D].

    Chunked-prefill semantics: q row t attends kv rows 0 .. (L - T) + t.
    """
    Hq, Hkv, T, L = q64.shape[2], k64.shape[1], q64.shape[1], k64.shape[0]
    t_all = torch.arange(T, device=q64.device)
    l = torch.arange(L, device=q64.device)[None, :]
    out = torch.empty_like(q64)
    for h in range(Hq):
        hk = h // (Hq // Hkv)
        k_h = k64[:, hk, :]  # [L, D]
        v_h = v64[:, hk, :]
        for t0 in range(0, T, 256):
            t1 = min(t0 + 256, T)
            q_h = q64[0, t0:t1, h, :]
            t = t_all[t0:t1]
            qk = (q_h @ k_h.T) * scale
            mask = l <= (L - T) + t[:, None]
            p = torch.softmax(qk.masked_fill(~mask, float("-inf")), dim=-1)
            out[0, t0:t1, h, :] = p @ v_h
    return out


@pytest.mark.parametrize(
    "T,L,shuffle,N",
    [
        (128, 512, False, 16),
        (128, 512, True, 16),
        (4096, 8192, False, 16),
        # page=64: the production NVFP4 KV cache page size (the in-kernel
        # loader is page-size agnostic; regression test for the page gate).
        (512, 2048, False, 64),
        (512, 2048, True, 64),
    ],
    ids=[
        "q128-kv512",
        "q128-kv512-shuffled",
        "q4096-kv8192",
        "q512-kv2048-page64",
        "q512-kv2048-page64-shuffled",
    ],
)
def test_nvfp4_inline_matches_staged_and_oracle(T, L, shuffle, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    paged_op, nvfp4_op = _load_ops()

    Hq, h_kv, D = 12, 2, 256
    F = D // 2 + D // 16  # 144
    scale = 1.0 / 16.0
    k_scale, v_scale = 0.003, 0.004
    nb = L // N
    assert L % 32 == 0
    assert L % N == 0

    SD = D // 16
    device = torch.device("cuda:0")
    g = torch.Generator(device="cpu").manual_seed(11)
    q = torch.randn(
        1, T, Hq, D, generator=g, dtype=torch.float16
    ).to(device)
    cache = _build_cache(nb, h_kv, N, F, SD, device, seed=12)

    T_pad = ((T + 63) // 64) * 64
    q_buf = torch.zeros(1, T_pad, Hq, D, dtype=torch.float16, device=device)
    q_buf[:, :T].copy_(q)

    # Block table (physical block ids in token order).
    if shuffle:
        order = torch.randperm(nb, generator=torch.Generator().manual_seed(13))
    else:
        order = torch.arange(nb)
    bt_row = order.to(torch.int32).to(device)
    bt_2d = bt_row.unsqueeze(0)
    # The staging kernel resolves physical blocks into logical order
    # (k_ws[logical_block]); the paged op therefore needs an identity block
    # table, not the shuffled one (which would double-apply the mapping).
    id_bt_2d = torch.arange(
        nb, dtype=torch.int32, device=device
    ).unsqueeze(0)

    # --- Path A: production staged path (Triton dequant + fp16 paged op) ---
    k_ws = torch.zeros(nb, N, h_kv, D, dtype=torch.float16, device=device)
    v_ws = torch.zeros_like(k_ws)
    sm70_nvfp4_paged_stage_nhd(
        cache, bt_row, L, k_ws, v_ws, k_scale, v_scale
    )
    out_a = torch.empty(1, T_pad, Hq, D, dtype=torch.float16, device=device)
    paged_op(q_buf, k_ws, v_ws, id_bt_2d, out_a, L, T, scale, True)

    # --- Path B: new in-kernel-dequant op ---
    out_b = torch.empty(1, T_pad, Hq, D, dtype=torch.float16, device=device)
    nvfp4_op(
        q_buf, cache, bt_row, out_b, L, T, k_scale, v_scale, scale, True
    )

    # A/B must match bit-for-bit (same decoded fp16 K/V -> same GEMMs).
    ab_diff = (out_a[:, :T].float() - out_b[:, :T].float()).abs().max().item()
    assert torch.isfinite(out_b[:, :T]).all()
    assert ab_diff == 0.0, f"staged vs inline max diff {ab_diff}"

    # --- Oracle: FP64 attention over the Python-decoded KV ---
    # Decode in logical order (apply the block table), so a shuffled
    # physical layout is re-ordered before the FP64 reference.
    cache_logical = cache[bt_row.long()]
    k64 = _decode_side(cache_logical, h_kv, N, F, SD, 0, k_scale)
    v64 = _decode_side(cache_logical, h_kv, N, F, SD, 1, v_scale)
    ref = _ref_causal_gqa(q.double(), k64.double(), v64.double(), scale)
    err_a = (out_a[:, :T].double() - ref).abs().max().item()
    err_b = (out_b[:, :T].double() - ref).abs().max().item()
    tol = 2e-2
    assert err_a < tol, f"staged vs oracle max err {err_a}"
    assert err_b < tol, f"inline vs oracle max err {err_b}"
