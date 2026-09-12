"""Unit test for _nvfp4_d256_prefill (SM120 NVFP4-KV D256 prefill).

Calls the module function directly with synthetic paged-cache tensors (no
engine): it must gather each request's NVFP4 blocks in block-table order,
convert to the QK MXFP8 / PV NVFP4 kernel inputs, run the dense kernel, and
match exact float32 causal GQA attention over the dequantized KV.

The raw cache is fabricated in the cache's *physical* layout: within each
side of a page the bytes are a head-major data region [Hk·N·(D//2)] followed
by a head-major scale region [Hk·N·(D//16)] — the layout
``nvfp4_split_data_scale`` reads back for the FA2 decode path. It is *not*
per-row [data | scale]: the hook must re-assemble the interleaved rows.

The KV content is structured (randn quantized into the NVFP4 format), not
uniform random bytes: with noise data the softmax top-key margins fall under
the fp8/fp4 requantization noise and the T2 precision gate (cos >= 0.99) is
unsatisfiable by any correct kernel.
"""

import types

import torch

from vllm.v1.attention.backends.flashinfer import (
    FIPrefill,
    _nvfp4_d256_prefill,
)

_E2M1_LUT = torch.tensor(
    [0, .5, 1, 1.5, 2, 3, 4, 6,
     -0., -.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def _torch_dequant_side(side):
    """Dequant one NVFP4 side in the physical layout (B, H, N, F) uint8.

    Within each block's side the flat bytes are a head-major data region
    [H·N·(D//2)] followed by a head-major scale region [H·N·(D//16)];
    data(h,s) lives at h·N·(D//2) + s·(D//2), sf(h,s) at the side base +
    H·N·(D//2) + h·N·(D//16) + s·(D//16).  Returns float32 (B, H, N, D).
    """
    B, H, N, F = side.shape
    D = 2 * (F * 8 // 9)
    data_dim = F * 8 // 9
    scale_dim = F - data_dim
    flat = side.reshape(B, -1)                    # (B, H*N*F), blocks contiguous
    data = flat[:, :H * N * data_dim].view(B, H, N, data_dim)
    sf = (flat[:, H * N * data_dim:]
          .view(B, H, N, scale_dim)
          .view(torch.float8_e4m3fn).float())
    lut = _E2M1_LUT.to(side.device)
    lo = lut[(data & 0x0F).long()]
    hi = lut[((data >> 4) & 0x0F).long()]
    per_dim = torch.stack([lo, hi], dim=-1).reshape(B, H, N, D)
    return per_dim * sf.repeat_interleave(D // 16, dim=-1)


def _nvfp4_data_scale(x):
    """Quantize (H, S, D) float into NVFP4 data/scale byte grids.

    Per 16-dim group: e4m3fn scale = amax / 6 (keeps every |dim / scale|
    within the e2m1 range), e2m1 data rounded to the nearest code. Returns
    (data (H, S, D//2) uint8, scale (H, S, D//16) uint8 as e4m3fn bits).
    """
    H, S, D = x.shape
    g = x.float().reshape(H, S, D // 16, 16)
    amax = g.abs().amax(dim=-1, keepdim=True)
    sf = (amax / 6.0).to(torch.float8_e4m3fn)
    sf_f = sf.float().view(H, S, D // 16, 1)
    sf_f = sf_f.where(sf_f > 0, torch.ones_like(sf_f))
    codes = ((g / sf_f).unsqueeze(-1) - _E2M1_LUT.to(x.device)).abs().argmin(-1)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return (packed.reshape(H, S, D // 2),
            sf.view(torch.uint8).reshape(H, S, D // 16))


def _side_blocks(data_grid, sf_grid, nblocks):
    """Lay per-block side bytes (nblocks, H, N, F) in the physical order:
    head-major data region then head-major scale region (token-minor)."""
    H, S, data_dim = data_grid.shape
    scale_dim = sf_grid.shape[2]
    N = S // nblocks
    out = torch.empty(
        nblocks, H, N, data_dim + scale_dim, dtype=torch.uint8,
        device=data_grid.device)
    for b in range(nblocks):
        flat = torch.cat([
            data_grid[:, b * N:(b + 1) * N].flatten(),
            sf_grid[:, b * N:(b + 1) * N].flatten(),
        ])
        out[b] = flat.view(H, N, data_dim + scale_dim)
    return out


def test_nvfp4_d256_prefill_matches_fa2_semantics():
    torch.manual_seed(3)
    Hq, Hk, D, page = 12, 2, 256, 16
    F = D // 2 + D // 16
    S, M = 3 * page + 7, 5       # kv_len=55 (last block partial), q_len=5
    nblocks = (S + page - 1) // page
    block_tables = torch.zeros(1, nblocks, dtype=torch.int32, device="cuda")
    block_tables[0, :3] = torch.arange(1, 4, device="cuda")
    tables = block_tables[0]     # physical blocks [1, 2, 3, 0]
    # Cache holds garbage everywhere; only the sides the request reads are
    # written with structured NVFP4 data (unused rows must not leak out).
    kv = torch.randint(0, 255, (8, 2 * Hk, page, F), dtype=torch.uint8,
                       device="cuda")
    kv_dense = torch.randn(2, Hk, nblocks * page, D, device="cuda",
                           dtype=torch.bfloat16)
    for off, dense in ((0, kv_dense[0]), (Hk, kv_dense[1])):
        data_grid, sf_grid = _nvfp4_data_scale(dense)
        kv[tables, off:off + Hk] = _side_blocks(data_grid, sf_grid, nblocks)
    pre = FIPrefill(
        wrapper=None,
        block_tables=block_tables,
        seq_lens=torch.tensor([S], device="cuda"),
        cum_seq_lens_q=torch.tensor([0, M], device="cuda"),
    )
    prefill_query = torch.randn(M, Hq, D, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(M, Hq, D, device="cuda", dtype=torch.bfloat16)
    impl = types.SimpleNamespace(
        page_size=page, num_kv_heads=Hk, num_heads=Hq, head_size=D,
        scale=D ** -0.5)
    scales = (1.0, 1.0, 1.0)
    _nvfp4_d256_prefill(impl, prefill_query, kv, pre, scales, out)
    torch.cuda.synchronize()
    # Reference: gather the same physical blocks (block-table order),
    # dequant KV per head from the physical data/scale regions, exact causal
    # GQA attention in float32.
    kv_sel = kv[tables[:nblocks]]
    k_ref = _torch_dequant_side(kv_sel[:, :Hk]).permute(1, 0, 2, 3)
    v_ref = _torch_dequant_side(kv_sel[:, Hk:]).permute(1, 0, 2, 3)
    k_exp = k_ref.reshape(Hk, -1, D)[:, :S].repeat_interleave(Hq // Hk, dim=0)
    v_exp = v_ref.reshape(Hk, -1, D)[:, :S].repeat_interleave(Hq // Hk, dim=0)
    q = prefill_query.transpose(0, 1).float()
    scores = q @ k_exp.transpose(1, 2) * impl.scale
    mask = torch.triu(torch.ones(M, S, device="cuda", dtype=torch.bool),
                      diagonal=S - M + 1)
    scores.masked_fill_(mask, float("-inf"))
    ref = torch.softmax(scores, dim=-1) @ v_exp          # (Hq, M, D)
    got = out.transpose(0, 1).float()
    cos = (got.flatten() @ ref.flatten()) / \
        (got.norm() * ref.norm())
    rel = (got - ref).norm() / ref.norm()
    assert cos >= 0.99, f"cos={cos:.4f}"
    assert rel <= 0.2, f"rel={rel:.4f}"
