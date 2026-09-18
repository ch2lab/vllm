#!/usr/bin/env python3
"""sm70_flashdecode unit tests (standalone).

Run: python3 test_flashdecode_unit.py -xvs

Covers:
- dequant LUTs vs the Triton implementation (GPU cross-check, known-good)
- vectorized reference self-consistency vs a naive byte-level reference
- causal q-row invariant (verify rows vs truncated q=1 references)
- the CUDA op itself (RED until plan Task 2 builds sm70_flashdecode.cu)
"""

import numpy as np
import pytest
import torch

from vllm.triton_utils import tl, triton

from vllm.v1.attention.ops.sm70_flashdecode import (
    _e2m1_vec,
    _e4m3_vec,
    sm70_flashdecode_op,
    sm70_nvfp4_reference_attention,
)
from vllm.v1.attention.ops.sm70_nvfp4_dequant import (
    _nvfp4_e2m1_decode,
    _nvfp4_e4m3_decode,
)

HQ = 12
H = 2
D = 256
N = 16
B = 64
DD = D // 2
SD = D // 16
F = DD + SD


def make_case(R=2, Q=1, seed=0, kv_lens=(275, 33), sf_max=31, page=N):
    N_ = page
    g = torch.Generator().manual_seed(seed)
    kv = torch.randint(0, 255, (B, 2 * H, N_, F), dtype=torch.uint8,
                       generator=g)
    # Scale bytes: clamp the physical scale region to <= sf_max. Default
    # sf_max=31 (e4m3 e <= 3) keeps the op tests' scores in an fp32-friendly
    # range; the full e4m3 range (e <= 14) is covered by the LUT test and
    # test_op_kv1_exact_dequant. The 4D [B, 2H, N, F] view is a byte-count
    # fiction — physical layout is [data region | scale region] per side
    # (LBHNC), so address bytes directly: offset = b*S0 + side_base + H*N*DD
    # + h*(N*SD) + n*SD + g.
    S0 = kv.stride(0)
    flat = kv.flatten()
    idx = (
        np.arange(B)[:, None, None, None, None] * S0
        + np.array([0, H * N_ * F])[None, :, None, None, None]
        + np.arange(H)[None, None, :, None, None] * (N_ * SD)
        + np.arange(N_)[None, None, None, :, None] * SD
        + np.arange(SD)[None, None, None, None, :]
        + H * N_ * DD
    )
    flat[idx] = flat[idx] % (sf_max + 1)
    bt = torch.randperm(B, generator=g).view(R, -1)[: :, : 64].to(torch.int32)
    kv_lens_t = torch.tensor(list(kv_lens), dtype=torch.int32)
    q = torch.randn(R, Q, HQ, D, dtype=torch.float32, generator=g).half()
    return q, kv, bt, kv_lens_t


def _e2m1_scalar(x):
    mag = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0, 6: 4.0, 7: 6.0}
    v = mag[int(x) & 7]
    return -v if (int(x) & 8) else v


def _e4m3_scalar(x):
    b = int(x)
    e = (b >> 3) & 0xF
    m = b & 7
    v = m * 0.001953125 if e == 0 else (1.0 + m * 0.125) * 2.0 ** (e - 7)
    return -v if (b & 0x80) else v


def _dequant_naive(kv_flat, b, h, n, d, side, scale, S0):
    """Literal docstring formula, one dim at a time (independent path)."""
    DD_ = D // 2
    SD_ = D // 16
    scale_off = H * N * DD_
    by = int(kv_flat[side + b * S0 + h * (N * DD_) + n * DD_ + d // 2])
    nib = (by >> 4) if d % 2 else (by & 0xF)
    sf_b = int(kv_flat[
        scale_off + side + b * S0 + h * (N * SD_) + n * SD_ + d // 16
    ])
    return _e2m1_scalar(nib) * _e4m3_scalar(sf_b) * scale


def _dequant_rows_naive(kv_cache, b_arr, h, toks, side, scale):
    kv_flat = kv_cache.flatten().numpy()
    S0 = kv_cache.stride(0)
    out = np.zeros((len(toks), D), dtype=np.float32)
    for i, t in enumerate(toks):
        b = int(b_arr[i])
        n = t % N
        for d in range(D):
            out[i, d] = _dequant_naive(kv_flat, b, h, n, d, side, scale, S0)
    return out


def test_luts_match_triton():
    """numpy LUTs vs the Triton LUTs (known-good, GPU run)."""
    @triton.jit
    def _lut_kernel(nib_ptr, sf_ptr, out1, out2, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        m1 = i < 16
        m2 = i < 248
        tl.store(
            out1 + i,
            _nvfp4_e2m1_decode(tl.load(nib_ptr + i, mask=m1, other=0)).to(
                tl.float32),
            mask=m1,
        )
        tl.store(
            out2 + i,
            _nvfp4_e4m3_decode(tl.load(sf_ptr + i, mask=m2, other=0)).to(
                tl.float32),
            mask=m2,
        )

    dev = "cuda"
    nib = torch.arange(16, dtype=torch.uint8, device=dev)
    sf = torch.arange(248, dtype=torch.uint8, device=dev)
    o1 = torch.zeros(16, dtype=torch.float32, device=dev)
    o2 = torch.zeros(248, dtype=torch.float32, device=dev)
    _lut_kernel[(1,)](nib, sf, o1, o2, BLOCK=256)
    assert np.allclose(o1.cpu().numpy(), _e2m1_vec(np.arange(16)),
                       atol=0, rtol=0)
    assert np.allclose(o2.cpu().numpy(), _e4m3_vec(np.arange(248)),
                       atol=1e-6 * np.abs(o2.cpu().numpy()).max(),
                       rtol=1e-6)


def test_reference_selfcheck():
    """Vectorized reference K/V dequant == naive byte-level dequant.

    Short kv_lens: the naive path is a pure-python byte loop (speed).
    """
    q, kv, bt, kv_lens = make_case(R=2, Q=1, seed=1, kv_lens=(41, 9))
    kv_flat = kv.flatten().numpy()
    S0 = kv.stride(0)
    for r in range(2):
        kv_len = int(kv_lens[r])
        toks = np.arange(kv_len)
        pages = bt[r, toks // N].numpy()
        for h in range(H):
            for side, scale in ((0, 1.0), (H * N * F, 1.0)):
                vec_rows = []
                # vectorized (same math as sm70_nvfp4_reference_attention)
                rows = toks % N
                db = pages * S0 + side + h * (N * DD) + rows * DD
                data = kv_flat[db[:, None] + np.arange(DD)[None, :]].astype(
                    np.uint8)
                nib = np.empty((kv_len, D), dtype=np.uint8)
                nib[:, 0::2] = data & 0x0F
                nib[:, 1::2] = data >> 4
                sb = (H * N * DD + pages * S0 + side + h * (N * SD)
                      + rows * SD)
                sf = kv_flat[sb[:, None] + np.arange(SD)[None, :]].astype(
                    np.uint8)
                vec = _e2m1_vec(nib) * np.repeat(_e4m3_vec(sf), 16,
                                                 axis=1) * scale
                naive = _dequant_rows_naive(kv, pages, h, toks, side, scale)
                assert np.allclose(vec, naive, atol=1e-7), (
                    f"r={r} h={h} side={side}")


def test_reference_empty_kv_is_zero():
    q, kv, bt, kv_lens = make_case(R=1, Q=1, seed=2, kv_lens=(0,))
    out = sm70_nvfp4_reference_attention(
        q, kv, bt, kv_lens, 1.0, 1.0, 1.0 / 16, -1)
    assert torch.all(out == 0)


def _tail_window_cache(kv, bt, kv_len, W):
    """Materialize the last W+1 tokens (window [kv_len-1-W, kv_len)) into a
    fresh contiguous cache: returns (kv2, bt2, kv_len2)."""
    lo = kv_len - 1 - W
    flat = kv.flatten().numpy()
    S0 = kv.stride(0)
    B2 = (W + 1 + N - 1) // N
    kv2 = torch.zeros(B2, 2 * H, N, F, dtype=torch.uint8)
    dst = kv2.flatten().numpy()
    pages = bt[0].numpy()
    for t in range(W + 1):
        s = lo + t
        sp = int(pages[s // N])
        dp = t // N
        for side in (0, H * N * F):
            for h in range(H):
                # data row (128B) + scale row (16B) per head
                ds = side + sp * S0 + h * (N * DD) + (s % N) * DD
                dt = side + dp * S0 + h * (N * DD) + (t % N) * DD
                dst[dt:dt + DD] = flat[ds:ds + DD]
                ds2 = (H * N * DD) + side + sp * S0 + h * (N * SD) + (s % N) * SD
                dt2 = (H * N * DD) + side + dp * S0 + h * (N * SD) + (t % N) * SD
                dst[dt2:dt2 + SD] = flat[ds2:ds2 + SD]
    bt2 = torch.arange(B2, dtype=torch.int32).view(1, -1)
    return kv2, bt2, torch.tensor([W + 1], dtype=torch.int32)


def test_reference_sliding_window():
    """window_left>=0: output equals no-window reference over a cache that
    holds exactly the attended tail span."""
    R, Q, seed = 1, 1, 3
    q, kv, bt, kv_lens = make_case(R=R, Q=Q, seed=seed, kv_lens=(60,))
    W = 32
    out_win = sm70_nvfp4_reference_attention(
        q, kv, bt, kv_lens, 1.0, 1.0, 1.0 / 16, W)
    kv2, bt2, kl2 = _tail_window_cache(kv, bt, 60, W)
    out_ref = sm70_nvfp4_reference_attention(
        q, kv2, bt2, kl2, 1.0, 1.0, 1.0 / 16, -1)
    assert torch.allclose(out_win, out_ref, atol=1e-5, rtol=1e-5)


def test_causality_q_rows():
    """Row qq of the Q-row call == q=1 reference with kv_len cut at the
    row's causal horizon kv_len - Q + qq + 1."""
    R, Q, seed = 1, 4, 4
    kv_len = 100
    q, kv, bt, kv_lens = make_case(R=R, Q=Q, seed=seed, kv_lens=(kv_len,))
    out_q = sm70_nvfp4_reference_attention(
        q, kv, bt, kv_lens, 1.0, 1.0, 1.0 / 16, -1)
    for qq in range(Q):
        hi = kv_len - Q + qq + 1
        out_1 = sm70_nvfp4_reference_attention(
            q[:, qq:qq + 1], kv, bt, torch.tensor([hi], dtype=torch.int32),
            1.0, 1.0, 1.0 / 16, -1)
        assert torch.allclose(out_q[:, qq:qq + 1], out_1, atol=1e-5,
                              rtol=1e-5), f"row {qq} horizon {hi}"


def test_op_q1_matches_reference():
    """R=2 decode (kv_lens 275/33) vs the numpy reference."""
    q, kv, bt, kv_lens = make_case(R=2, Q=1, seed=5)
    R, Q, HQ_, D_ = q.shape
    H_ = kv.shape[1] // 2
    splits = 4
    ws = torch.zeros(
        R * H_ * splits * Q * 6 * (D_ + 2), dtype=torch.float32,
        device="cuda")
    out = torch.empty_like(q, device="cuda")
    sm70_flashdecode_op(
        q.cuda(), kv.cuda(), bt.cuda(), kv_lens.cuda(), out, 1.0, 1.0,
        1.0 / 16, -1, splits=splits, q_rows=1, workspace=ws)
    ref = sm70_nvfp4_reference_attention(
        q, kv, bt, kv_lens, 1.0, 1.0, 1.0 / 16, -1)
    assert torch.allclose(out.cpu().float(), ref, atol=1e-2, rtol=1e-2)


def test_op_chunked_blocks():
    """page=32 blocks: in-block chunk offsets, split boundaries not 16-
    aligned (100/3, 71/3), and partial trailing blocks (100 = 3x32+4)."""
    q, kv, bt, kv_lens = make_case(R=2, Q=1, seed=8,
                                   kv_lens=(100, 71), page=32)
    R, Q, HQ_, D_ = q.shape
    H_ = kv.shape[1] // 2
    splits = 3
    ws = torch.zeros(
        R * H_ * splits * Q * 6 * (D_ + 2), dtype=torch.float32,
        device="cuda")
    out = torch.empty_like(q, device="cuda")
    sm70_flashdecode_op(
        q.cuda(), kv.cuda(), bt.cuda(), kv_lens.cuda(), out, 1.0, 1.0,
        1.0 / 16, -1, splits=splits, q_rows=1, workspace=ws)
    ref = sm70_nvfp4_reference_attention(
        q, kv, bt, kv_lens, 1.0, 1.0, 1.0 / 16, -1)
    assert torch.allclose(out.cpu().float(), ref, atol=1e-2, rtol=1e-2)


def test_op_q4_verify_matches_reference():
    """q_rows=4 speculative-verify rows (causal horizons) vs the reference.

    Row qq attends to [0, kv_len - Q + qq + 1) — the per-row horizon must
    hold inside one kernel pass over the same staged KV.
    """
    q, kv, bt, kv_lens = make_case(R=1, Q=4, seed=6, kv_lens=(123,))
    H_ = kv.shape[1] // 2
    splits = 4
    ws = torch.zeros(
        1 * H_ * splits * 4 * 6 * (D + 2), dtype=torch.float32,
        device="cuda")
    out = torch.empty_like(q, device="cuda")
    sm70_flashdecode_op(
        q.cuda(), kv.cuda(), bt.cuda(), kv_lens.cuda(), out, 1.0, 1.0,
        1.0 / 16, -1, splits=splits, q_rows=4, workspace=ws)
    ref = sm70_nvfp4_reference_attention(
        q, kv, bt, kv_lens, 1.0, 1.0, 1.0 / 16, -1)
    assert torch.allclose(out.cpu().float(), ref, atol=1e-2, rtol=1e-2)


def test_op_window_matches_reference():
    """sliding window_left=32 vs the reference (tail-window semantics)."""
    q, kv, bt, kv_lens = make_case(R=1, Q=1, seed=7, kv_lens=(60,))
    H_ = kv.shape[1] // 2
    splits = 4
    ws = torch.zeros(
        1 * H_ * splits * 1 * 6 * (D + 2), dtype=torch.float32,
        device="cuda")
    out = torch.empty_like(q, device="cuda")
    sm70_flashdecode_op(
        q.cuda(), kv.cuda(), bt.cuda(), kv_lens.cuda(), out, 1.0, 1.0,
        1.0 / 16, 32, splits=splits, q_rows=1, workspace=ws)
    ref = sm70_nvfp4_reference_attention(
        q, kv, bt, kv_lens, 1.0, 1.0, 1.0 / 16, 32)
    assert torch.allclose(out.cpu().float(), ref, atol=1e-2, rtol=1e-2)


def test_op_kv1_exact_dequant():
    """kv_len=1: softmax degenerates (p == 1) so out == V[0] exactly.

    Exact dequant gate with FULL e4m3 scale range (sf_max=247) — catches
    nibble/scale-group misaddressing that a moderate-magnitude softmax
    test could wash out. Tolerance = fp16 out quantization.
    """
    q, kv, bt, kv_lens = make_case(R=2, Q=1, seed=8, kv_lens=(1, 1),
                                   sf_max=247)
    H_ = kv.shape[1] // 2
    splits = 2
    ws = torch.zeros(
        2 * H_ * splits * 1 * 6 * (D + 2), dtype=torch.float32,
        device="cuda")
    out = torch.empty_like(q, device="cuda")
    sm70_flashdecode_op(
        q.cuda(), kv.cuda(), bt.cuda(), kv_lens.cuda(), out, 1.0, 1.0,
        1.0 / 16, -1, splits=splits, q_rows=1, workspace=ws)
    ref = sm70_nvfp4_reference_attention(
        q, kv, bt, kv_lens, 1.0, 1.0, 1.0 / 16, -1)
    assert torch.allclose(out.cpu().float(), ref, atol=1.0, rtol=1e-2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-x", "-s", *sys.argv[1:]]))
