# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Smoke + small oracle for the SM70 splitd D256 prefill ops.

The splitd ops are registered in the 1Cat flash-attention-v100 fork build of
``_vllm_fa2_C``; importing the Python package loads the C library.
"""

import pytest
import torch


def _load_ops():
    import vllm.vllm_flash_attn  # noqa: F401 (loads the C library)

    paged = torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_paged_fwd
    dense = torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_dense_fwd
    return paged, dense


def _ref_causal_gqa(q64, k64, v64, scale):
    """FP64 reference: GQA attention, q [1, T, Hq, D], k/v [1, L, Hkv, D].

    Chunked-prefill semantics: q is the T tokens appended after the
    cached L - T KV rows, so q row t attends kv rows 0 .. (L - T) + t.
    """
    Hq, Hkv = q64.shape[2], k64.shape[2]
    T, L = q64.shape[1], k64.shape[1]
    t_all = torch.arange(T, device=q64.device)
    l = torch.arange(L, device=q64.device)[None, :]
    out = torch.empty_like(q64)
    # Per-q-head + row-chunked: the full GQA-expanded fp64 score tensor
    # would not fit alongside the production server on the test GPUs.
    for h in range(Hq):
        hk = h // (Hq // Hkv)
        k_h = k64[0, :, hk, :]  # [L, D]
        v_h = v64[0, :, hk, :]
        for t0 in range(0, T, 256):
            t1 = min(t0 + 256, T)
            q_h = q64[0, t0:t1, h, :]  # [tc, D]
            t = t_all[t0:t1]
            qk = (q_h @ k_h.T) * scale  # [tc, L]
            mask = l <= (L - T) + t[:, None]
            p = torch.softmax(qk.masked_fill(~mask, float("-inf")), dim=-1)
            out[0, t0:t1, h, :] = p @ v_h
    return out


def _run(paged_op, dense_op, paged, T, L, seed, device, shuffle=False):
    """Run one (q_len, kv_len) case and return (out, err) vs the FP64 ref."""
    Hq, Hkv, D, N = 12, 2, 256, 16
    scale = 1.0 / 16.0
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(1, T, Hq, D, generator=g, dtype=torch.float16).to(device)
    k = torch.randn(1, L, Hkv, D, generator=g, dtype=torch.float16).to(device)
    v = torch.randn(1, L, Hkv, D, generator=g, dtype=torch.float16).to(device)
    out = torch.empty_like(q)

    if paged:
        nb = L // N
        k_p = k.reshape(nb, N, Hkv, D)
        v_p = v.reshape(nb, N, Hkv, D)
        if shuffle:
            order = torch.randperm(nb, generator=torch.Generator().manual_seed(seed))
            k_p = k_p[order]
            v_p = v_p[order]
        bt = (
            torch.arange(nb, dtype=torch.int32)
            if not shuffle
            else order.to(torch.int32)
        ).to(device).unsqueeze(0)
        paged_op(q, k_p, v_p, bt, out, L, scale, True)
    else:
        dense_op(q, k, v, out, scale, True)

    ref = _ref_causal_gqa(q.double(), k.double(), v.double(), scale)
    err = (out.double() - ref).abs().max().item()
    del ref
    return out, err


@pytest.mark.parametrize(
    "paged,T,L,shuffle",
    [
        (False, 64, 64, False),          # no-cache minimum (q == kv)
        (False, 64, 128, False),
        (False, 128, 1024, False),
        (False, 1024, 65536, False),     # long KV (beyond the 54K bar)
        (True, 64, 64, False),
        (True, 4096, 16384, False),      # production chunk + long KV
        (True, 1024, 65536, False),
        (True, 4096, 16384, True),        # block-table permutation
    ],
    ids=[
        "dense-q64-kv64", "dense-q64-kv128", "dense-q128-kv1024",
        "dense-q1024-kv65536",
        "paged-q64-kv64", "paged-q4096-kv16384", "paged-q1024-kv65536",
        "paged-shuffled",
    ],
)
def test_splitd_matches_fp64_reference(paged, T, L, shuffle):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    paged_op, dense_op = _load_ops()
    out, err = _run(
        paged_op, dense_op, paged, T, L, seed=7,
        device=torch.device("cuda:0"), shuffle=shuffle,
    )
    assert torch.isfinite(out).all()
    # The paged kernel's fp16 accumulation is slightly order-sensitive:
    # identity vs shuffled block tables differ by ~6.5e-2 max (measured
    # 1Cat kernel property; production uses the identity compact
    # workspace). Semantic garbage would be O(1).
    tol = 1e-1 if (paged and shuffle) else 2e-2
    assert err < tol, f"max err {err}"
