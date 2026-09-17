# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NHD staging op (nvfp4 paged -> fp16 NHD workspaces) vs the validated
paged-shadow dequant: must be bit-exact (same raw bytes, same decode)."""

import pytest
import torch

from vllm.v1.attention.ops.sm70_nvfp4_dequant import (
    sm70_nvfp4_paged_dequant,
    sm70_nvfp4_paged_stage_nhd,
)

H_KV = 2
D = 256
F = D // 2 + D // 16  # 144 raw bytes per (side, head, token)


def _raw_cache(num_blocks: int, N: int, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(
        0, 256, (num_blocks, 2 * H_KV, N, F),
        dtype=torch.uint8, device=device, generator=g,
    )


def _check(N: int, kv_cache, kv_len, k_scale, v_scale, device):
    M = (kv_len + N - 1) // N
    block_table = torch.arange(
        M, dtype=torch.int32, device=device
    ).view(1, -1)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)

    # Reference: existing paged shadow dequant (validated bit-exact).
    shadow = torch.empty(
        M, H_KV, N, 2 * D, dtype=torch.float16, device=device
    )
    sm70_nvfp4_paged_dequant(
        kv_cache, block_table, kv_lens, shadow, k_scale, v_scale
    )

    # New: NHD staging (K and V in separate [M, N, H, D] workspaces).
    k_out = torch.zeros(M, N, H_KV, D, dtype=torch.float16, device=device)
    v_out = torch.zeros(M, N, H_KV, D, dtype=torch.float16, device=device)
    sm70_nvfp4_paged_stage_nhd(
        kv_cache, block_table[0], kv_len, k_out, v_out, k_scale, v_scale
    )

    # Reindex HND shadow -> NHD for comparison.
    ref_k = shadow[:, :, :, :D].permute(0, 2, 1, 3)
    ref_v = shadow[:, :, :, D:].permute(0, 2, 1, 3)
    full = kv_len // N
    rem = kv_len % N
    assert torch.equal(ref_k[:full], k_out[:full]), "K mismatch (full blocks)"
    assert torch.equal(ref_v[:full], v_out[:full]), "V mismatch (full blocks)"
    if rem:
        assert torch.equal(ref_k[full, :rem], k_out[full, :rem]), (
            "K mismatch (partial block)"
        )
        assert torch.equal(ref_v[full, :rem], v_out[full, :rem]), (
            "V mismatch (partial block)"
        )


# (page_size, kv_len, k_scale, v_scale). Page size 64 is the production
# NVFP4 HND KV cache layout; 16 covers the smaller-page path. The 28637
# case (7x4091 spec-decode chunks) is the kv where ceil32(kv) is not a
# multiple of 64 — the production shape that broke the old sizing.
CASES = [
    (16, 32, 1.0, 1.0),          # minimum kernel-eligible KV
    (16, 64, 0.37, 12.5),        # non-unit e4m3 layer scales
    (16, 4848, 1.0, 1.0),         # kv%32==0
    (16, 100, 1.0, 1.0),          # partial last block
    (16, 16384, 0.61, 2.0),
    (64, 64, 1.0, 1.0),           # one full page
    (64, 100, 1.0, 1.0),          # partial page
    (64, 28637, 1.0, 1.0),        # production 54K chunk-7 kv
    (64, 53778, 0.61, 2.0),       # production 54K total kv
]


@pytest.mark.parametrize(
    "N,kv_len,k_scale,v_scale",
    CASES,
    ids=[f"n{N}-kv{kv}" for N, kv, _, _ in CASES],
)
def test_stage_bit_exact_vs_paged_shadow(N, kv_len, k_scale, v_scale):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    M = (kv_len + N - 1) // N
    kv_cache = _raw_cache(M, N, device, seed=kv_len)
    _check(N, kv_cache, kv_len, k_scale, v_scale, device)
