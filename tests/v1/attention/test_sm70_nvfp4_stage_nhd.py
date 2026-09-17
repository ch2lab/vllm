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

N = 16  # KV block size (production config)
H_KV = 2
D = 256
F = D // 2 + D // 16  # 144 raw bytes per (side, head, token)


def _raw_cache(num_blocks: int, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(
        0, 256, (num_blocks, 2 * H_KV, N, F),
        dtype=torch.uint8, device=device, generator=g,
    )


def _check(kv_cache, kv_len, k_scale, v_scale, device):
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


@pytest.mark.parametrize(
    "kv_len,k_scale,v_scale",
    [
        (32, 1.0, 1.0),          # minimum kernel-eligible KV
        (64, 0.37, 12.5),        # non-unit e4m3 layer scales
        (4848, 1.0, 1.0),         # 303 blocks, kv%32==0
        (100, 1.0, 1.0),          # partial last block (kv%16 != 0)
        (16384, 0.61, 2.0),       # 1024 blocks
    ],
)
def test_stage_bit_exact_vs_paged_shadow(kv_len, k_scale, v_scale):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    M = (kv_len + N - 1) // N
    kv_cache = _raw_cache(M, device, seed=kv_len)
    _check(kv_cache, kv_len, k_scale, v_scale, device)
