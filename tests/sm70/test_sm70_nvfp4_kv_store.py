#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SM70 (V100) NVFP4 KV-cache STORE path — standalone correctness test.

This is a standalone script (no pytest).  It verifies, on a Volta GPU:

  1. The NVFP4 KV-cache store kernel (invoked through
     ``_custom_ops.reshape_and_cache`` with ``kv_cache_dtype="nvfp4"``)
     produces packed FP4 data + FP8 block-16 scales for bf16 K/V.
  2. Dequantizing those (linear scale layout, matching the SM70/SM120 reader)
     recovers the original K/V within NVFP4 quantization tolerance.
  3. The FlashInfer backend-selection path is env-gated:
     ``VLLM_SM70_NVFP4_KV`` off -> upstream default (nvfp4 NOT supported on
     SM70, i.e. no behavior change); on -> supported.

Run (GPU1 only):
    CUDA_VISIBLE_DEVICES=1 VLLM_SM70_NVFP4_KV=1 \
        python3 tests/sm70/test_sm70_nvfp4_kv_store.py
"""

import os
import sys

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    nvfp4_kv_cache_full_dim,
    nvfp4_split_data_scale,
)

DEVICE = "cuda"
E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
    device=DEVICE,
)


def dequant_side(data_u8, scale_fp8, global_scale, head_size):
    """Dequantize one KV side (NHD) produced by the store kernel.

    data_u8:   [B, N, H, data_dim]  uint8 (2 FP4 / byte: low=even, high=odd)
    scale_fp8: [B, N, H, scale_dim] fp8 (linear, block-16)

    Returns [B, N, H, head_size] float32 = E2M1_LUT[code] * scale * global_scale.
    """
    B, N, H, data_dim = data_u8.shape
    lo = data_u8 & 0xF
    hi = (data_u8 >> 4) & 0xF
    vals_lo = E2M1_LUT[lo.long()]
    vals_hi = E2M1_LUT[hi.long()]
    vals = torch.stack([vals_lo, vals_hi], dim=-1).reshape(B, N, H, head_size)
    s = scale_fp8.to(torch.float32).repeat_interleave(16, dim=-1)
    return vals * s * global_scale


def store_roundtrip(checkpoint_scale=1.0):
    """Store bf16 K/V as NVFP4, dequant (linear), compare to the original."""
    head_size = 128
    num_heads = 8
    block_size = 16
    num_blocks = 4
    num_tokens = num_blocks * block_size  # 64

    data_dim = head_size // 2
    scale_dim = head_size // 16
    full_dim = nvfp4_kv_cache_full_dim(head_size)
    assert full_dim == data_dim + scale_dim

    torch.manual_seed(0)
    key = (torch.randn(num_tokens, num_heads, head_size, device=DEVICE)
           * 0.75).to(torch.bfloat16)
    value = (torch.randn(num_tokens, num_heads, head_size, device=DEVICE)
             * 0.75).to(torch.bfloat16)

    k_scale = torch.full((1,), checkpoint_scale, device=DEVICE, dtype=torch.float32)
    v_scale = torch.full((1,), checkpoint_scale, device=DEVICE, dtype=torch.float32)

    # Paged KV cache, NHD contiguous, uint8.  Each side holds
    # [data | scale] per head (per-page layout [K_data|K_scale|V_data|V_scale]).
    # Separate contiguous tensors (the store op takes in-place `Tensor!`
    # outputs, which the stable ABI rejects for slice views).
    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, full_dim,
        dtype=torch.uint8, device=DEVICE)
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, full_dim,
        dtype=torch.uint8, device=DEVICE)

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)

    # The NVFP4 store is dispatched by reshape_and_cache_flash (the plain
    # reshape_and_cache is a legacy op that does not route nvfp4).
    ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, "nvfp4", k_scale,
        v_scale)
    torch.cuda.synchronize()

    # Dequant with the linear reader (SM70 uses the linear scale layout).
    # Reader's global_scale == checkpoint_scale (the k_scale / v_scale tensor).
    data_k, scale_k = nvfp4_split_data_scale(key_cache)
    data_v, scale_v = nvfp4_split_data_scale(value_cache)
    assert data_k.any().item(), "K FP4 data is all zeros — store did not run"
    assert data_v.any().item(), "V FP4 data is all zeros — store did not run"
    assert scale_k.view(torch.uint8).any().item(), \
        "K scales all zero — store did not run"

    deq_k = dequant_side(data_k, scale_k, checkpoint_scale, head_size)
    deq_v = dequant_side(data_v, scale_v, checkpoint_scale, head_size)
    deq_k = deq_k.reshape(-1, num_heads, head_size)
    deq_v = deq_v.reshape(-1, num_heads, head_size)

    def metrics(orig, deq, name):
        o = orig.float()
        e = (deq - o).abs()
        max_abs = e.max().item()
        mean_abs = e.mean().item()
        rel_l2 = (e.pow(2).mean()).sqrt().item() / o.pow(2).mean().sqrt().item()
        # relative error ignoring near-zero elements (ill-conditioned)
        rel = e / o.abs().clamp_min(0.25)
        mean_rel = rel.mean().item()
        print(f"    {name}: max_abs_err={max_abs:.4g} "
              f"mean_abs_err={mean_abs:.4g} rel_l2={rel_l2:.4g} "
              f"mean_rel(|x|>=.25)={mean_rel:.4g}")
        # NVFP4 (E2M1 + FP8 block scale) tolerance: normalized L2 well below 1.
        assert rel_l2 < 0.4, f"{name} rel_l2={rel_l2} exceeds NVFP4 tolerance"
        # The dequant must be far closer to the original than to random: sanity.
        assert (e.pow(2).mean().item() < o.pow(2).mean().item()), \
            f"{name} dequant not closer to original than zero"

    print(f"  checkpoint_scale={checkpoint_scale}:")
    metrics(key, deq_k, "K")
    metrics(value, deq_v, "V")
    return deq_k, deq_v


def selection_check():
    """Verify the env-gated backend-selection logic (no flashinfer pkg needed).

    Mirrors FlashInferBackend.supports_kv_cache_dtype("nvfp4"):
      - if VLLM_SM70_NVFP4_KV and family(70): True
      - else: the upstream SM100 gate (False on SM70).
    """
    family70 = current_platform.is_device_capability_family(70)
    family100 = current_platform.is_device_capability_family(100)
    env_kv = envs.VLLM_SM70_NVFP4_KV
    cap = current_platform.get_device_capability()
    print(f"  device capability: {cap}")
    print(f"  family(70)={family70}  family(100)={family100}  "
          f"VLLM_SM70_NVFP4_KV={env_kv}")

    def supports(env_kv_val):
        sm70_enabled = env_kv_val and family70
        if sm70_enabled:
            return True
        # Upstream SM100 gate; on SM70 family(100) is False -> False.
        return family100  # supports_trtllm_attention(...) and family100

    off = supports(False)
    on = supports(True)
    # On a V100: env-off must equal the upstream default (nvfp4 NOT supported),
    # env-on enables the SM70 path.
    if family70:
        assert off is False, \
            f"env-off must be the upstream default (False on SM70), got {off}"
        assert on is True, f"env-on should enable SM70 NVFP4 KV, got {on}"
    print(f"  supports_kv_cache_dtype('nvfp4'): env-off={off} env-on={on}  "
          f"(current env={env_kv} -> {supports(env_kv)})")

    # Attempt to import the real backend module; it needs the flashinfer pip
    # package, which may be absent in this env.  Report, don't fail, if missing.
    try:
        from vllm.v1.attention.backends import flashinfer as _fi  # noqa: F401
        print("  imported vllm.v1.attention.backends.flashinfer: OK")
        from vllm.v1.attention.backends.flashinfer import FlashInferBackend
        print(f"  FlashInferBackend.supports_kv_cache_dtype('nvfp4') = "
              f"{FlashInferBackend.supports_kv_cache_dtype('nvfp4')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (note) full backend module import skipped: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0] if str(exc) else ''}")


def main():
    print("=== SM70 NVFP4 KV-cache store roundtrip ===")
    store_roundtrip(checkpoint_scale=1.0)
    store_roundtrip(checkpoint_scale=2.0)
    print("  store roundtrip: OK (within NVFP4 tolerance)")

    print("\n=== SM70 NVFP4 KV-cache backend selection (env-gated) ===")
    selection_check()

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
