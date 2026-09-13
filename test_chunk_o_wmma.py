"""Correctness: fla_chunk_o_sm70 vs Triton chunk_fwd_kernel_o.

Runs on device with synthetic varlen data matching production shapes
(Hg=16, H=48, K=V=128 per-rank equivalents; TP2 uses 8/24 but we test
both). Requires a free GPU.
"""
import os
import torch

os.environ["VLLM_SM70_FLA_CHUNK_O_WMMA"] = "0"  # force triton path below

from vllm.third_party.flash_linear_attention.ops.chunk_o import (  # noqa: E402
    chunk_fwd_kernel_o, chunk_fwd_o)
from vllm.third_party.flash_linear_attention.ops.index import (  # noqa: E402
    prepare_chunk_indices)
from vllm.triton_utils import triton  # noqa: E402

torch.manual_seed(0)
dev = "cuda"


def run_case(seqlens, Hg, H, K, V, BT=64):
    B = len(seqlens)
    T = sum(seqlens)
    NT = sum((s + BT - 1) // BT for s in seqlens)
    # varlen packing: single batch row holding all sequences back to back
    q = torch.randn(1, T, Hg, K, device=dev, dtype=torch.float16) * 0.5
    k = torch.randn(1, T, Hg, K, device=dev, dtype=torch.float16) * 0.5
    v = torch.randn(1, T, H, V, device=dev, dtype=torch.float16) * 0.5
    h = torch.randn(NT, H, V, K, device=dev, dtype=torch.float16) * 0.5
    g = (torch.rand(1, T, H, device=dev, dtype=torch.float32) * -3.0)
    cu = torch.tensor([0] + list(torch.cumsum(
        torch.tensor(seqlens), 0)), device=dev, dtype=torch.int32)
    ci = prepare_chunk_indices(cu, BT)
    scale = K ** -0.5

    # Triton reference (single config to avoid autotune nondeterminism)
    o_ref = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q, k, v, h, g, o_ref, cu, ci, scale,
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT,
    )

    os.environ["VLLM_SM70_FLA_CHUNK_O_WMMA"] = "1"
    import vllm.third_party.flash_linear_attention.ops.chunk_o as co
    co._SM70_CHUNK_O_OK = None
    o_new = chunk_fwd_o(q, k, v, h, g, scale=scale,
                        cu_seqlens=cu, chunk_indices=ci, chunk_size=BT)
    os.environ["VLLM_SM70_FLA_CHUNK_O_WMMA"] = "0"
    co._SM70_CHUNK_O_OK = None

    diff = (o_new.float() - o_ref.float()).abs()
    denom = o_ref.float().abs().clamp_min(1e-2)
    rel = (diff / denom)
    print(f"seqlens={seqlens} H={H}: max_abs={diff.max().item():.4f} "
          f"max_rel={rel.max().item():.4f} "
          f"mean_abs={diff.mean().item():.5f}")
    return diff.max().item()


def main():
    worst = 0.0
    worst = max(worst, run_case([200], 16, 48, 128, 128))
    worst = max(worst, run_case([64, 128, 37], 16, 48, 128, 128))
    worst = max(worst, run_case([3300], 16, 48, 128, 128))
    worst = max(worst, run_case([129, 63], 8, 24, 128, 128))  # TP2 shapes
    print("WORST:", worst)
    print("PASS" if worst < 0.05 else "FAIL")


if __name__ == "__main__":
    main()
