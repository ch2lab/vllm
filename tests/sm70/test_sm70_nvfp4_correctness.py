# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone numeric-correctness test for the SM70 NVFP4 GEMM op.

Runs on a real V100 (GPU0, exact SM70) and compares the SM70 GEMM **wrapper**
``vllm.sm70_nvfp4_ops.gemm`` against an independent FP32 reference computed with
the **true** (natural-scale) dequant.

The SM70 kernel uses the fast/skip_flop dequant, which emits 2^-22x the true
dequantized weight; the wrapper compensates by scaling the caller's
(natural-scale) ``global_scale`` by 2^22, so the kernel output has the correct
end-to-end magnitude. This test validates exactly that SP2 concern: feeding a
natural-scale ``global_scale`` (the kind a real NVFP4 checkpoint like QUASAR
stores -- order 1e3, NOT pre-scaled by 2^22) yields an output that matches the
true FP32 reference within a small FP16-accumulation tolerance (and is NOT
2^-22x too small).

Per shape it:
  1. builds synthetic NVFP4 weights (packed 4-bit E2M1 codes) + group-16
     FP8-E4M3 scales + a natural per-tensor FP32 global_scale + FP16 acts;
  2. computes the independent FP32 true reference;
  3. runs the SM70 GEMM wrapper;
  4. compares and prints PASS/FAIL with max abs / rel error + magnitude.

This script is standalone (no pytest). Run with:
    cd /data/src/vllm && CUDA_VISIBLE_DEVICES=0 VLLM_SM70_NVFP4=1 \
        python3 tests/sm70/test_sm70_nvfp4_correctness.py
"""

import sys
import numpy as np
import torch

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Load the C extension (registers torch.ops._C.sm70_nvfp4_marlin_gemm) and the
# Python wrapper. Importing vllm._C_stable_libtorch loads the top-level `vllm`
# package (which does NOT pull in the flash-attn subtree), so the wrapper
# vllm.sm70_nvfp4_ops is importable alongside it.
# ---------------------------------------------------------------------------
try:
    import vllm._C_stable_libtorch  # noqa: F401
    from vllm.sm70_nvfp4_ops import gemm as sm70_gemm
except Exception as exc:  # pragma: no cover - environment dependent
    print(f"FATAL: could not load stable extension / wrapper: {exc}")
    sys.exit(2)

_OP = "sm70_nvfp4_marlin_gemm"
if not hasattr(torch.ops._C, _OP):
    print("FATAL: torch.ops._C." + _OP + " not registered.")
    sys.exit(2)

DEVICE = "cuda"

# E2M1 (FP4) code -> value LUT (code 0..15).
LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

# The kernel's fast (skip_flop=true) dequant omits the FP4->FP16 and FP8->FP16
# exponent biases, so its raw output is 2^-22x the true dequantized weight.
# The wrapper multiplies the caller's natural-scale global_scale by 2^22 to
# cancel this (see vllm.sm70_nvfp4_ops.gemm).
DEQ_BIAS = float(2.0) ** 22  # the wrapper's compensation factor

# A NATURAL per-tensor scale, same order as the real QUASAR-NVFP4 checkpoint
# (measured ~5173/8048) -- critically NOT the 2^22 (~4.19e6) that 1Cat
# checkpoints bake in. With this gs the wrapper must produce the TRUE output
# magnitude (not 2^-22x smaller).
GS = 5120.0
# Group-16 fp8 scale pool (all exactly representable in fp8_e4m3fn).
SCALE_POOL = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
# Activation amplitude: keeps the FP16 GEMM output in range for this natural gs
# (the relationship out == gs*(a @ b_deq.T) is scale-invariant in a).
ACT_SCALE = 0.002

# Global (Frobenius) relative error tolerance for the FP16 kernel vs FP32
# reference: robust to per-element cancellation and isolates the FP16-
# accumulation difference.
RTOL = 1e-2


# ---------------------------------------------------------------------------
# Packing (matches csrc .../marlin/sm70_nvfp4_common.cuh)
# ---------------------------------------------------------------------------
def packed_macro_n(n: int) -> int:
    if n % 256 == 0:
        return 256
    if n % 128 == 0:
        return 128
    if n % 64 == 0:
        return 64
    raise ValueError(f"size_n={n} not divisible by 64")


def _u4_offset(packed_macro_n: int, size_n: int, k, n):
    """Vectorized u4_packed_macro_n_qweight_offset_from_logical (word index)."""
    KQT = 16  # kQuantTileK
    KQN = 64  # kQuantTileN
    kgt = packed_macro_n // KQN
    k_tile = k // KQT
    local_k = k - k_tile * KQT
    n_tile = n // KQN
    group_n_tile = n_tile // kgt
    subtile = n_tile - group_n_tile * kgt
    local_n_vec = (n - n_tile * KQN) // 8
    local_word = local_k * (KQN // 8) + local_n_vec
    return (k_tile * (size_n * 2)
            + group_n_tile * kgt * (KQT * KQN // 8)
            + local_word * kgt + subtile)


# The kernel's B-iterator dequant permutes the 8 nibbles of a packed code word
# and the 8 scale bytes within an 8-column group. Empirically determined (and
# stable across CTA geometries): output column c (c = n % 8) reads the code
# nibble at CODE_P[c] and the scale byte at SCALE_P[c]. Both are involutions.
CODE_P = np.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int64)
SCALE_P = np.array([0, 2, 1, 3, 4, 6, 5, 7], dtype=np.int64)


def pack_qweight(code_mat: np.ndarray, k: int, n: int) -> torch.Tensor:
    """Pack a (K, N) matrix of 4-bit codes into the kernel's uint8 layout.

    ``code_mat[k, n]`` is the code for the weight at (output col n, input k).
    Returns a 1-D uint8 tensor of size K*N/2.
    """
    pmn = packed_macro_n(n)
    K = code_mat.shape[0]
    N = code_mat.shape[1]
    kk = np.arange(K, dtype=np.int64)[:, None]
    nn = np.arange(N, dtype=np.int64)[None, :]
    word = _u4_offset(pmn, N, kk, nn)  # (K, N)
    nib = CODE_P[nn % 8]               # (1, N) -> broadcast (K, N)
    vals = (code_mat.astype(np.int64) & 0xF) << (4 * nib)  # (K, N)
    words = np.zeros(K * N // 8, dtype=np.uint32)
    np.add.at(words, word.ravel(), vals.ravel())
    # K*N//8 uint32 words == K*N/2 bytes, matching the kernel's packed layout.
    return torch.from_numpy(np.ascontiguousarray(words)).view(torch.uint8)


def to_scale_bytes(scale_vals: np.ndarray) -> np.ndarray:
    """scale_vals (N, K//16) float -> (K//16, N) uint8 fp8_e4m3fn, row-major.

    The kernel's scale dequant reads the 8 scale bytes of an 8-column group in a
    permuted order: output column c (c = n % 8) is scaled by the byte at group
    position SCALE_P[c]. So buffer position q within the group must hold the
    logical scale of column (group_start + SCALE_P[q]) (SCALE_P is an involution).
    """
    N, G = scale_vals.shape
    buf = np.empty_like(scale_vals)
    n = np.arange(N)
    src = (n // 8) * 8 + SCALE_P[n % 8]  # buffer pos n holds logical scale_vals at src
    buf[:] = scale_vals[src, :]
    b = torch.from_numpy(buf.astype(np.float32)).to(torch.float8_e4m3fn)
    return b.view(torch.uint8).t().contiguous().cpu().numpy()


def build(m: int, n: int, k: int, gs: float):
    """Build synthetic inputs: FP16 acts, packed codes, fp8 scales, gs, output."""
    a = (torch.randn(m, k, dtype=torch.float32, device=DEVICE) * ACT_SCALE).half()
    code_mat = np.random.randint(0, 16, size=(k, n)).astype(np.int64)
    g = k // 16
    scale_vals = SCALE_POOL[np.random.randint(0, len(SCALE_POOL), size=(n, g))]
    qw = pack_qweight(code_mat, k, n).to(DEVICE)
    scales = torch.from_numpy(to_scale_bytes(scale_vals)).to(DEVICE)
    gs_t = torch.tensor([gs], dtype=torch.float32, device=DEVICE)
    c = torch.zeros(m, n, dtype=torch.float16, device=DEVICE)
    return a, code_mat, scale_vals, qw, scales, gs_t, c


def true_reference(a, code_mat, scale_vals, gs):
    """Independent FP32 TRUE reference: ``gs * (a @ (e2m1(code)*fp8).T)``.

    This is the natural-scale dequant (NO 2^-22 factor). With the wrapper's
    2^22 compensation, the kernel output should match this exactly in magnitude
    (before the fix it was 2^-22x smaller).
    """
    lut_vals = LUT[code_mat]  # (K, N)
    b_deq = (lut_vals.T * np.repeat(scale_vals, 16, axis=1)).astype(np.float32)
    a_t = a.float().to(DEVICE)                  # (M, K) fp32
    b_t = torch.from_numpy(b_deq).to(DEVICE)    # (N, K) fp32
    ref = (a_t @ b_t.T).mul_(gs)                # (M, N)
    return ref.cpu().numpy().astype(np.float32)


def run_shape(m, n, k, gs, verbose=True):
    a, code_mat, scale_vals, qw, scales, gs_t, c = build(m, n, k, gs)
    # The wrapper scales gs by 2^22 internally to cancel the kernel's 2^-22.
    out = sm70_gemm(a, c, qw, scales, gs_t, m, n, k, 16)
    torch.cuda.synchronize()
    out_f = out.float().cpu().numpy().astype(np.float32)
    ref = true_reference(a, code_mat, scale_vals, gs)

    d = out_f - ref
    max_abs = float(np.abs(d).max())
    # Global (Frobenius) relative error: robust to per-element cancellation,
    # isolates the FP16 (kernel) vs FP32 (reference) accumulation difference.
    frob = float(np.linalg.norm(d) / max(np.linalg.norm(ref), 1e-30))
    ok = frob < RTOL
    # Magnitude check: with the wrapper's compensation the output has the TRUE
    # magnitude, so ||out||/||ref|| ~ 1.0 (was ~2^-22 before the fix).
    mag_ratio = float(np.linalg.norm(out_f) / max(np.linalg.norm(ref), 1e-30))

    if verbose:
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] M={m:<4d} N={n:<6d} K={k:<6d} | "
              f"frob_rel={frob:.3e} max_abs={max_abs:.3e} ok={ok}")
        print(f"        | ||out||/||true_ref||={mag_ratio:.4e} (expect ~1.0)")
    return ok, frob, mag_ratio


def probe_real_dims(verbose=True):
    """Probe the real-model dims K=N=11008 (Qwen3.5): accepted + true magnitude."""
    results = {}
    for (m, n, k) in [(1, 11008, 11008), (16, 11008, 11008)]:
        try:
            ok, frob, mag_ratio = run_shape(m, n, k, GS)
            results[(m, n, k)] = ("ACCEPTED", frob, mag_ratio)
            if verbose:
                print(f"[probe] M={m} N={n} K={k}: ACCEPTED by op; "
                      f"mech PASS={ok} frob_rel={frob:.4e} "
                      f"||out||/||true_ref||={mag_ratio:.4e}")
        except Exception as exc:  # TORCH_CHECK / alignment failure
            results[(m, n, k)] = ("REJECTED", str(exc)[:200], None)
            if verbose:
                print(f"[probe] M={m} N={n} K={k}: REJECTED: {str(exc)[:200]}")
    return results


def main():
    if not torch.cuda.is_available():
        print("FATAL: no CUDA device available.")
        sys.exit(2)
    cap = tuple(torch.cuda.get_device_capability())
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm{'.'.join(map(str, cap))})")
    if cap != (7, 0):
        print("WARNING: not exact SM70; the op may not dispatch. "
              "Continuing for reporting.")

    shapes = [(m, n, k)
              for m in (1, 16, 64, 512)
              for n in (4096, 8192)
              for k in (4096, 8192)]

    print("=" * 78)
    print(f"SM70 NVFP4 GEMM true-magnitude test (wrapper compensates 2^-22; "
          f"natural gs={GS:g})")
    print("=" * 78)
    n_pass = 0
    frobs = []
    mag_ratios = []
    for (m, n, k) in shapes:
        try:
            ok, frob, mag_ratio = run_shape(m, n, k, GS)
            n_pass += int(ok)
            frobs.append(frob)
            mag_ratios.append(mag_ratio)
        except Exception as exc:
            print(f"[ERROR] M={m} N={n} K={k}: {str(exc)[:200]}")

    print("-" * 78)
    print(f"True-magnitude: {n_pass}/{len(shapes)} shapes PASS "
          f"(Frobenius rel err < {RTOL})")
    if frobs:
        print(f"worst Frobenius rel err: {max(frobs):.4e} "
              f"(min={min(frobs):.4e})")
    if mag_ratios:
        rmin, rmax = min(mag_ratios), max(mag_ratios)
        print(f"||out||/||true_ref|| range: [{rmin:.4e}, {rmax:.4e}] "
              f"(expect ~1.0; was ~2^-22={DEQ_BIAS ** -1:.4e} before the fix)")
    print("-" * 78)
    print("Real-model dim probe (Qwen3.5, natural gs):")
    probe_real_dims()

    # Exit 0 iff every shape matched the true reference within tolerance.
    sys.exit(0 if n_pass == len(shapes) else 1)


if __name__ == "__main__":
    main()
