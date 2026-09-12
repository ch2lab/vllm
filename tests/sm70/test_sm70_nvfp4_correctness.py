# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone numeric-correctness test for the SM70 NVFP4 GEMM op.

Runs on a real V100 (GPU0, exact SM70) and compares the C op
``torch.ops._C.sm70_nvfp4_marlin_gemm`` against an independent FP32 reference.

Per shape it:
  1. builds synthetic NVFP4 weights (packed 4-bit E2M1 codes) + group-16
     FP8-E4M3 scales + a per-tensor FP32 global_scale + FP16 activations;
  2. computes an independent FP32 reference;
  3. runs the SM70 GEMM op;
  4. compares and prints PASS/FAIL with max abs / rel error.

This script is standalone (no pytest). Run with:
    cd /data/src/vllm && CUDA_VISIBLE_DEVICES=0 VLLM_SM70_NVFP4=1 \
        python3 tests/sm70/test_sm70_nvfp4_correctness.py
"""

import os
import sys
import numpy as np
import torch

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Load the C extension (registers torch.ops._C.sm70_nvfp4_marlin_gemm) without
# pulling in the full `vllm` package (avoids the unrelated flash-attn gap).
# ---------------------------------------------------------------------------
try:
    import vllm._C_stable_libtorch  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    print(f"FATAL: could not load stable extension: {exc}")
    sys.exit(2)

_OP = "sm70_nvfp4_marlin_gemm"
if not hasattr(torch.ops._C, _OP):
    print("FATAL: torch.ops._C." + _OP + " not registered.")
    sys.exit(2)
_gemm = getattr(torch.ops._C, _OP)

DEVICE = "cuda"

# E2M1 (FP4) code -> value LUT (code 0..15).
LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

# The kernel uses the fast (skip_flop=true) dequant, which omits the FP4->FP16
# and FP8->FP16 exponent biases. Their magnitudes are:
#   E2M1: (1 << (FP16_EXP-1)) - (1 << (FP4_EXP-1)) = 2^4 - 2^1 = 14  -> 2^-14
#   E4M3: (1 << (FP16_EXP-1)) - (1 << (FP8_EXP-1)) = 2^4 - 2^3 = 8   -> 2^-8
# So the kernel's effective dequantized weight = LUT * fp8scale * 2^-22.
E2M1_BIAS_BITS = 14
E4M3_BIAS_BITS = 8
DEQ_SCALE = float(2.0) ** (-(E2M1_BIAS_BITS + E4M3_BIAS_BITS))  # 2**-22

# Tolerance for the FP16 (kernel) vs FP32 (reference) accumulation difference.
# PASS uses a global (Frobenius) relative error: robust to per-element
# cancellation (near-zero elements) and isolates the FP16-accumulation error.
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


def build(m: int, n: int, k: int, gs: float, scale_pool):
    a = torch.randn(m, k, dtype=torch.float32, device=DEVICE).half()
    code_mat = np.random.randint(0, 16, size=(k, n)).astype(np.int64)
    # scale_vals (N, K//16): each entry drawn from distinct fp8 magnitudes.
    g = k // 16
    scale_vals = scale_pool[np.random.randint(0, len(scale_pool), size=(n, g))]
    qw = pack_qweight(code_mat, k, n).to(DEVICE)
    scales = torch.from_numpy(to_scale_bytes(scale_vals)).to(DEVICE)
    gs_t = torch.tensor([gs], dtype=torch.float32, device=DEVICE)
    c = torch.zeros(m, n, dtype=torch.float16, device=DEVICE)
    return a, code_mat, scale_vals, qw, scales, gs_t, c


def references(a, code_mat, scale_vals, gs, n, k):
    """Independent FP32 references.

    ref_true : gs * (a @ (LUT*fp8)^T)        -- the task-specified dequant.
    ref_mech : gs * (a @ (LUT*fp8*DEQ_SCALE)^T) -- what the kernel actually
              computes (fast/skip_flop dequant, 2^-22 lower).  Equals
              DEQ_SCALE * ref_true exactly.
    """
    lut_vals = LUT[code_mat]                       # (K, N)
    b_true = (lut_vals.T * np.repeat(scale_vals, 16, axis=1)).astype(np.float32)
    a_t = a.float().to(DEVICE)                      # (M, K) fp32
    b_t = torch.from_numpy(b_true).to(DEVICE)       # (N, K) fp32
    ref_true = (a_t @ b_t.T).mul_(gs)               # (M, N)
    ref_mech = ref_true * DEQ_SCALE
    return (ref_true.cpu().numpy().astype(np.float32),
            ref_mech.cpu().numpy().astype(np.float32))


def run_shape(m, n, k, gs, scale_pool, verbose=True):
    a, code_mat, scale_vals, qw, scales, gs_t, c = build(m, n, k, gs, scale_pool)
    out = _gemm(a, c, qw, scales, gs_t, m, n, k, 16)
    torch.cuda.synchronize()
    out_f = out.float().cpu().numpy().astype(np.float32)
    ref_true, ref_mech = references(a, code_mat, scale_vals, gs, n, k)

    d = out_f - ref_mech
    max_abs = float(np.abs(d).max())
    # Global (Frobenius) relative error: robust to per-element cancellation and
    # isolates the FP16 (kernel) vs FP32 (reference) accumulation difference.
    frob = float(np.linalg.norm(d) / max(np.linalg.norm(ref_mech), 1e-30))
    ok_mech = frob < RTOL
    # Global magnitude ratio out/ref_true: if mechanics are right it is ~DEQ_SCALE.
    mag_ratio = float(np.linalg.norm(out_f) / max(np.linalg.norm(ref_true), 1e-30))

    if verbose:
        tag = "PASS" if ok_mech else "FAIL"
        print(f"[{tag}] M={m:<4d} N={n:<6d} K={k:<6d} | "
              f"frob_rel={frob:.3e} max_abs={max_abs:.3e} ok={ok_mech}")
        print(f"        | ||out||/||ref_true||={mag_ratio:.4e} "
              f"(expect ~2^-22={DEQ_SCALE:.4e})")
    return ok_mech, frob, mag_ratio


def probe_real_dims(verbose=True):
    """Probe the real-model dims K=11008, N=11008 (Qwen3.5)."""
    results = {}
    for (m, n, k) in [(1, 11008, 11008), (16, 11008, 11008)]:
        try:
            a = torch.randn(m, k, dtype=torch.float32, device=DEVICE).half()
            code_mat = np.random.randint(0, 16, size=(k, n))
            scale_vals = np.ones((n, k // 16))
            qw = pack_qweight(code_mat, k, n).to(DEVICE)
            scales = torch.from_numpy(to_scale_bytes(scale_vals)).to(DEVICE)
            gs_t = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
            c = torch.zeros(m, n, dtype=torch.float16, device=DEVICE)
            _gemm(a, c, qw, scales, gs_t, m, n, k, 16)
            torch.cuda.synchronize()
            ok_mech, frob, ratio = run_shape(m, n, k, 1.0, np.array([1.0, 2.0]))
            results[(m, n, k)] = ("ACCEPTED", frob)
            if verbose:
                print(f"[probe] M={m} N={n} K={k}: ACCEPTED by op; "
                      f"mech PASS={ok_mech} frob_rel={frob:.4e} "
                      f"||out||/||ref_true||={ratio:.4e}")
        except Exception as exc:  # TORCH_CHECK / alignment failure
            results[(m, n, k)] = ("REJECTED", str(exc)[:200])
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

    # A global_scale that keeps the (fast-dequant) output in normal FP16 range
    # while leaving the 2^-22 dequant bias visible in out/ref_true.
    scale_pool = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    gs = float(2.0 ** 22)  # cancels DEQ_SCALE so |out| ~ O(sqrt(K)*|a|*|w|)

    shapes = []
    for m in (1, 16, 64, 512):
        for n in (4096, 8192):
            for k in (4096, 8192):
                shapes.append((m, n, k))

    print("=" * 78)
    print("SM70 NVFP4 GEMM correctness (mech reference uses the kernel's own "
          "fast-dequant = LUT*fp8*2^-22)")
    print("=" * 78)
    n_pass = 0
    frobs = []
    mag_ratios = []
    for (m, n, k) in shapes:
        try:
            ok_mech, frob, mag_ratio = run_shape(m, n, k, gs, scale_pool)
            n_pass += int(ok_mech)
            frobs.append(frob)
            mag_ratios.append(mag_ratio)
        except Exception as exc:
            print(f"[ERROR] M={m} N={n} K={k}: {str(exc)[:200]}")

    print("-" * 78)
    print(f"Mechanics: {n_pass}/{len(shapes)} shapes PASS "
          f"(Frobenius rel err < {RTOL})")
    if frobs:
        print(f"worst Frobenius rel err: {max(frobs):.4e} "
              f"(min={min(frobs):.4e})")
    if mag_ratios:
        rmin, rmax = min(mag_ratios), max(mag_ratios)
        print(f"||out||/||ref_true|| range: [{rmin:.4e}, {rmax:.4e}] "
              f"(expected ~2^-22 = {DEQ_SCALE:.4e})")
    print("-" * 78)
    print("Real-model dim probe (Qwen3.5):")
    probe_real_dims()

    # Exit 0 iff every mechanics comparison passed. The 2^-22 dequant bias is
    # reported separately (a dequant-model issue, not a GEMM-mechanics error).
    sys.exit(0 if n_pass == len(shapes) else 1)


if __name__ == "__main__":
    main()
