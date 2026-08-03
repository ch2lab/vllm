"""Compare WMMA vs TurboMind AWQ decode GEMM: correctness + bandwidth.

Reference = awq_dequantize + fp32 matmul (ground truth).
WMMA      = torch.ops._C.awq_gemm_sm70 (our existing 4-arg kernel).
TurboMind = torch.ops._sm70tm.awq_sm70_prepare + awq_gemm_sm70 (vendored).
"""
import time

import torch

import vllm._custom_ops  # noqa: register _C ops
import vllm._sm70_turbomind_C  # noqa: register _sm70tm ops

PEAK_BW = 900e9  # V100 HBM2 GB/s


def make(IC, OC, group=128, dev="cuda"):
    qw = torch.randint(-2**31, 2**31 - 1, (IC, OC // 8), dtype=torch.int32, device=dev)
    sc = torch.randn(IC // group, OC, dtype=torch.float16, device=dev).abs() + 0.1
    zp = torch.randint(-2**31, 2**31 - 1, (IC // group, OC // 8), dtype=torch.int32, device=dev)
    return qw, sc, zp


def timeit(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bytes_moved(M, IC, OC):
    w = IC * OC * 0.5
    sc = (IC // 128) * OC * 2
    zp = (IC // 128) * OC * 0.5
    return w + sc + zp + M * IC * 2 + M * OC * 2


def run(M, IC, OC):
    qw, sc, zp = make(IC, OC)
    x = torch.randn(M, IC, dtype=torch.float16, device="cuda")
    group_size = IC // sc.shape[0]

    w_fp16 = torch.ops._C.awq_dequantize(qw, sc, zp, 0, 0, 0)
    ref = (x.float() @ w_fp16.float()).half()

    # WMMA path
    wmma = torch.ops._C.awq_gemm_sm70(x, qw, sc, zp)
    wmma_rel = (wmma.float() - ref.float()).abs().max().item() / (
        ref.float().abs().max().item() + 1e-6)

    # TurboMind path
    tm_w, tm_s, meta = torch.ops._sm70tm.awq_sm70_prepare(qw, sc, zp, group_size, False)
    k_ld, q_ld = int(meta[0].item()), int(meta[1].item())
    tm = torch.ops._sm70tm.awq_gemm_sm70(x, tm_w, tm_s, group_size, k_ld, q_ld)
    tm_rel = (tm.float() - ref.float()).abs().max().item() / (
        ref.float().abs().max().item() + 1e-6)
    # cross-check turbomind vs wmma directly
    cross = (tm.float() - wmma.float()).abs().max().item() / (
        wmma.float().abs().max().item() + 1e-6)

    bw = bytes_moved(M, IC, OC)
    t_wmma = timeit(lambda: torch.ops._C.awq_gemm_sm70(x, qw, sc, zp))
    t_tm = timeit(lambda: torch.ops._sm70tm.awq_gemm_sm70(x, tm_w, tm_s, group_size, k_ld, q_ld))

    print(f"  M={M} IC={IC:5d} OC={OC:5d} | "
          f"WMMA {t_wmma*1e6:7.1f}us {bw/t_wmma/PEAK_BW*100:5.1f}% rel={wmma_rel:.2e} | "
          f"TM {t_tm*1e6:7.1f}us {bw/t_tm/PEAK_BW*100:5.1f}% rel={tm_rel:.2e} | "
          f"cross={cross:.2e} | speedup={t_wmma/t_tm:.2f}x")
    return t_wmma, t_tm


if __name__ == "__main__":
    shapes = [
        ("gate_up", 5120, 17408),
        ("down", 8704, 5120),
        ("o_proj", 3072, 5120),
        ("qkv_gdn", 5120, 6144),
        ("gdn_in", 5120, 8192),
        ("gdn_out", 6144, 5120),
    ]
    for M in [1, 2, 5]:
        print(f"=== M={M} ===")
        sw = st = 0.0
        for name, IC, OC in shapes:
            w, t = run(M, IC, OC)
            sw += w
            st += t
        print(f"  SUM WMMA={sw*1e6:.1f}us TM={st*1e6:.1f}us speedup={sw/st:.2f}x\n")
