"""Measure awq_gemm_sm70 achieved bandwidth on dominant decode shapes."""
import torch, time

PEAK_BW = 900e9  # V100 HBM2 GB/s

def make(IC, OC, group=128, dev='cuda'):
    qw = torch.randint(-2**31, 2**31-1, (IC, OC//8), dtype=torch.int32, device=dev)
    sc = torch.randn(IC//group, OC, dtype=torch.float16, device=dev).abs() + 0.1
    zp = torch.randint(-2**31, 2**31-1, (IC//group, OC//8), dtype=torch.int32, device=dev)
    return qw, sc, zp

def bench(M, IC, OC, iters=200):
    qw, sc, zp = make(IC, OC)
    x = torch.randn(M, IC, dtype=torch.float16, device='cuda')
    op = torch.ops._C.awq_gemm_sm70
    # correctness vs tested awq_dequantize + matmul
    w_fp16 = torch.ops._C.awq_dequantize(qw, sc, zp, 0, 0, 0)
    ref = (x.float() @ w_fp16.float()).half()
    got = op(x, qw, sc, zp, 128, 2560)
    abs_err = (got.float() - ref.float()).abs().max().item()
    rel = abs_err / (ref.float().abs().max().item() + 1e-6)
    for _ in range(20):
        op(x, qw, sc, zp, 128, 2560)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        op(x, qw, sc, zp, 128, 2560)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    # bytes: weights (4bit = 0.5/elem) + scales + zeros + input + output
    w_bytes = IC * OC * 0.5
    sc_bytes = (IC//128) * OC * 2
    zp_bytes = (IC//128) * OC * 0.5
    total = w_bytes + sc_bytes + zp_bytes + M*IC*2 + M*OC*2
    bw = total / dt
    return dt*1e6, bw/PEAK_BW*100, w_bytes/1e6, abs_err, rel

if __name__ == '__main__':
    import vllm._custom_ops  # noqa: register ops
    shapes = [
        ("gate_up", 5120, 17408),
        ("down",    8704, 5120),
        ("o_proj",  3072, 5120),
        ("qkv_gdn", 5120, 6144),
        ("gdn_in",  5120, 8192),
        ("gdn_out", 6144, 5120),
        ("wide_sk1", 5120, 82048),  # OC/128>=640 -> split_k==1 fast path
    ]
    for M in [1, 2, 5]:
        print(f"\n=== M={M} ===")
        tot_us = 0
        for name, IC, OC in shapes:
            us, pct, wmb, aerr, rerr = bench(M, IC, OC)
            tot_us += us
            print(f"  {name:8s} IC={IC:5d} OC={OC:5d}  {us:7.1f}us  BW={pct:5.1f}%  W={wmb:.0f}MB  err={aerr:.3f} rel={rerr:.2e}")
        print(f"  {'SUM':8s} {tot_us:7.1f}us")
