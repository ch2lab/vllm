import time

import torch

import vllm._C_stable_libtorch

dev = "cuda"
torch.manual_seed(0)
AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
INV_ORDER = [0] * 8
for i, o in enumerate(AWQ_ORDER):
    INV_ORDER[o] = i


def make(K, N, G):
    qw = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device=dev)
    sc = (torch.rand(K // G, N, device=dev, dtype=torch.float16) + 0.1)
    qz = torch.randint(0, 2**31, (K // G, N // 8), dtype=torch.int32, device=dev)
    return qw, sc, qz


def ref_gemm(x, qw, sc, qz, G):
    K, N8 = qw.shape
    N = N8 * 8
    w = torch.zeros(K, N, dtype=torch.float32, device=dev)
    z = torch.zeros(K // G, N, dtype=torch.float32, device=dev)
    qwv = qw.to(torch.int64)
    qzv = qz.to(torch.int64)
    for j in range(8):
        logical = INV_ORDER[j]
        w[:, logical::8] = ((qwv >> (j * 4)) & 0xF).float()
        z[:, logical::8] = ((qzv >> (j * 4)) & 0xF).float()
    zg = z.repeat_interleave(G, dim=0)
    wdeq = (w - zg) * sc.repeat_interleave(G, dim=0).float()
    return (x.float() @ wdeq).half()


for (M, K, N, G) in [(1, 4096, 6912, 128), (5, 4096, 6912, 128),
                     (1, 5120, 1024, 128), (128, 4096, 6912, 128)]:
    qw, sc, qz = make(K, N, G)
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    out = torch.ops._C.awq_gemm_sm70(x, qw, sc, qz, G, 640)
    ref = ref_gemm(x, qw, sc, qz, G)
    err = (out.float() - ref.float()).abs()
    rel = (err / (ref.float().abs() + 1e-3)).max().item()
    # timing
    for _ in range(5):
        torch.ops._C.awq_gemm_sm70(x, qw, sc, qz, G, 640)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        torch.ops._C.awq_gemm_sm70(x, qw, sc, qz, G, 640)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 100 * 1e3
    wbytes = K * N // 2
    gbps = wbytes / (dt * 1e-3) / 1e9
    print(f"M={M} K={K} N={N}: rel_err={rel:.4f} t={dt:.3f} ms "
          f"eff_bw={gbps:.0f} GB/s")
