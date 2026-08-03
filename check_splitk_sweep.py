"""Sweep split_k target for AWQ SM70 GEMV/GEMM decode shapes."""
import torch

import vllm._custom_ops  # noqa: F401

dev = "cuda"
torch.manual_seed(0)
N, K, G = 4096, 5120, 128
qw = torch.randint(0, 2**31, (K, N // 8), device=dev, dtype=torch.int32)
sc = torch.rand(K // G, N, device=dev, dtype=torch.half) * 0.01
qz = torch.randint(0, 2**31, (K // G, N // 8), device=dev, dtype=torch.int32)

for M in (1, 5):
    x = torch.randn(M, K, device=dev, dtype=torch.half)
    for sk in (640, 1280, 2560, 5120):
        for _ in range(5):
            y = torch.ops._C.awq_gemm_sm70(x, qw, sc, qz, G, sk)
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(50):
            y = torch.ops._C.awq_gemm_sm70(x, qw, sc, qz, G, sk)
        e.record(); torch.cuda.synchronize()
        us = s.elapsed_time(e) * 1000 / 50
        bw = (K * N / 2 + K * N * 2 / G) / (us * 1e-6) / 1e9
        print(f"M={M} split_k_target={sk}: {us:.1f} us  eff_BW={bw:.0f} GB/s")
