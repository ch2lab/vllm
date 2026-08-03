"""ncu target: AWQ SM70 GEMM/GEMV at decode shapes (27B TP2-like)."""
import torch

import vllm._custom_ops  # noqa: F401

dev = "cuda"
torch.manual_seed(0)
N, K, G = 4096, 5120, 128
qw = torch.randint(0, 2**31, (K, N // 8), device=dev, dtype=torch.int32)
sc = (torch.rand(K // G, N, device=dev, dtype=torch.half) * 0.01)
qz = torch.randint(0, 2**31, (K // G, N // 8), device=dev, dtype=torch.int32)

for M in (1, 5):
    x = torch.randn(M, K, device=dev, dtype=torch.half)
    for _ in range(3):
        y = torch.ops._C.awq_gemm_sm70(x, qw, sc, qz, G, 640)
torch.cuda.synchronize()
print("done")
