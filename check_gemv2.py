import torch

import vllm._C_stable_libtorch
import vllm._custom_ops as ops

dev = "cuda"
torch.manual_seed(0)
K, N, G = 4096, 256, 128
qw = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device=dev)
sc = torch.rand(K // G, N, device=dev, dtype=torch.float16) + 0.1
qz = torch.randint(0, 2**31, (K // G, N // 8), dtype=torch.int32, device=dev)
x = torch.ones(1, K, device=dev, dtype=torch.float16)
o = torch.ops._C.awq_gemm_sm70(x, qw, sc, qz, G, 640)
w = ops.awq_dequantize(qw, sc, qz, 0, 0, 0)
ref = x.float() @ w.float()
d = (o.float() - ref).abs()
print("max abs:", d.max().item(), "ref max:", ref.abs().max().item())
bad = (d > 1.0).nonzero()
print("bad count:", bad.shape[0], "of", N)
print("first bad cols:", bad[:8, 1].tolist())
print("o sample:", o[0, :4].tolist())
print("ref sample:", ref[0, :4].tolist())
