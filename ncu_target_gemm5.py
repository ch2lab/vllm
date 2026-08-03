"""ncu target: M=5 gate_up gemm with split_k_target=2560."""
import torch
import vllm._custom_ops  # noqa: F401

IC, OC = 5120, 17408
qw = torch.randint(-2**31, 2**31 - 1, (IC, OC // 8), dtype=torch.int32,
                   device='cuda')
sc = (torch.randn(IC // 128, OC, dtype=torch.float16, device='cuda').abs()
      + 0.1)
zp = torch.randint(-2**31, 2**31 - 1, (IC // 128, OC // 8), dtype=torch.int32,
                   device='cuda')
x = torch.randn(5, IC, dtype=torch.float16, device='cuda')
op = torch.ops._C.awq_gemm_sm70
for _ in range(8):
    y = op(x, qw, sc, zp, 128, 2560)
torch.cuda.synchronize()
