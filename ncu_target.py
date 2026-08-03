"""Single-shape kernel launch for ncu profiling."""
import torch
import vllm._custom_ops  # noqa

IC, OC, group = 5120, 17408, 128
qw = torch.randint(-2**31, 2**31-1, (IC, OC//8), dtype=torch.int32, device='cuda')
sc = torch.randn(IC//group, OC, dtype=torch.float16, device='cuda').abs() + 0.1
zp = torch.randint(-2**31, 2**31-1, (IC//group, OC//8), dtype=torch.int32, device='cuda')
x = torch.randn(5, IC, dtype=torch.float16, device='cuda')
op = torch.ops._C.awq_gemm_sm70
for _ in range(5):
    op(x, qw, sc, zp)
torch.cuda.synchronize()
for _ in range(3):
    op(x, qw, sc, zp)
torch.cuda.synchronize()
