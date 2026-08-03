import torch

a = torch.randn(512, 512, device="cuda", dtype=torch.half)
for _ in range(10):
    b = a @ a
torch.cuda.synchronize()
