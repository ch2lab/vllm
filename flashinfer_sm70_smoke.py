"""FlashInfer SM70 smoke test: JIT-compile FA2 paged decode for sm_70.

Run with:
  FLASHINFER_ALLOW_SM70=1 FLASHINFER_CUDA_ARCH_LIST="7.0" \
  PYTHONPATH=/data/src/flashinfer python3 flashinfer_sm70_smoke.py
"""
import torch

import flashinfer

print("flashinfer:", flashinfer.__file__)

torch.manual_seed(0)
dev = "cuda"
num_layers = 1
num_heads = 8
num_kv_heads = 2
head_dim = 128
page_size = 16
seq_len = 300
batch = 1

kv_data = torch.randn(
    batch * ((seq_len + page_size - 1) // page_size), 2,
    page_size, num_kv_heads, head_dim, device=dev, dtype=torch.float16,
) * 0.5
q = torch.randn(batch, num_heads, head_dim, device=dev, dtype=torch.float16)
kv_page_indptr = torch.tensor([0, (seq_len + page_size - 1) // page_size],
                              dtype=torch.int32, device=dev)
kv_last_page_len = torch.tensor(
    [seq_len - (seq_len // page_size) * page_size or page_size],
    dtype=torch.int32, device=dev)

wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev),
    kv_layout="NHD",
)
wrapper.plan(
    kv_page_indptr,
    torch.arange(kv_page_indptr[-1].item(), dtype=torch.int32, device=dev),
    kv_last_page_len,
    num_heads, num_kv_heads, head_dim, page_size,
)
o = wrapper.run(q, kv_data)
torch.cuda.synchronize()

# reference: SDPA over the paged KV
import torch.nn.functional as F
rep = num_heads // num_kv_heads
Kside = kv_data[:, 0].reshape(-1, num_kv_heads, head_dim)[:seq_len]
Vside = kv_data[:, 1].reshape(-1, num_kv_heads, head_dim)[:seq_len]
Kx = Kside.permute(1, 0, 2).repeat_interleave(rep, 0).unsqueeze(0).float()
Vx = Vside.permute(1, 0, 2).repeat_interleave(rep, 0).unsqueeze(0).float()
Qx = q.float().unsqueeze(2)  # [batch, H, 1, hd]
ref = F.scaled_dot_product_attention(Qx, Kx, Vx).squeeze(2).half()
d = (o.float() - ref.float()).abs()
rel = d.max().item() / ref.float().abs().max().item()
print(f"FLASHINFER SM70 DECODE: rel={rel:.5f} -> "
      f"{'PASS' if rel < 1e-2 else 'FAIL'}")
