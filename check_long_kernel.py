import torch
import torch.nn.functional as F

import vllm._C_stable_libtorch

torch.manual_seed(1)
dev = "cuda"
D, H, KVH, BS = 256, 8, 2, 1072
seq_len = 721
NB = 4
kv16 = (torch.randn(NB, KVH, BS, 2 * D, device=dev) * 0.5).half()
kv8 = kv16.float().clamp(-448, 448).to(torch.float8_e4m3fn)
bt = torch.tensor([list(range(4)) + [-1] * 4], dtype=torch.int32, device=dev)
qsl = torch.tensor([0, seq_len], dtype=torch.int32, device=dev)
sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
Q = (torch.randn(seq_len, H, D, device=dev) * 0.5).half()
O8 = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
    Q, kv8, bt, qsl, sl, 1, seq_len, 0.088, True, 1.0, 1.0
)
kv8h = kv8.to(torch.float16)
toks = [kv8h[i // BS, :, i % BS, :] for i in range(seq_len)]
t = torch.stack(toks, 1)
K, V = t[..., :D], t[..., D:]
rep = H // KVH
Kx = K.repeat_interleave(rep, 0).unsqueeze(0)
Vx = V.repeat_interleave(rep, 0).unsqueeze(0)
Qx = Q.unsqueeze(0).transpose(1, 2)
print("QKV:", tuple(Qx.shape), tuple(Kx.shape), tuple(Vx.shape))
ref = F.scaled_dot_product_attention(Qx, Kx, Vx, is_causal=True)
ref = ref.transpose(1, 2)
print("ref:", tuple(ref.shape))
d = (O8.float() - ref.squeeze(0).float()).abs()
print("O8 vs SDPA causal: max", d.max().item(), "mean", d.mean().item())
rowmax = d.amax(dim=(1, 2))
print("row err first:", [f"{x:.3f}" for x in rowmax[:8].tolist()])
print("row err last:", [f"{x:.3f}" for x in rowmax[-8:].tolist()])
bad = (rowmax > 0.1).nonzero().flatten()
print("rows err>0.1 count:", bad.numel(), bad[:12].tolist())
