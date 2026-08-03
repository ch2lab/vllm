import torch
import torch.nn.functional as F

import vllm._C_stable_libtorch

dev = "cuda"
D, H, KVH, BS = 256, 8, 2, 1072
seq_len = 722
NB = 4
torch.manual_seed(3)

kv = torch.randn(NB, KVH, BS, 2 * D, device=dev) * 0.002
kv[..., D:] = torch.randn(NB, KVH, BS, D, device=dev)
kv8 = kv.clamp(-448, 448).to(torch.float8_e4m3fn)
bt = torch.tensor([list(range(4)) + [-1] * 4], dtype=torch.int32, device=dev)
qsl = torch.tensor([0, 1], dtype=torch.int32, device=dev)
sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
Q = torch.randn(1, H, D, device=dev).half()

O = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
    Q, kv8, bt, qsl, sl, 1, 1, 0.0625, True, 1.0, 1.0
)[0]

kv8h = kv8.to(torch.float16)
toks = [kv8h[i // BS, :, i % BS, :] for i in range(seq_len)]
t = torch.stack(toks, 1)
K, V = t[..., :D], t[..., D:]
rep = H // KVH
Kx = K.repeat_interleave(rep, 0).unsqueeze(0)
Vx = V.repeat_interleave(rep, 0).unsqueeze(0)
Qx = Q.unsqueeze(0).transpose(1, 2)

ref32 = F.scaled_dot_product_attention(
    Qx.float(), Kx.float(), Vx.float(), scale=0.0625
)[0].transpose(0, 1)[:, 0, :]
ref16 = F.scaled_dot_product_attention(
    Qx.half(), Kx.half(), Vx.half(), scale=0.0625
)[0].transpose(0, 1)[:, 0, :]

# manual exact fp32 with explicit softmax
sc = (Qx.float() @ Kx.float().transpose(-1, -2)) * 0.0625
p = torch.softmax(sc, dim=-1)
refman = (p @ Vx.float())[0].transpose(0, 1)[:, 0, :]

for name, r in [("sdpa32", ref32), ("sdpa16", ref16), ("manual32", refman)]:
    d = (O.float() - r).abs()
    print(f"{name}: max={d.max().item():.6f} mean={d.mean().item():.6f}")
print("sdpa32 vs manual32:", (ref32 - refman).abs().max().item())
print("sdpa16 vs manual32:", (ref16.float() - refman).abs().max().item())
