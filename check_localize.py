import torch
import torch.nn.functional as F

import vllm._C_stable_libtorch

dev = "cuda"
D, H, KVH, BS = 256, 8, 2, 1072
seq_len = 722
NB = 4
torch.manual_seed(3)
kv = torch.zeros(NB, KVH, BS, 2 * D, device=dev)
kv[..., :D] = 0.0001
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
Kx = K.repeat_interleave(rep, 0).unsqueeze(0).float()
Vx = V.repeat_interleave(rep, 0).unsqueeze(0).float()
Qx = Q.unsqueeze(0).transpose(1, 2).float()
ref = F.scaled_dot_product_attention(Qx, Kx, Vx, scale=0.0625)
ref = ref[0].transpose(0, 1)[:, 0, :]  # [H, D]
d = (O.float() - ref).abs()
print("per-head max err:", [f"{x:.4f}" for x in d.amax(dim=1).tolist()])
print("per-head mean err:", [f"{x:.4f}" for x in d.mean(dim=1).tolist()])
per16 = d.view(H, 16, 16).amax(dim=2)
print("head0 per-16col-block max:", [f"{x:.4f}" for x in per16[0].tolist()])
# ratio check: is O a scaled ref?
ratio = (O.float() / ref).mean(dim=1)
print("mean O/ref ratio per head:", [f"{x:.4f}" for x in ratio.tolist()])
