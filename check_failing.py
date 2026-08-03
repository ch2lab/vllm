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
print("torch dequant of K byte:", kv8.to(torch.float16).flatten()[0].item())
print("K bytes unique:", kv8[..., :D].flatten().unique().tolist()[:10])

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
sc = (Qx @ Kx.transpose(-1, -2)) * 0.0625
print("scores min/max:", sc.min().item(), sc.max().item())
p = torch.softmax(sc, dim=-1)
print("p min/max:", p.min().item(), p.max().item())
ref = (p @ Vx)[0].transpose(0, 1)[:, 0, :]
d = (O.float() - ref).abs()
print("err per head:", [f"{x:.4f}" for x in d.amax(dim=1).tolist()])

# uniform-softmax prediction: mean of V over positions
meankv1 = V[1].mean(dim=0)  # [S, D] mean over positions
pred0 = (p[0, 0] @ Vx[0, 0])
print("head0 ref vs kernel:", ref[0, :3].tolist(), O[0, :3].tolist())
print("head4 ref vs kernel:", ref[4, :3].tolist(), O[4, :3].tolist())
print("mean V head0:", V[0].mean(dim=0)[:3].tolist())
print("mean V head1:", V[1].mean(dim=0)[:3].tolist())
