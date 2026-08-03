import torch
import torch.nn.functional as F

import vllm._C_stable_libtorch

dev = "cuda"
D, H, KVH, BS = 256, 8, 2, 1072
seq_len = 722
NB = 4
torch.manual_seed(3)

kv = torch.randn(NB, KVH, BS, 2 * D, device=dev)
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "mixed"
if MODE == "mixed":
    kv = kv * torch.where(
        torch.rand_like(kv) < 0.7,
        torch.full_like(kv, 0.002),
        torch.full_like(kv, 1.0),
    )
elif MODE == "tiny":
    kv = kv * 0.002
elif MODE == "ones":
    kv = torch.sign(kv) * 0.5
elif MODE == "rand":
    pass
elif MODE == "Kbig":  # K unit, V tiny
    kv = kv * 0.002
    kv[..., :D] = torch.randn(NB, KVH, BS, D, device=dev)
elif MODE == "Vbig":  # K tiny, V unit
    kv = kv * 0.002
    kv[..., D:] = torch.randn(NB, KVH, BS, D, device=dev)
elif MODE == "Qsmall":
    pass
kv8 = kv.clamp(-448, 448).to(torch.float8_e4m3fn)

bt = torch.tensor([list(range(4)) + [-1] * 4], dtype=torch.int32, device=dev)
qsl = torch.tensor([0, 1], dtype=torch.int32, device=dev)
sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
Q = (torch.randn(1, H, D, device=dev) * 1.0).half()

O_batched = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
    Q, kv8, bt, qsl, sl, 1, 1, 0.0625, True, 1.0, 1.0
)[0]  # [H, D]
Qp = Q.unsqueeze(2)  # [1,H,1,D]
O_paged = torch.ops._C.flash_attn_sm70_prefill_paged(
    Qp, kv8, bt, 0.0625, True, seq_len, 1.0, 1.0
)[0, :, 0, :]  # [H, D]
print("batched vs paged max diff:", (O_batched - O_paged).abs().max().item())

kv8h = kv8.to(torch.float16)
toks = [kv8h[i // BS, :, i % BS, :] for i in range(seq_len)]
t = torch.stack(toks, 1).float()
K, V = t[..., :D], t[..., D:]
rep = H // KVH
Kx = K.repeat_interleave(rep, 0).unsqueeze(0)
Vx = V.repeat_interleave(rep, 0).unsqueeze(0)
Qx = Q.unsqueeze(0).transpose(1, 2).float()
ref = F.scaled_dot_product_attention(Qx, Kx, Vx, scale=0.0625)
ref = ref[0].transpose(0, 1)  # [H, 1, D] -> no; [1,H,1,D] -> [H,D]
ref = ref[:, 0, :]
print(
    "batched vs SDPA subnormal-heavy:",
    (O_batched.float() - ref).abs().max().item(),
    (O_batched.float() - ref).abs().mean().item(),
    "refnorm", ref.norm().item(),
)
