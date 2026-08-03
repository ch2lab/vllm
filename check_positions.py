import torch

import vllm._C_stable_libtorch

dev = "cuda"
D, H, KVH, BS = 256, 8, 2, 1072
seq_len = 722
NB = 4
kv = torch.zeros(NB, KVH, BS, 2 * D, device=dev)
# K tiny, V encodes position: V[pos] = pos/100
pos = torch.arange(BS, device=dev).float()
for b in range(NB):
    kv[b, :, :, D:] = (pos[:, None] + b * BS) / 100.0
kv[..., :D] = 0.0001
kv8 = kv.clamp(-448, 448).to(torch.float8_e4m3fn)
bt = torch.tensor([list(range(4)) + [-1] * 4], dtype=torch.int32, device=dev)
qsl = torch.tensor([0, 1], dtype=torch.int32, device=dev)
sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
Q = torch.ones(1, H, D, device=dev).half()

O = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
    Q, kv8, bt, qsl, sl, 1, 1, 0.0625, True, 1.0, 1.0
)
# Expected with uniform softmax: mean of positions 0..721 = 360.5 / 100 = 3.605
print("O[0,0,0] =", O[0, 0, 0].item(), "expected ~3.605")
print("O per head:", O[0, :, 0].tolist())

# Now with seq_len crossing block boundary differently
for L in (700, 722, 1000, 1100, 2000):
    sl2 = torch.tensor([L], dtype=torch.int32, device=dev)
    O2 = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
        Q, kv8, bt, qsl, sl2, 1, 1, 0.0625, True, 1.0, 1.0
    )
    exp = (L - 1) / 2 / 100.0
    print(f"L={L}: O={O2[0,0,0].item():.4f} expected={exp:.4f}")
