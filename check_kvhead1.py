import torch

import vllm._C_stable_libtorch

dev = "cuda"
D, H, KVH, BS = 256, 8, 2, 1072
seq_len = 722
NB = 4
kv = torch.zeros(NB, KVH, BS, 2 * D, device=dev)
kv[..., :D] = 0.0001
pos = torch.arange(BS, device=dev).float()
for b in range(NB):
    kv[b, 1, :, D:] = ((pos + b * BS) / 100.0)[:, None]  # kv head 1 only
kv8 = kv.clamp(-448, 448).to(torch.float8_e4m3fn)
bt = torch.tensor([list(range(4)) + [-1] * 4], dtype=torch.int32, device=dev)
qsl = torch.tensor([0, 1], dtype=torch.int32, device=dev)
sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
Q = torch.ones(1, H, D, device=dev).half()
O = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
    Q, kv8, bt, qsl, sl, 1, 1, 0.0625, True, 1.0, 1.0
)
print("expected ~3.605 for heads 4-7, 0 for heads 0-3")
print("O per head:", [f"{O[0,h,0].item():.4f}" for h in range(H)])

# sweep L to find the boundary where it breaks
for L in (32, 64, 128, 256, 512, 640, 672, 704, 722):
    sl2 = torch.tensor([L], dtype=torch.int32, device=dev)
    O2 = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
        Q, kv8, bt, qsl, sl2, 1, 1, 0.0625, True, 1.0, 1.0
    )
    exp = (L - 1) / 2 / 100.0
    got = O2[0, 4, 0].item()
    print(f"L={L}: head4 O={got:.4f} expected={exp:.4f} ratio={got/exp if exp else 0:.4f}")
