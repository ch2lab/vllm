import torch
import torch.nn.functional as F

import vllm._C_stable_libtorch

dev = "cuda"
D, H, KVH, BS = 256, 8, 2, 1072
seq_len = 722
NB = 4
torch.manual_seed(3)


def case(name, Kmag, kconst=False):
    kv = torch.zeros(NB, KVH, BS, 2 * D, device=dev)
    if kconst:
        kv[..., :D] = Kmag
    else:
        kv[..., :D] = torch.randn(NB, KVH, BS, D, device=dev) * Kmag
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
    ref = ref[0].transpose(0, 1)[:, 0, :]
    d = (O.float() - ref).abs()
    print(f"{name}: max={d.max().item():.6f} mean={d.mean().item():.6f}")


case("Kconst_tiny", 0.0001, kconst=True)
case("Krand_0.0003", 0.0003)
case("Krand_0.002", 0.002)
case("Krand_0.01", 0.01)
case("Krand_0.05", 0.05)
case("Krand_0.1", 0.1)
