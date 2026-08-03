import torch
import torch.nn.functional as F

import vllm._C_stable_libtorch

dev = "cuda"
D, H, KVH, BS = 256, 8, 2, 1072
seq_len = 722
NB = 4


def run(seed, kmag, vmag, qmag, Sq=1):
    torch.manual_seed(seed)
    kv16 = torch.zeros(NB, KVH, BS, 2 * D, device=dev)
    kv16[..., :D] = torch.randn(NB, KVH, BS, D, device=dev) * kmag
    kv16[..., D:] = torch.randn(NB, KVH, BS, D, device=dev) * vmag
    kv8 = kv16.clamp(-448, 448).to(torch.float8_e4m3fn)
    bt = torch.tensor([list(range(4)) + [-1] * 4], dtype=torch.int32, device=dev)
    qsl = torch.tensor([0, Sq], dtype=torch.int32, device=dev)
    sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    Q = (torch.randn(Sq, H, D, device=dev) * qmag).half()
    O8 = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
        Q, kv8, bt, qsl, sl, 1, Sq, 0.0625, True, 1.0, 1.0
    )
    # reference: gather 722 tokens
    kv8h = kv8.to(torch.float16)
    toks = [kv8h[i // BS, :, i % BS, :] for i in range(seq_len)]
    t = torch.stack(toks, 1).float()
    K, V = t[..., :D], t[..., D:]
    rep = H // KVH
    Kx = K.repeat_interleave(rep, 0).unsqueeze(0)
    Vx = V.repeat_interleave(rep, 0).unsqueeze(0)
    Qx = Q.permute(1, 0, 2).unsqueeze(0).float()
    ref = F.scaled_dot_product_attention(Qx, Kx, Vx, scale=0.0625,
                                         is_causal=(Sq > 1))
    ref = ref.squeeze(0).permute(1, 0, 2)[:Sq]  # [Sq,H,D]
    d = (O8.float() - ref.float()).abs()
    print(
        f"Sq={Sq} seed={seed} kmag={kmag}: "
        f"max={d.max().item():.5f} mean={d.mean().item():.6f} "
        f"refnorm={ref.norm().item():.2f}"
    )


run(1, 0.5, 0.5, 0.5, Sq=1)
run(1, 2.0, 2.0, 2.0, Sq=1)
run(2, 2.0, 2.0, 2.0, Sq=1)
run(1, 0.5, 0.5, 0.5, Sq=722)
run(1, 2.0, 2.0, 2.0, Sq=722)
