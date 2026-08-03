"""Integration test: SM70 backend _nvfp4_write + WMMA nvfp4 read kernel."""
import types

import torch
import torch.nn.functional as F

import vllm._custom_ops  # noqa: F401
from vllm.v1.attention.backends.sm70_wmma_attn import SM70WMMAAttentionImpl

torch.manual_seed(0)
dev = "cuda"

D = 128
H = 8
KVH = 2
BS = 16
NB = 64
L = 300
nb_used = (L + BS - 1) // BS
FULL = D // 2 + D // 16
attn_scale = 1.0 / (D ** 0.5)

cache = torch.zeros(NB, 2 * KVH, BS, FULL, dtype=torch.uint8, device=dev)
perm = torch.randperm(NB, device=dev)[:nb_used]
block_table = torch.zeros(1, 32, dtype=torch.int32, device=dev)
block_table[0, :nb_used] = perm.int()

Kf = torch.randn(L, KVH, D, device=dev, dtype=torch.half) * 1.5
Vf = torch.randn(L, KVH, D, device=dev, dtype=torch.half) * 1.5

slot_mapping = torch.zeros(L, dtype=torch.long, device=dev)
for i in range(L):
    slot_mapping[i] = perm[i // BS].item() * BS + i % BS

stub = types.SimpleNamespace(head_size=D)
layer = types.SimpleNamespace(
    _k_scale=torch.tensor(0.25, device=dev),
    _v_scale=torch.tensor(0.25, device=dev),
)
SM70WMMAAttentionImpl._nvfp4_write(
    stub, layer, Kf, Vf, cache, slot_mapping
)
torch.cuda.synchronize()

# Dequantized reference: replicate the quant math, then invert it
LEVELS = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=dev)

def qdq(x, gs):
    xf = x.float().view(L, KVH, D // 16, 16)
    sf = (xf.abs().amax(-1, keepdim=True) / (6.0 * gs)).to(torch.float8_e4m3fn)
    y = xf / (sf.float() * gs).clamp_min(1e-30)
    idx = (y.abs().unsqueeze(-1) - LEVELS).abs().argmin(-1)
    deq = LEVELS[idx] * (sf.float() * gs) * torch.where(y < 0, -1.0, 1.0)
    return deq.view(L, KVH, D).half()

Kdq = qdq(Kf, 0.25)
Vdq = qdq(Vf, 0.25)

ok_kernel = True
ok_quant = True
for Sq in (1, 16, 64, 300):
    qsl = torch.tensor([0, Sq], dtype=torch.int32, device=dev)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    Q = torch.randn(Sq, H, D, device=dev).half()
    O = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
        Q, cache[:, :KVH], block_table, qsl, sl, 1, Sq, attn_scale, True,
        0.25, 0.25, 3,
    )
    rep = H // KVH
    mask = torch.ones(Sq, L, dtype=torch.bool, device=dev).tril(
        diagonal=L - Sq
    )
    Qx = Q.float().permute(1, 0, 2).unsqueeze(0)
    res = {}
    for name, KK, VV in (("deq", Kdq, Vdq), ("fp16", Kf, Vf)):
        Kx = KK.permute(1, 0, 2).repeat_interleave(rep, 0).unsqueeze(0).float()
        Vx = VV.permute(1, 0, 2).repeat_interleave(rep, 0).unsqueeze(0).float()
        ref = F.scaled_dot_product_attention(
            Qx, Kx, Vx, attn_mask=mask, scale=attn_scale
        )
        res[name] = ref.squeeze(0).permute(1, 0, 2).half()
    d = (O.float() - res["deq"].float()).abs()
    rel = d.max().item() / res["deq"].float().abs().max().item()
    cos16 = F.cosine_similarity(
        O.float().flatten(), res["fp16"].float().flatten(), dim=0
    ).item()
    print(f"Sq={Sq:4d}: vs_deq rel={rel:.5f} | vs_fp16 cos={cos16:.5f}")
    ok_kernel &= rel < 5e-3
    ok_quant &= cos16 > 0.97

print("KERNEL CORRECTNESS:", "PASS" if ok_kernel else "FAIL")
print("QUANT NOISE ACCEPTABLE:", "PASS" if ok_quant else "FAIL")
