"""NVFP4 KV mode (kv_mode=3) microtest, upstream-compatible layout.

Cache: uint8 [NB, 2*KVH, BS, HD/2 + HD/16]
  head rows 0..KVH-1 = K, KVH..2KVH-1 = V
  slot = [e2m1 nibbles (HD/2 B) | fp8-e4m3 block scales (HD/16 B)]
Dequant: x = lut[nib] * fp8(block_scale) * global_scale
"""
import torch
import torch.nn.functional as F

import vllm._custom_ops  # noqa: F401

torch.manual_seed(0)
dev = "cuda"

D = 128
H = 8
KVH = 2
BS = 16
NB = 64
L = 300
nb_used = (L + BS - 1) // BS
scale = 1.0 / (D ** 0.5)
FULL = D // 2 + D // 16
LEVELS = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=dev)
GS = 0.25  # global scale (k_scale/v_scale)


def nvfp4_quant(x, gs):
    """x: [..., D] -> (packed uint8 [..., D//2], scale bytes [..., D//16], deq half)"""
    shape = x.shape
    xf = x.float().reshape(*shape[:-1], shape[-1] // 16, 16)
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    sfp8 = (amax / (6.0 * gs)).to(torch.float8_e4m3fn)
    s = sfp8.float() * gs
    y = xf / s
    mag_idx = (y.abs().unsqueeze(-1) - LEVELS).abs().argmin(-1)
    sign = (y < 0).to(torch.uint8)
    nib = mag_idx.to(torch.uint8) | (sign << 3)
    deq = LEVELS[mag_idx.long()] * s * torch.where(sign > 0, -1.0, 1.0)
    nib = nib.reshape(*shape[:-1], shape[-1])
    packed = nib[..., 0::2] | (nib[..., 1::2] << 4)
    return packed, sfp8.view(torch.uint8).squeeze(-1), deq.reshape(shape).half()


Kf = torch.randn(KVH, L, D, device=dev) * 1.5
Vf = torch.randn(KVH, L, D, device=dev) * 1.5
Kp, Ks, Kdq = nvfp4_quant(Kf, GS)
Vp, Vs, Vdq = nvfp4_quant(Vf, GS)

cache = torch.zeros(NB, 2 * KVH, BS, FULL, dtype=torch.uint8, device=dev)
perm = torch.randperm(NB, device=dev)[:nb_used]
block_table = torch.zeros(1, 32, dtype=torch.int32, device=dev)
block_table[0, :nb_used] = perm.int()
# Engine region layout: page = [K data | K scales | V data | V scales],
# NHD within each region.
flat = cache.view(-1)
page = 2 * KVH * BS * FULL
DD, SD = D // 2, D // 16
hidx = torch.arange(KVH, device=dev)
for i in range(L):
    b, off = perm[i // BS].item(), i % BS
    base = b * page
    data_idx = (base + hidx * (BS * DD) + off * DD)[:, None] + torch.arange(
        DD, device=dev
    )
    scale_idx = (
        base + KVH * BS * DD + hidx * (BS * SD) + off * SD
    )[:, None] + torch.arange(SD, device=dev)
    flat[data_idx] = Kp[:, i]
    flat[scale_idx] = Ks[:, i]
    flat[data_idx + KVH * BS * FULL] = Vp[:, i]
    flat[scale_idx + KVH * BS * FULL] = Vs[:, i]

ok = True
for Sq in (1, 16, 64, 300):
    qsl = torch.tensor([0, Sq], dtype=torch.int32, device=dev)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    Q = torch.randn(Sq, H, D, device=dev).half()
    O = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
        Q, cache[:, :KVH], block_table, qsl, sl, 1, Sq, scale, True,
        GS, GS, 3,  # kv_mode=3 NVFP4
    )
    rep = H // KVH
    Kx = Kdq.repeat_interleave(rep, 0).unsqueeze(0).float()
    Vx = Vdq.repeat_interleave(rep, 0).unsqueeze(0).float()
    Qx = Q.float().permute(1, 0, 2).unsqueeze(0)
    mask = torch.ones(Sq, L, dtype=torch.bool, device=dev).tril(
        diagonal=L - Sq
    )
    ref = F.scaled_dot_product_attention(
        Qx, Kx, Vx, attn_mask=mask, scale=scale
    )
    ref = ref.squeeze(0).permute(1, 0, 2).half()
    d = (O.float() - ref.float()).abs()
    rel = d.max().item() / ref.float().abs().max().item()
    cos = F.cosine_similarity(
        O.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    print(f"Sq={Sq:4d}: max_abs={d.max().item():.5f} rel={rel:.5f} cos={cos:.6f}")
    ok &= rel < 2e-2 and cos > 0.999

print("NVFP4-KV TEST:", "PASS" if ok else "FAIL")
