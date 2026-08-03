"""INT4 KV mode microtest for flash_attn_sm70_prefill_paged_batched.

Cache layout (uint8): [NB, KVH, BS, HD] bytes per slot =
  [K: HD nibbles in HD/2 bytes | V: HD nibbles in HD/2 bytes]
Element e -> byte e//2, nibble e%2 (low=even, high=odd).
Dequant: x = (nib - 8) * scale.
"""
import torch
import torch.nn.functional as F

import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

torch.manual_seed(0)
dev = "cuda"

D = 128
H = 8
KVH = 2
BS = 16
NB = 64
seq_len = 300
nblocks = (seq_len + BS - 1) // BS
k_scale, v_scale = 0.02, 0.03
scale = 1.0 / (D ** 0.5)

Kf = torch.randn(KVH, seq_len, D, device=dev)
Vf = torch.randn(KVH, seq_len, D, device=dev)

def pack_int4(x, s):
    nib = (x / s).round().add(8).clamp(0, 15).to(torch.uint8)
    return nib[..., 0::2] | (nib[..., 1::2] << 4)

cache = torch.zeros(NB, KVH, BS, D, dtype=torch.uint8, device=dev)
perm = torch.randperm(NB, device=dev)[:nblocks]
block_table = torch.zeros(1, 32, dtype=torch.int32, device=dev)
block_table[0, :nblocks] = perm.int()
for i in range(seq_len):
    b, off = perm[i // BS], i % BS
    cache[b, :, off, : D // 2] = pack_int4(Kf[:, i], k_scale)
    cache[b, :, off, D // 2 :] = pack_int4(Vf[:, i], v_scale)

# Dequantized reference KV (as the kernel sees it)
Kdq = ((Kf / k_scale).round().clamp(-8, 7) * k_scale).half()
Vdq = ((Vf / v_scale).round().clamp(-8, 7) * v_scale).half()

def run(Sq):
    qsl = torch.tensor([0, Sq], dtype=torch.int32, device=dev)
    sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    Q = torch.randn(Sq, H, D, device=dev).half()
    O = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
        Q, cache, block_table, qsl, sl, 1, Sq, scale, True,
        k_scale, v_scale, 2,  # kv_mode=2 INT4
    )
    rep = H // KVH
    Kx = Kdq.repeat_interleave(rep, 0).unsqueeze(0).float()
    Vx = Vdq.repeat_interleave(rep, 0).unsqueeze(0).float()
    Qx = Q.float().permute(1, 0, 2).unsqueeze(0)
    off = seq_len - Sq
    mask = torch.ones(Sq, seq_len, dtype=torch.bool, device=dev).tril(
        diagonal=off
    )
    ref = F.scaled_dot_product_attention(
        Qx, Kx, Vx, attn_mask=mask, scale=scale
    )
    ref = ref.squeeze(0).permute(1, 0, 2).half()
    d = (O.float() - ref.float()).abs()
    denom = ref.float().abs().max().item()
    cos = F.cosine_similarity(
        O.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    print(
        f"Sq={Sq:4d}: max_abs={d.max().item():.5f} "
        f"rel={d.max().item() / denom:.5f} cos={cos:.6f}"
    )
    return d.max().item() / denom, cos

ok = True
for Sq in (1, 16, 64, 300):
    r, c = run(Sq)
    ok &= r < 2e-2 and c > 0.999

# FP8 regression: mode=1 explicit and mode=0 auto
Kq = (Kf / k_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
Vq = (Vf / v_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
cache8 = torch.zeros(
    NB, KVH, BS, 2 * D, dtype=torch.float8_e4m3fn, device=dev
)
for i in range(seq_len):
    b, off = perm[i // BS], i % BS
    cache8[b, :, off, :D] = Kq[:, i]
    cache8[b, :, off, D:] = Vq[:, i]
c8 = cache8.view(torch.uint8)
qsl = torch.tensor([0, 1], dtype=torch.int32, device=dev)
sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
Q1 = torch.randn(1, H, D, device=dev).half()
for mode in (0, 1):
    O8 = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
        Q1, c8, block_table, qsl, sl, 1, 1, scale, True,
        k_scale, v_scale, mode,
    )
    print(
        f"fp8 mode={mode}: Onorm={O8.float().norm().item():.3f} "
        f"finite={torch.isfinite(O8).all().item()}"
    )

print("INT4-KV TEST:", "PASS" if ok else "FAIL")
