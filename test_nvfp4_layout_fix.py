#!/usr/bin/env python3
"""Unit test: verify NVFP4 store+attention kernels with correct cache layout."""
import torch
import sys
sys.path.insert(0, "/data/src/vllm")

from vllm.v1.attention.ops.triton_nvfp4_kv_cache import (
    reshape_and_cache_nvfp4,
    dequantize_nvfp4_kv_cache,
)
from vllm.v1.attention.ops.nvfp4_attention import unified_attention_nvfp4

torch.manual_seed(42)
device = "cuda"

# Model params (Qwen3.5-4B-like)
NUM_KV_HEADS = 8
NUM_Q_HEADS = 32
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 4
SEQ_LEN = 32  # 2 blocks per seq

FULL_DIM = HEAD_SIZE // 2 + HEAD_SIZE // 16  # 64 + 8 = 72

print(f"Cache shape: ({NUM_BLOCKS}, {NUM_KV_HEADS}, {BLOCK_SIZE}, {2*FULL_DIM})")
print(f"FULL_DIM={FULL_DIM}, HEAD_SIZE={HEAD_SIZE}")

# Allocate cache with CORRECT layout: (num_blocks, num_kv_heads, block_size, 2*full_dim)
kv_cache = torch.zeros(
    (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * FULL_DIM),
    dtype=torch.uint8, device=device,
)

# Create random K/V: [num_tokens, num_kv_heads, head_size]
num_tokens = SEQ_LEN
K = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
V = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)

# Slot mapping: sequential slots
slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

# Store
print("\n[1] Testing store kernel...")
reshape_and_cache_nvfp4(K, V, kv_cache, slot_mapping)
print("  Store OK (no crash)")

# Dequantize and check round-trip error
print("\n[2] Testing dequantize round-trip...")
dequant = dequantize_nvfp4_kv_cache(kv_cache, HEAD_SIZE)
# dequant shape: [num_blocks, num_kv_heads, block_size, 2*head_size]
# K is at [..., :head_size], V is at [..., head_size:]

# Reconstruct K/V from dequantized cache
K_deq = torch.zeros_like(K)
V_deq = torch.zeros_like(V)
for t in range(num_tokens):
    blk = t // BLOCK_SIZE
    slot = t % BLOCK_SIZE
    K_deq[t] = dequant[blk, :, slot, :HEAD_SIZE]
    V_deq[t] = dequant[blk, :, slot, HEAD_SIZE:]

k_err = (K.float() - K_deq.float()).abs()
v_err = (V.float() - V_deq.float()).abs()
k_rel = k_err.mean() / K.float().abs().mean()
v_rel = v_err.mean() / V.float().abs().mean()
print(f"  K mean abs error: {k_err.mean():.4f}, relative: {k_rel:.4f}")
print(f"  V mean abs error: {v_err.mean():.4f}, relative: {v_rel:.4f}")
# FP4 E2M1 has ~12.5% max relative error per element, expect <30% mean
assert k_rel < 0.5, f"K round-trip error too high: {k_rel}"
assert v_rel < 0.5, f"V round-trip error too high: {v_rel}"
print("  PASS: round-trip error within FP4 tolerance")

# Test attention kernel
print("\n[3] Testing attention kernel (decode)...")
NUM_SEQS = 1
q = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
out = torch.zeros(1, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
block_table = torch.tensor([[0, 1, 0, 0]], dtype=torch.int32, device=device)
seq_lens = torch.tensor([SEQ_LEN], dtype=torch.int32, device=device)
softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)

unified_attention_nvfp4(
    q, kv_cache, out, block_table, seq_lens, softmax_scale, causal=True
)
torch.cuda.synchronize()
print(f"  Output norm: {out.float().norm():.4f}")
assert out.float().norm() > 0.1, "Output is zero/near-zero — kernel broken"
print("  PASS: non-zero output")

# Reference: compute attention in FP32 using dequantized K/V
print("\n[4] Comparing against FP32 reference attention...")
# For decode: q attends to all SEQ_LEN positions
q_ref = q[0].float()  # [num_q_heads, head_size]
K_ref = K_deq[:SEQ_LEN].float()  # [seq_len, num_kv_heads, head_size]
V_ref = V_deq[:SEQ_LEN].float()  # [seq_len, num_kv_heads, head_size]

num_queries_per_kv = NUM_Q_HEADS // NUM_KV_HEADS
out_ref = torch.zeros(NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float32, device=device)

for h in range(NUM_Q_HEADS):
    kv_h = h // num_queries_per_kv
    # scores: [seq_len]
    scores = torch.einsum("d,sh->s", q_ref[h], K_ref[:, kv_h, :]) * softmax_scale
    probs = torch.softmax(scores, dim=0)
    out_ref[h] = torch.einsum("s,sh->h", probs, V_ref[:, kv_h, :])

# Compare
diff = (out[0].float() - out_ref).abs()
rel_err = diff.mean() / out_ref.abs().mean()
print(f"  Mean abs diff: {diff.mean():.6f}")
print(f"  Relative error: {rel_err:.6f}")
print(f"  Max abs diff: {diff.max():.6f}")
# The kernel and reference both use the same quantized data, so should match closely
# (differences only from FP16 vs FP32 accumulation)
assert rel_err < 0.05, f"Attention output mismatch: rel_err={rel_err}"
print("  PASS: attention output matches reference")

print("\n" + "=" * 50)
print("ALL TESTS PASSED - NVFP4 layout fix verified!")
print("=" * 50)
