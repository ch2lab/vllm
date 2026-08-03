#!/usr/bin/env python3
"""Debug: NVFP4 attention kernel with minimal cases."""
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

NUM_KV_HEADS = 1
NUM_Q_HEADS = 1
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 2
FULL_DIM = HEAD_SIZE // 2 + HEAD_SIZE // 16  # 72

# --- Test A: 1 KV position, output should = dequantized V[0] ---
print("=" * 60)
print("Test A: 1 KV position (softmax trivial, output = V[0])")
print("=" * 60)

kv_cache = torch.zeros(
    (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * FULL_DIM),
    dtype=torch.uint8, device=device,
)

K = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
V = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
slot_mapping = torch.zeros(1, dtype=torch.long, device=device)

reshape_and_cache_nvfp4(K, V, kv_cache, slot_mapping)

# Dequantize to get reference V
dequant = dequantize_nvfp4_kv_cache(kv_cache, HEAD_SIZE)
V_ref = dequant[0, 0, 0, HEAD_SIZE:]  # [head_size]
print(f"  V_ref norm: {V_ref.float().norm():.4f}")
print(f"  V_ref[:8]: {V_ref[:8].tolist()}")

# Run attention with seq_len=1
q = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
out = torch.zeros(1, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
block_table = torch.tensor([[0, 0]], dtype=torch.int32, device=device)
seq_lens = torch.tensor([1], dtype=torch.int32, device=device)
softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)

unified_attention_nvfp4(q, kv_cache, out, block_table, seq_lens, softmax_scale, causal=True)
torch.cuda.synchronize()

print(f"  out norm: {out[0,0].float().norm():.4f}")
print(f"  out[:8]:  {out[0,0,:8].tolist()}")

diff = (out[0, 0].float() - V_ref.float()).abs()
print(f"  max diff from V_ref: {diff.max():.6f}")
print(f"  mean diff: {diff.mean():.6f}")
if diff.max() < 0.01:
    print("  PASS: output matches V[0]")
else:
    print("  FAIL: output does NOT match V[0]")
    # Check if it matches K instead (layout swap bug)
    K_ref = dequant[0, 0, 0, :HEAD_SIZE]
    diff_k = (out[0, 0].float() - K_ref.float()).abs()
    print(f"  max diff from K_ref: {diff_k.max():.6f}")
    if diff_k.max() < 0.01:
        print("  BUG: output matches K instead of V!")

# --- Test B: uniform Q (all ones), 4 positions ---
print("\n" + "=" * 60)
print("Test B: 4 KV positions, check score computation")
print("=" * 60)

kv_cache2 = torch.zeros(
    (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * FULL_DIM),
    dtype=torch.uint8, device=device,
)
SEQ_LEN = 4
K2 = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
V2 = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
slot_mapping2 = torch.arange(SEQ_LEN, dtype=torch.long, device=device)
reshape_and_cache_nvfp4(K2, V2, kv_cache2, slot_mapping2)

dequant2 = dequantize_nvfp4_kv_cache(kv_cache2, HEAD_SIZE)
K2_deq = dequant2[0, 0, :SEQ_LEN, :HEAD_SIZE].float()  # [4, head_size]
V2_deq = dequant2[0, 0, :SEQ_LEN, HEAD_SIZE:].float()  # [4, head_size]

q2 = torch.ones(1, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
out2 = torch.zeros(1, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
block_table2 = torch.tensor([[0, 0]], dtype=torch.int32, device=device)
seq_lens2 = torch.tensor([SEQ_LEN], dtype=torch.int32, device=device)

unified_attention_nvfp4(q2, kv_cache2, out2, block_table2, seq_lens2, softmax_scale, causal=True)
torch.cuda.synchronize()

# Reference
scores = (q2[0, 0].float() @ K2_deq.T) * softmax_scale  # [4]
probs = torch.softmax(scores, dim=0)  # [4]
out_ref = probs @ V2_deq  # [head_size]

diff2 = (out2[0, 0].float() - out_ref).abs()
rel2 = diff2.mean() / out_ref.abs().mean()
print(f"  Scores (ref): {scores.tolist()}")
print(f"  Probs (ref): {probs.tolist()}")
print(f"  out_ref norm: {out_ref.norm():.4f}")
print(f"  kernel out norm: {out2[0,0].float().norm():.4f}")
print(f"  mean diff: {diff2.mean():.6f}")
print(f"  rel error: {rel2:.6f}")
if rel2 < 0.02:
    print("  PASS")
else:
    print("  FAIL")
    # Print first few elements for comparison
    print(f"  out_ref[:8]: {out_ref[:8].tolist()}")
    print(f"  kernel[:8]:  {out2[0,0,:8].float().tolist()}")
