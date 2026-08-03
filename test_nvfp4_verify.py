#!/usr/bin/env python3
"""Final verification: NVFP4 layout fix with proper reference."""
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

NUM_KV_HEADS = 8
NUM_Q_HEADS = 32
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 4
SEQ_LEN = 32
FULL_DIM = HEAD_SIZE // 2 + HEAD_SIZE // 16  # 72
softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)

print(f"Config: {NUM_KV_HEADS} KV heads, {NUM_Q_HEADS} Q heads, seq_len={SEQ_LEN}")
print(f"Cache: ({NUM_BLOCKS}, {NUM_KV_HEADS}, {BLOCK_SIZE}, {2*FULL_DIM})")

kv_cache = torch.zeros(
    (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * FULL_DIM),
    dtype=torch.uint8, device=device,
)

K = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
V = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
slot_mapping = torch.arange(SEQ_LEN, dtype=torch.long, device=device)

reshape_and_cache_nvfp4(K, V, kv_cache, slot_mapping)
print("Store OK")

# Dequantize for reference
dequant = dequantize_nvfp4_kv_cache(kv_cache, HEAD_SIZE)
# dequant: [NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2*HEAD_SIZE]

# Reconstruct per-position K/V
K_deq = torch.zeros(SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
V_deq = torch.zeros(SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
for t in range(SEQ_LEN):
    blk = t // BLOCK_SIZE
    slot = t % BLOCK_SIZE
    K_deq[t] = dequant[blk, :, slot, :HEAD_SIZE]
    V_deq[t] = dequant[blk, :, slot, HEAD_SIZE:]

# Run attention kernel
q = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
out = torch.zeros(1, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)

blocks_per_seq = (SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
bt = torch.zeros(1, max(blocks_per_seq, 4), dtype=torch.int32, device=device)
for i in range(blocks_per_seq):
    bt[0, i] = i
seq_lens = torch.tensor([SEQ_LEN], dtype=torch.int32, device=device)

print(f"Block table: {bt[0].tolist()}")
print(f"Seq lens: {seq_lens.tolist()}")

unified_attention_nvfp4(q, kv_cache, out, bt, seq_lens, softmax_scale, causal=True)
torch.cuda.synchronize()

# Reference: standard attention using dequantized K/V
num_queries_per_kv = NUM_Q_HEADS // NUM_KV_HEADS
out_ref = torch.zeros(NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float32, device=device)
for h in range(NUM_Q_HEADS):
    kv_h = h // num_queries_per_kv
    scores = (q[0, h].float() @ K_deq[:, kv_h, :].float().T) * softmax_scale
    probs = torch.softmax(scores, dim=0)
    out_ref[h] = probs @ V_deq[:, kv_h, :].float()

diff = (out[0].float() - out_ref).abs()
rel_err = diff.mean() / out_ref.abs().mean()
print(f"\nKernel output norm: {out[0].float().norm():.4f}")
print(f"Reference norm: {out_ref.norm():.4f}")
print(f"Mean abs diff: {diff.mean():.6f}")
print(f"Relative error: {rel_err:.6f}")
print(f"Max abs diff: {diff.max():.6f}")

# Per-head analysis
print("\nPer-head relative errors:")
for h in range(0, NUM_Q_HEADS, 4):
    kv_h = h // num_queries_per_kv
    d = (out[0, h].float() - out_ref[h]).abs()
    r = d.mean() / out_ref[h].abs().mean()
    print(f"  head {h:2d} (kv={kv_h}): rel={r:.6f} max={d.max():.6f}")

if rel_err < 0.02:
    print("\nPASS: NVFP4 attention kernel matches reference!")
else:
    print(f"\nFAIL: rel_err={rel_err:.6f}")
    # Debug: check head 0 scores
    kv_h = 0
    scores_ref = (q[0, 0].float() @ K_deq[:, 0, :].float().T) * softmax_scale
    print(f"  Reference scores head 0: {scores_ref[:8].tolist()}")
    print(f"  Reference probs head 0: {torch.softmax(scores_ref, dim=0)[:8].tolist()}")
