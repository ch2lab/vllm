#!/usr/bin/env python3
"""Debug: isolate multi-block vs GQA issue in NVFP4 attention."""
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
HEAD_SIZE = 128
BLOCK_SIZE = 16
FULL_DIM = HEAD_SIZE // 2 + HEAD_SIZE // 16  # 72
softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)

def test_config(num_kv_heads, num_q_heads, seq_len, num_blocks):
    kv_cache = torch.zeros(
        (num_blocks, num_kv_heads, BLOCK_SIZE, 2 * FULL_DIM),
        dtype=torch.uint8, device=device,
    )
    K = torch.randn(seq_len, num_kv_heads, HEAD_SIZE, dtype=torch.float16, device=device)
    V = torch.randn(seq_len, num_kv_heads, HEAD_SIZE, dtype=torch.float16, device=device)
    slot_mapping = torch.arange(seq_len, dtype=torch.long, device=device)
    reshape_and_cache_nvfp4(K, V, kv_cache, slot_mapping)

    dequant = dequantize_nvfp4_kv_cache(kv_cache, HEAD_SIZE)
    K_deq = torch.zeros(seq_len, num_kv_heads, HEAD_SIZE, dtype=torch.float16, device=device)
    V_deq = torch.zeros(seq_len, num_kv_heads, HEAD_SIZE, dtype=torch.float16, device=device)
    for t in range(seq_len):
        blk = t // BLOCK_SIZE
        slot = t % BLOCK_SIZE
        K_deq[t] = dequant[blk, :, slot, :HEAD_SIZE]
        V_deq[t] = dequant[blk, :, slot, HEAD_SIZE:]

    q = torch.randn(1, num_q_heads, HEAD_SIZE, dtype=torch.float16, device=device)
    out = torch.zeros(1, num_q_heads, HEAD_SIZE, dtype=torch.float16, device=device)
    
    # Block table: sequential blocks
    blocks_per_seq = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    bt = torch.zeros(1, max(blocks_per_seq, 4), dtype=torch.int32, device=device)
    for i in range(blocks_per_seq):
        bt[0, i] = i
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    unified_attention_nvfp4(q, kv_cache, out, bt, seq_lens, softmax_scale, causal=True)
    torch.cuda.synchronize()

    # Reference
    num_queries_per_kv = num_q_heads // num_kv_heads
    out_ref = torch.zeros(num_q_heads, HEAD_SIZE, dtype=torch.float32, device=device)
    for h in range(num_q_heads):
        kv_h = h // num_queries_per_kv
        scores = (q[0, h].float() @ K_deq[:, kv_h, :].float().T) * softmax_scale
        probs = torch.softmax(scores, dim=0)
        out_ref[h] = probs @ V_deq[:, kv_h, :].float()

    diff = (out[0].float() - out_ref).abs()
    rel_err = diff.mean() / out_ref.abs().mean()
    return rel_err, diff.max()

# Test 1: 1 head, 1 block (should pass)
r, m = test_config(1, 1, 8, 1)
print(f"1 KV head, 1 Q head, 8 pos, 1 block: rel={r:.6f} max={m:.6f} {'PASS' if r < 0.02 else 'FAIL'}")

# Test 2: 1 head, 2 blocks
r, m = test_config(1, 1, 32, 2)
print(f"1 KV head, 1 Q head, 32 pos, 2 blocks: rel={r:.6f} max={m:.6f} {'PASS' if r < 0.02 else 'FAIL'}")

# Test 3: 2 KV heads, 1 block
r, m = test_config(2, 2, 8, 1)
print(f"2 KV heads, 2 Q heads, 8 pos, 1 block: rel={r:.6f} max={m:.6f} {'PASS' if r < 0.02 else 'FAIL'}")

# Test 4: GQA - 2 KV, 4 Q
r, m = test_config(2, 4, 8, 1)
print(f"2 KV heads, 4 Q heads (GQA), 8 pos: rel={r:.6f} max={m:.6f} {'PASS' if r < 0.02 else 'FAIL'}")

# Test 5: 8 KV, 32 Q, 1 block
r, m = test_config(8, 32, 8, 1)
print(f"8 KV, 32 Q (GQA), 8 pos, 1 block: rel={r:.6f} max={m:.6f} {'PASS' if r < 0.02 else 'FAIL'}")

# Test 6: 8 KV, 32 Q, 2 blocks (the failing case)
r, m = test_config(8, 32, 32, 2)
print(f"8 KV, 32 Q (GQA), 32 pos, 2 blocks: rel={r:.6f} max={m:.6f} {'PASS' if r < 0.02 else 'FAIL'}")

# Test 7: 1 KV, 1 Q, 2 blocks, seq_len=17 (cross boundary)
r, m = test_config(1, 1, 17, 2)
print(f"1 KV, 1 Q, 17 pos, 2 blocks: rel={r:.6f} max={m:.6f} {'PASS' if r < 0.02 else 'FAIL'}")
