# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for gathering paged KV cache into contiguous tensors.

NOTE: Superseded by flash_attn_sm70_prefill_paged, which reads KV directly
from the paged cache (with vectorized loads + native GQA) and is ~2x faster
than Triton. This gather is retained as a reference/utility.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kv_kernel(
    kv_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    output_ptr,
    stride_cache_b,
    stride_cache_h,
    stride_cache_s,
    stride_out_seq,
    stride_out_h,
    stride_out_s,
    block_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    d_range = tl.arange(0, HEAD_DIM)

    for block_in_seq in range(MAX_BLOCKS):
        pos = block_in_seq * block_size
        if pos < seq_len:
            block_id = tl.load(
                block_table_ptr + seq_idx * MAX_BLOCKS + block_in_seq
            )
            actual_len = tl.minimum(block_size, seq_len - pos)

            for s in range(block_size):
                if s < actual_len:
                    src_offset = (
                        block_id * stride_cache_b
                        + head_idx * stride_cache_h
                        + s * stride_cache_s
                    )
                    dst_offset = (
                        seq_idx * stride_out_seq
                        + head_idx * stride_out_h
                        + (pos + s) * stride_out_s
                    )
                    vals = tl.load(kv_cache_ptr + src_offset + d_range)
                    tl.store(output_ptr + dst_offset + d_range, vals)


def gather_paged_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Gather paged KV into contiguous [num_seqs, num_kv_heads, max_seq_len, head_dim]."""
    num_seqs = seq_lens.shape[0]
    num_kv_heads = kv_cache.shape[1]
    block_size = kv_cache.shape[2]
    head_dim = kv_cache.shape[3]
    max_blocks = block_table.shape[1]

    output = torch.zeros(
        num_seqs, num_kv_heads, max_seq_len, head_dim,
        dtype=kv_cache.dtype, device=kv_cache.device,
    )

    grid = (num_seqs, num_kv_heads)
    _gather_kv_kernel[grid](
        kv_cache,
        block_table,
        seq_lens,
        output,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_size,
        HEAD_DIM=head_dim,
        MAX_BLOCKS=max_blocks,
    )
    return output
