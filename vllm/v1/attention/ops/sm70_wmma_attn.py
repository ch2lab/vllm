# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DEPRECATED: This module is superseded by:
- vllm.v1.attention.ops.sm70_gather_kv (Triton paged KV gather)
- torch.ops._C.flash_attn_sm70_prefill_paged (direct paged WMMA kernel)

The SM70 attention backend is at:
- vllm.v1.attention.backends.sm70_wmma_attn
"""
