# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Admission predicate for the SM70 splitd D256 paged prefill kernel.

The splitd kernel (ported from 1Cat's flash-attention-v100 fork, op
``torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_paged_fwd``) is an exact-tile
causal prefill kernel: q_len must be a multiple of its 64-row tile and the
live kv_len a multiple of its 32-column tile, with head_dim 256 and FP16
inputs. Steps that do not meet the shape contract run on the FA2 paged
path (byte-identical to the pre-port behavior).
"""

import torch


def sm70_splitd_admit(
    num_prefills: int,
    q_len: int,
    kv_len: int,
    causal: bool,
    q_dtype,
) -> bool:
    """Whether this step may route its prefill to the splitd paged kernel.

    Phase 1 admits a single prefill request per step (the paged op consumes
    one block-table row); multi-request prefill steps fall back to FA2.

    Args:
        num_prefills: prefill requests in this step.
        q_len: this request's prefill query length for the step.
        kv_len: this request's live KV length (cached + new tokens).
        causal: the layer uses causal attention.
        q_dtype: the query tensor dtype.

    Returns:
        True iff every splitd shape contract holds.
    """
    return (
        num_prefills == 1
        and causal
        and q_dtype is torch.float16
        and 64 <= q_len <= 8192
        and q_len % 64 == 0
        and 0 < kv_len <= 262144
        and kv_len % 32 == 0
        and q_len <= kv_len
    )
