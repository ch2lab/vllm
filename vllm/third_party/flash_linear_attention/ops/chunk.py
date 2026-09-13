# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril
from .utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL, input_guard
from .wy_fast import recompute_w_u_fwd

# Max per-sequence query length of the next chunk_gated_delta_rule_fwd call,
# published by the GDN layer from the attention metadata. Worker execution is
# sequential, so a module global is safe and avoids threading an extra
# argument through the autograd Function.
_SM70_MAX_QUERY_LEN: int | None = None
_FLA_DBG_COUNT = 0


def sm70_set_max_query_len(v: int | None) -> None:
    global _SM70_MAX_QUERY_LEN
    _SM70_MAX_QUERY_LEN = v


def _fused_sm70_verify(q, k, v, beta, g, initial_state, cu_seqlens, scale):
    """Fused single-chunk gated-delta-rule (MTP-verify path, T<=8).

    Replaces cumsum/kkt/solve_tril/wy/delta_h/chunk_o with one CUDA WMMA
    kernel launch per layer. Returns (o, final_state).
    """
    B, T, Hg, K = q.shape
    H = v.shape[2]
    V = v.shape[3]
    # The WMMA kernel reads fp16 only; the gating projections can hand over
    # fp32/bf16 tensors (e.g. during cudagraph dummy runs).
    if q.dtype != torch.float16:
        q = q.half()
    if k.dtype != torch.float16:
        k = k.half()
    if v.dtype != torch.float16:
        v = v.half()
    beta2 = beta.reshape(-1, H)
    g2 = g.reshape(-1, H)
    if beta2.dtype != torch.float16:
        beta2 = beta2.half()
    if g2.dtype != torch.float16:
        g2 = g2.half()
    h0 = initial_state if initial_state is not None else torch.empty(
        0, dtype=torch.float32, device=q.device)
    # The kernel reads the state as fp32; the ssm cache is bf16.
    if h0.numel() > 0 and h0.dtype != torch.float32:
        h0 = h0.float()
    cu32 = cu_seqlens.to(torch.int32)
    N = cu_seqlens.numel() - 1
    # The stable-libtorch binding mangles float scalar args; pass the fp32
    # bit pattern as an int64 instead.
    import struct as _s
    scale_bits = _s.unpack('q', _s.pack('<f', scale) + b'\0' * 4)[0]
    o, h_new = torch.ops._C.fla_verify_sm70(
        q.contiguous(), k.contiguous(), v.contiguous(),
        beta2, g2, h0, cu32, scale_bits, N)
    return o.view(B, T, H, V), h_new


def chunk_gated_delta_rule_fwd(
    q, k, v, g, beta, scale, initial_state, output_final_state,
    cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
    core_attn_out=None,
):
    # SM70 fused single-chunk path for the small-T (MTP verify) case.
    from vllm.platforms import current_platform
    import os as _os

    # The per-sequence query length comes from the attention metadata when
    # the caller set it (decode/verify batches pad many small sequences, so
    # total-T checks misclassify them); fall back to the conservative
    # total<=8 check otherwise.
    mq = _SM70_MAX_QUERY_LEN
    small_t = (1 <= mq <= 8) if mq is not None else (1 <= k.shape[1] <= 8)
    if (_os.environ.get("VLLM_SM70_FLA_FUSED_VERIFY", "1") != "0"
            and current_platform.is_cuda()
            and current_platform.has_device_capability(70)
            and not current_platform.has_device_capability(80)
            and cu_seqlens is not None
            and q.shape[-1] == 128 and v.shape[-1] == 128
            and v.shape[2] % q.shape[2] == 0
            and small_t):
        o, fs = _fused_sm70_verify(q, k, v, beta, g, initial_state,
                                   cu_seqlens, scale)
        return None, o, None, fs, None, None, None
    g = chunk_local_cumsum(
        g, chunk_size=FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        core_attn_out=core_attn_out,
    )
    return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        core_attn_out: torch.Tensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        if core_attn_out is not None:
            assert not torch.is_grad_enabled(), (
                "core_attn_out buffer reuse is only supported for inference"
            )
            assert q.dtype == o.dtype, "Incompatible dtype for inplace computation"
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    core_attn_out: torch.Tensor | None = None,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) Gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, V, K, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        use_qk_l2norm_in_kernel,
        core_attn_out,
    )
    return o, final_state
