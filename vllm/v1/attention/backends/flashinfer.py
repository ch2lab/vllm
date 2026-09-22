# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""

import os
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from typing import ClassVar

import numpy as np
import torch
from flashinfer import (
    BatchAttentionWithAttentionSinkWrapper,
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
    get_seq_lens,
)
from flashinfer.decode import fast_decode_plan, trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import FP4Tensor
from typing_extensions import override

from vllm import _custom_ops as custom_ops
from vllm import envs
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    flashinfer_xqa_batch_decode_with_kv_cache,
    force_use_trtllm_attention,
    pin_host_range_buf,
    supports_trtllm_attention,
    use_trtllm_attention,
)
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    PIN_MEMORY,
    canonicalize_singleton_dim_strides,
    get_dtype_size,
    is_quantized_kv_cache,
    is_strictly_contiguous,
    nvfp4_kv_cache_full_dim,
    nvfp4_split_data_scale,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    get_flashinfer_layout_string,
    get_num_attention_heads_from_layers,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    log2_lse_to_ln,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.dcp import (
    cp_lse_ag_out_rs,
    dcp_a2a_lse_reduce,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.sm70_nvfp4_decode import sm70_nvfp4_paged_decode
from vllm.v1.attention.ops.sm70_flashdecode import sm70_flashdecode_op
from vllm.v1.attention.ops.sm70_nvfp4_dequant import (
    sm70_fp8_paged_dequant,
    sm70_nvfp4_paged_dequant,
    sm70_nvfp4_paged_stage_nhd,
    sm70_shadow_plan_tensors,
)
from vllm.v1.attention.ops.sm70_nvfp4_prefill import sm70_nvfp4_paged_prefill
from vllm.v1.attention.ops.sm70_splitd_policy import sm70_splitd_admit
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    KVCacheSpec,
    KVQuantMode,
    iter_layer_specs,
)
from vllm.v1.utils import CpuGpuBuffer

# SM70 NVFP4: opt-in native in-kernel-dequant paged prefill (retires the
# transient fp16 shadow prefill). Read once per process; builder and impl
# in the same worker process agree on the value.
_SM70_NATIVE_PREFILL = os.environ.get("VLLM_SM70_NATIVE_PREFILL", "0") == "1"

# FlashDecoding split-count multiplier (SM70 NVFP4 unified decode kernel).
# The split-KV block is small (128 thr, ~12KB smem) so each SM hosts ~8 of
# them, but the default grid (SM//RH) launches only 1/SM, leaving the SMs
# under-utilized and the serial per-token GQA warp-reductions under-
# parallelized. Raising the target SM count launches more concurrent KV-
# reader blocks (higher MLP + more reduction parallelism); the reduce's
# O(splits) cost grows with it, so the sweet spot is ~sqrt(part/reduce) ~= 2.
# Module-level so FlashInferMetadataBuilder and FlashInferImpl agree.
# (Cycle 2: 1->2 fixed the short-ctx regression once the reduce was
# parallelized; 2->4 fills the SMs (2/SM -> 4/SM) for more long-ctx MLP.
# K=4 is the measured optimum on BOTH kernels:
#   old kernel: K=8 regressed (8 blocks/SM > smem cap 12.2KB x 7 < 96KB
#     -> 1.6 waves, plus O(splits) reduce).
#   new kernel (cycle 3, per-chunk ws RMW removed): K=6 regressed
#     (20K 29.4 vs 28.3, 40K 31.8 vs 29.9) -> partition no longer
#     block-count-bound at 4/SM. See nospec_slope_fd_k{4,6,8}*.log.
#   row-shared kernel (cycle 4, rpb passes + PRMT): REVERTED 6->4 -- the
#     row-shared build passed the full offline matrix (131K q4, n_page
#     2848, perm BT) but produced degenerate output ONLY in the live TP2
#     service (see memory 10349); bisect vs the 11:42-working tree pending.
#   v4 HMMA kernel (2026-09-22): offline fixed-splits sweep said 3/SM beat
#     4/SM by 10-12%, but the live MTP3 A/B (mixed q1-decode + q4-verify
#     steps) REGRESSED 44.7->46.1 / 57.1->64.1 / 83.3->95.8 ms/step. SF=4
#     stands for both kernels; production benches, not shape sweeps, rule.
_SM70_FD_SPLIT_FACTOR = 4
# Verify-path split factor (grid is R=1 capture-invariant, see builder):
# 4 for the per-row kernel (splits=80 optimum), 6 for the merged v4
# kernel (splits=120, -15% at all ctx).
_SM70_FD_VERIFY_FACTOR = 6 if envs.VLLM_SM70_FD_HMMA else 4

# SM70 NVFP4: opt-in in-kernel-dequant SPLITD paged prefill. The WMMA
# tensor-core kernel reads the packed NVFP4 cache directly (1/8 the KV HBM
# traffic of the staged-fp16 path, no staging pass). Read once per process;
# builder and impl in the same worker process agree on the value.
_SM70_SPLITD_NVFP4_INLINE = (
    os.environ.get("VLLM_SM70_SPLITD_NVFP4_INLINE", "0") == "1"
)

FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024
FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM = 16

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def sm70_nvfp4_kv_enabled() -> bool:
    """Whether the opt-in SM70 (family 70) NVFP4 KV-cache store path is active.

    Gated on VLLM_SM70_NVFP4_KV so that, unless explicitly enabled, this
    backend is byte-identical to upstream (NVFP4 KV only on SM100 trtllm-gen).
    """
    return envs.VLLM_SM70_NVFP4_KV and current_platform.is_device_capability_family(
        70
    )


def sm70_flashinfer_attn_enabled() -> bool:
    """Whether the opt-in SM70 (family 70) native FlashInfer paged attention is
    active.

    Gated on VLLM_SM70_FLASHINFER_ATTN so that, unless explicitly enabled, backend
    selection is byte-identical to upstream (the FlashInfer floor stays SM80+).
    On SM70 the 16-bit paged decode+prefill path (f16 prefill, f16/bf16 decode) is
    used; QUASAR-style bf16 models run as fp16 (see the V100 attention notes).
    """
    return envs.VLLM_SM70_FLASHINFER_ATTN and current_platform.is_device_capability_family(
        70
    )


def _nvfp4_kv_backend(is_nvfp4: bool) -> str:
    """FlashInfer backend for the NVFP4 KV-cache attention path.

    SM100/SM103 reads NVFP4 KV via trtllm-gen; SM70 (opt-in,
    VLLM_SM70_NVFP4_KV) and SM12x read it via the native/FA2 paged path, so
    use "auto".  Non-NVFP4 KV always uses "auto".
    """
    if is_nvfp4 and current_platform.is_device_capability_family(100):
        return "trtllm-gen"
    return "auto"


def _nvfp4_o_dtype(is_nvfp4: bool, model_dtype: torch.dtype) -> torch.dtype:
    """Output dtype for the FlashInfer attention path.

    SM100 trtllm-gen NVFP4 outputs FP8; SM70 (opt-in) and SM12x NVFP4
    output the model dtype.  Non-NVFP4 always uses the model dtype.
    """
    if is_nvfp4 and current_platform.is_device_capability_family(100):
        return FP8_DTYPE
    return model_dtype

logger = init_logger(__name__)

trtllm_workspace_buffer = None


def _get_trtllm_workspace_buffer():
    global trtllm_workspace_buffer
    if trtllm_workspace_buffer is None:
        trtllm_workspace_buffer = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda"
        )
    return trtllm_workspace_buffer


def _pack_draft_block_bool_mask(
    bool_mask: torch.Tensor, num_packed: int
) -> torch.Tensor:
    """Pack a boolean draft mask into XQA's uint16 layout."""
    num_rows = bool_mask.shape[0]
    bits = 1 << torch.arange(32, device=bool_mask.device, dtype=torch.int64)
    mask_u32 = (
        (bool_mask.view(num_rows, num_packed, 32).to(torch.int64) * bits)
        .sum(dim=-1)
        .to(torch.uint32)
    )
    return mask_u32.view(torch.uint16).reshape(num_rows, num_packed * 2)


def _make_xqa_draft_block_mask(
    q_len: int, causal: bool, device: torch.device
) -> torch.Tensor:
    """Build a packed XQA draft mask for a uniform request."""
    num_packed = (q_len + 31) // 32
    padded = num_packed * 32
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(padded, device=device).unsqueeze(0)
    bool_mask = kv_idx <= q_idx if causal else (kv_idx < q_len).expand(q_len, padded)
    return _pack_draft_block_bool_mask(bool_mask, num_packed)


def _make_xqa_ragged_draft_block_mask(
    q_lens: list[int], max_q_len: int, causal: bool, device: torch.device
) -> torch.Tensor:
    """Build a packed XQA draft mask for a ragged batch."""
    num_packed = (max_q_len + 31) // 32
    padded = num_packed * 32
    q_lens_t = torch.tensor(q_lens, device=device)
    row_lens = torch.repeat_interleave(q_lens_t, q_lens_t)
    request_starts = torch.cumsum(q_lens_t, dim=0) - q_lens_t
    row_starts = torch.repeat_interleave(request_starts, q_lens_t)
    q_idx = torch.arange(sum(q_lens), device=device) - row_starts
    kv_idx = torch.arange(padded, device=device).unsqueeze(0)
    bool_mask = kv_idx < row_lens.unsqueeze(1)
    if causal:
        bool_mask &= kv_idx <= q_idx.unsqueeze(1)
    return _pack_draft_block_bool_mask(bool_mask, num_packed)


@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    src_stride_page,
    src_stride_kv,
    src_stride_head,
    src_stride_block,
    src_stride_head_size,
    DST_K_CACHE_STRIDE: tl.constexpr,
    DST_KV_CACHE_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    ).to(tl.int64)
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    mock_page_idx = batch_idx * block_table_stride + mock_block_table_idx + 1
    logical_offsets = tl.arange(0, HEAD_STRIDE)
    block_offsets = logical_offsets // HEAD_SIZE
    head_size_offsets = logical_offsets % HEAD_SIZE

    for h in range(NUM_KV_HEADS):
        h_off = tl.cast(h, tl.int64)

        # Read K from source (supports non-contiguous page/kv/head strides)
        src_k = (
            orig_page_num * src_stride_page
            + h_off * src_stride_head
            + block_offsets * src_stride_block
            + head_size_offsets * src_stride_head_size
        )
        fp8_k = tl.load(kv_cache_ptr + src_k)
        dequant_k = (fp8_k.to(tl.float32) * k_scale_val).to(dequant_dtype)

        # Write K to contiguous mock cache
        dst_k = mock_page_idx * DST_KV_CACHE_STRIDE + h * HEAD_STRIDE + logical_offsets
        tl.store(mock_kv_cache_ptr + dst_k, dequant_k)

        # Read V from source (offset by src_stride_kv for the V half)
        src_v = (
            orig_page_num * src_stride_page
            + src_stride_kv
            + h_off * src_stride_head
            + block_offsets * src_stride_block
            + head_size_offsets * src_stride_head_size
        )
        fp8_v = tl.load(kv_cache_ptr + src_v)
        dequant_v = (fp8_v.to(tl.float32) * v_scale_val).to(dequant_dtype)

        # Write V to contiguous mock cache
        dst_v = (
            mock_page_idx * DST_KV_CACHE_STRIDE
            + DST_K_CACHE_STRIDE
            + h * HEAD_STRIDE
            + logical_offsets
        )
        tl.store(mock_kv_cache_ptr + dst_v, dequant_v)


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)

    # Logical source layout is (B, 2, H, N, D). The tensor may be a
    # non-contiguous view, so the Triton kernel indexes it with actual strides.
    strides = kv_cache.stride()
    num_kv_heads, block_size, head_size = s[2], s[3], s[4]
    head_stride = block_size * head_size
    k_cache_stride = num_kv_heads * head_stride
    kv_cache_stride = k_cache_stride * s[1]

    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    # mock kv cache contains just the pages needed by this prefill
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device=kv_cache.device)
    # we simply sequentially index the pages needed by this prefill
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device=block_tables_prefill.device,
    ).reshape(batch_size, num_of_page_per_token)
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        k_cache_stride,
        kv_cache_stride,
        head_stride,
        head_size,
        num_kv_heads,
    )
    return mock_kv_cache, mock_block_table


class BatchDCPPrefillWrapper:
    def __init__(
        self,
        kv_layout: str,
        workspace_buffer: torch.Tensor | None = None,
        dcp_a2a: bool = False,
    ):
        if dcp_a2a:
            self._dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self._dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)
        self._context = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
        self._new_tokens = BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer)

    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        kv_lens_cpu: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        dcp_world_size: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        q_data_type: torch.dtype,
        kv_cache_dtype: torch.dtype,
        prefill_fixed_split_size: int,
        disable_split_kv: bool,
    ):
        """Plan the prefill operation with given parameters."""
        self._context.plan(
            qo_indptr=qo_indptr_cpu,
            paged_kv_indptr=paged_kv_indptr_cpu,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len_cpu,
            seq_lens=kv_lens_cpu,
            num_qo_heads=num_qo_heads * dcp_world_size,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=False,  # This is context run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_cache_dtype,
            fixed_split_size=prefill_fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self._new_tokens.plan(
            qo_indptr=qo_indptr_cpu,
            kv_indptr=qo_indptr_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            causal=True,  # This is newtokens run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
        )

    def run(
        self,
        layer: torch.nn.Module,
        prefill_query: torch.Tensor,
        kv_cache_tuple: tuple[torch.Tensor, torch.Tensor],
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
    ):
        prefill_query_across_dcp = get_dcp_group().all_gather(
            prefill_query.contiguous(), dim=1
        )
        output_context_tmp, lse_context_tmp = self._context.run(
            prefill_query_across_dcp,
            kv_cache_tuple,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            return_lse=True,
        )
        output_context, lse_context = self._dcp_combine(
            output_context_tmp,
            lse_context_tmp,
            get_dcp_group(),
            return_lse=True,
        )
        lse_context = log2_lse_to_ln(lse_context.transpose(0, 1).contiguous())

        output_query, lse_query = self._new_tokens.run(
            prefill_query,
            key,
            value,
            return_lse=True,
        )
        lse_query = log2_lse_to_ln(lse_query.transpose(0, 1).contiguous())

        merge_attn_states(
            out,
            output_context,
            lse_context,
            output_query,
            lse_query,
        )
        return out


class FlashInferBackend(AttentionBackend):
    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        """NVFP4 stores K and V as separate per-head slots of packed fp4 data
        plus fp8 block scales."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
            return spec
        hs_k = nvfp4_kv_cache_full_dim(spec.head_size)
        hs_v = nvfp4_kv_cache_full_dim(spec.head_size_v)
        assert hs_k == hs_v, "nvfp4 with asymmetric K/V head sizes not yet supported"
        return replace(
            spec,
            num_head_slots=2 * spec.num_kv_heads,
            state_content_bytes=hs_k * get_dtype_size(spec.dtype),
        )

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
        "nvfp4_4over6",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Page sizes >= 128 only run on the trtllm-gen dynamic kernel (GQA/MQA
        # on Blackwell); advertise them only when usable so selection never
        # picks a large kernel block we cannot serve.
        use_large_pages = False
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            pc = vllm_config.parallel_config
            mc = vllm_config.model_config
            num_qo_heads = mc.get_num_attention_heads(pc)
            num_kv_heads = mc.get_num_kv_heads(pc)
            use_large_pages = (
                num_kv_heads > 0
                and num_qo_heads // num_kv_heads > 1
                and current_platform.is_device_capability_family(100)
                and can_use_trtllm_attention(num_qo_heads, num_kv_heads)
            )
        if not use_large_pages:
            return [16, 32, 64]
        return [16, 32, 64, 128, 256, 512, 1024]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Only trtllm-gen decode (SM100 family) reads query offsets from the
        # device tensor; XQA and wrapper decode plan from qo_indptr_cpu, and
        # under DCP spec queries are routed to the prefill path.
        if not current_platform.is_device_capability_family(100):
            return False
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None or vllm_config.model_config is None:
            return True
        pc = vllm_config.parallel_config
        mc = vllm_config.model_config
        return pc.decode_context_parallel_size == 1 and can_use_trtllm_attention(
            mc.get_num_attention_heads(pc), mc.get_num_kv_heads(pc), is_prefill=False
        )

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMetadataBuilder"]:
        return FlashInferMetadataBuilder

    @staticmethod
    def get_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        elif kv_cache_dtype.startswith("nvfp4"):
            return torch.uint8
        else:
            raise ValueError(f"Unrecognized dtype: {kv_cache_dtype}")

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is not None and kv_cache_dtype.startswith("nvfp4"):
            if sm70_nvfp4_kv_enabled():
                # SM70 (family 70) NVFP4 KV-cache: store path + native paged
                # attention read (see VLLM_SM70_NVFP4_KV).
                return True
            # SM100/SM103: NVFP4 KV cache is serviced by the trtllm-gen
            # prefill/decode kernels (both phases required).
            if current_platform.is_device_capability_family(100):
                return supports_trtllm_attention(
                    is_prefill=True
                ) and supports_trtllm_attention(is_prefill=False)
            # SM12x: NVFP4 KV cache is serviced by the FlashInfer XQA decode
            # and native (FA2) prefill paths, which are JIT-compiled and need
            # no TRTLLM cubins. The decode kernel (XQA vs native) is resolved
            # at runtime the same way as FP8 KV cache.
            return current_platform.is_device_capability_family(120)
        return super().supports_kv_cache_dtype(kv_cache_dtype)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        return [64, 128, 256, 512]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # SM70 (Volta V100): the native 16-bit paged attention (f16 prefill,
        # f16/bf16 decode) is available and env-gated. Allow selection when
        # explicitly enabled; otherwise keep the SM80 floor (upstream behavior).
        if capability == DeviceCapability(7, 0) and sm70_flashinfer_attn_enabled():
            return True
        # FlashInfer supports SM75+, but is currently broken on SM75 (Turing):
        # https://github.com/flashinfer-ai/flashinfer/issues/3620 (fix:
        # https://github.com/flashinfer-ai/flashinfer/pull/3621). Temporarily
        # raise the floor to SM80 so it is not auto-selected on SM75 until
        # that fix lands; revert to DeviceCapability(7, 5) once it does.
        return capability >= DeviceCapability(8, 0) and capability <= DeviceCapability(
            12, 1
        )

    @classmethod
    def supports_sink(cls) -> bool:
        """Whether FlashInfer can serve attention sinks on this platform."""
        from vllm.utils.flashinfer import (
            force_use_trtllm_attention,
        )

        if force_use_trtllm_attention() is False:
            return False

        if current_platform.is_device_capability(
            90
        ) or current_platform.is_device_capability_family(120):
            return supports_trtllm_attention(is_prefill=False)

        if not current_platform.is_device_capability_family(100):
            return False

        return supports_trtllm_attention(
            is_prefill=False
        ) and supports_trtllm_attention(is_prefill=True)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...] | None:
        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 10:
            # The trtllm-gen kernels consume head-major block interiors; the L/B
            # nesting outside the block is immaterial to them.
            return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)
        if capability is not None and capability.major == 12:
            # NVFP4 KV on consumer Blackwell (sm120/sm121, FA2 path): each K/V
            # side of the cache packs [data | scale] regions carved out of the
            # side's byte range (reshape_and_cache_nvfp4 writes the scales at
            # side_base + num_heads * block_size * data_dim;
            # nvfp4_split_data_scale reads them back with derived strides).
            # That carve is only byte-coherent when each side's heads own one
            # contiguous region per page, i.e. under the head-major block
            # interior (BLHNC/LBHNC). Under NHD-style layouts the K and V head
            # rows interleave within each token and the side-region offsets
            # land inside the other side's data -> silent KV corruption.
            vllm_config = get_current_vllm_config_or_none()
            if (
                vllm_config is not None
                and vllm_config.cache_config is not None
                and vllm_config.cache_config.cache_dtype.startswith("nvfp4")
            ):
                return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)
        return super().supported_kv_cache_layouts()

    forward_includes_kv_cache_update: bool = False


_SM70_SPLITD_OP = None
_SM70_SPLITD_NVFP4_OP = None
_SM70_FLASHDECODE_OP = None
"""Cached splitd paged prefill ops (False = extension not built)."""

_SPLITD_ROUTE_STATS = {"routed": 0, "fallback": 0}


def _get_sm70_splitd_op():
    """Lazily look up the splitd paged prefill op (False if not built)."""
    global _SM70_SPLITD_OP
    if _SM70_SPLITD_OP is None:
        try:
            # The op registers when the C library is loaded into the
            # process; the Python package import is what loads it.
            import vllm.vllm_flash_attn  # noqa: F401

            _SM70_SPLITD_OP = (
                torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_paged_fwd
            )
        except Exception:
            _SM70_SPLITD_OP = False
    return _SM70_SPLITD_OP


def _get_sm70_splitd_nvfp4_op():
    """Lazily look up the NVFP4 in-kernel-dequant paged op (False if not
    built). The op reads the packed paged KV cache directly, so an
    admitted step skips the fp16 staging pass entirely."""
    global _SM70_SPLITD_NVFP4_OP
    if _SM70_SPLITD_NVFP4_OP is None:
        try:
            import vllm.vllm_flash_attn  # noqa: F401

            _SM70_SPLITD_NVFP4_OP = (
                torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_nvfp4_paged_fwd
            )
        except Exception:
            _SM70_SPLITD_NVFP4_OP = False
    return _SM70_SPLITD_NVFP4_OP


def _get_sm70_flashdecode_op():
    """Lazily look up the unified flashdecode op (False if not built)."""
    global _SM70_FLASHDECODE_OP
    if _SM70_FLASHDECODE_OP is None:
        try:
            # Built only for sm_70 (build_sm70.sh); absent on other archs.
            import vllm  # noqa: F401

            _SM70_FLASHDECODE_OP = torch.ops._C.sm70_flashdecode
        except Exception:
            _SM70_FLASHDECODE_OP = False
    return _SM70_FLASHDECODE_OP


@dataclass
class FIPrefill:
    """Metadata for the native FlashInfer prefill pathway (non-TRTLLM)."""

    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper
    block_tables: torch.Tensor
    """Prefill-only block table slice. [num_prefills, max_num_blocks_per_seq]."""
    seq_lens: torch.Tensor
    """Prefill-only KV sequence lengths. [num_prefills]."""
    cum_seq_lens_q: torch.Tensor
    """Shifted qo_indptr over prefill requests (GPU). [num_prefills + 1]."""
    shadow_kv: torch.Tensor | None = None
    """SM70 NVFP4: fp16 shadow paged KV ``[B_s, H, N, 2*head_dim]``."""
    shadow_kv_lens: torch.Tensor | None = None
    """SM70 NVFP4: host int32 prefill KV lengths for the shadow dequant."""
    shadow_kv_windowed: bool = False
    """SM70 NVFP4: sliding-window shadow buffer holds only the last
    window_left tokens; the dequant slices the block table to that tail."""
    shadow_bt_start: int = 0
    """SM70 NVFP4 windowed: first used block-table column of the window
    tail (block tables are left-aligned, right-padded)."""
    native_qo_indptr: torch.Tensor | None = None
    """SM70 NVFP4 native prefill: device int32 ``[num_prefills + 1]`` q
    offsets (into the prefill query/output rows)."""
    native_kv_lens: torch.Tensor | None = None
    """SM70 NVFP4 native prefill: device int32 ``[num_prefills]`` KV lengths."""
    native_max_q_len: int = 0
    """SM70 NVFP4 native prefill: host max per-request query length."""
    splitd_admit: bool = False
    """SM70 splitd: this step's single prefill request is admitted."""
    splitd_q_len: int = 0
    """SM70 splitd: admitted request's prefill query length (host)."""
    splitd_kv_len: int = 0
    """SM70 splitd: admitted request's live KV length (host)."""
    splitd_bt_row: torch.Tensor | None = None
    """SM70 splitd: the request's nvfp4 block ids (int32 [M], device)."""
    splitd_bt: torch.Tensor | None = None
    """SM70 splitd: identity int32 block table [1, M] for the splitd op."""
    splitd_k_ws: torch.Tensor | None = None
    """SM70 splitd: fp16 NHD K stage [M, N, H_kv, D] (per layer, reused)."""
    splitd_v_ws: torch.Tensor | None = None
    """SM70 splitd: fp16 NHD V stage [M, N, H_kv, D] (per layer, reused)."""
    splitd_q_ws: torch.Tensor | None = None
    """SM70 splitd: padded fp16 query buffer [1, ceil64(q_len), H_q, D]."""
    splitd_out_ws: torch.Tensor | None = None
    """SM70 splitd: padded fp16 op output [1, ceil64(q_len), H_q, D]."""
    fd_verify: bool = False
    """SM70 flashdecode verify: this step's single prefill request has
    q_len <= 8 (spec verify) and is routed to the in-kernel-dequant
    flashdecode op in forward — 1x KV reads, no fp16 shadow staging."""
    fd_kv_lens_dev: torch.Tensor | None = None
    """SM70 flashdecode verify: static int32 [1] device kv-len buffer
    (pointer stable across cudagraph replays)."""


@dataclass
class FIDecode:
    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""

    wrapper: BatchDecodeWithPagedKVCacheWrapper
    shadow_kv: torch.Tensor | None = None
    """SM70 NVFP4: fp16 shadow paged KV ``[B_s, H, N, 2*head_dim]``."""
    shadow_kv_lens: torch.Tensor | None = None
    """SM70 NVFP4: host int32 decode KV lengths for the shadow dequant."""
    block_tables: torch.Tensor | None = None
    """SM70 NVFP4: device block table slice for the shadow dequant."""
    shadow_kv_lens_dev: torch.Tensor | None = None
    """SM70 NVFP4 graph decode: device int32 ``[num_reqs]`` KV lengths."""
    shadow_num_reqs: int = 0
    """SM70 NVFP4: decode batch size (requests, fixed per captured graph)."""
    seq_lens_source: torch.Tensor | None = None
    """SM70 NVFP4: device seq_lens tensor this step's kv-lens were built
    from; update_draft_decode_metadata re-reads it between draft steps."""
    shadow_splits: int = 0
    """SM70 NVFP4: split-KV count for the decode (fixed per captured graph)."""
    shadow_ws: torch.Tensor | None = None
    """SM70 NVFP4: shared split-KV partial-state workspace (f32)."""
    shadow_max_blocks: int = 0
    """SM70 NVFP4 graph decode: uniform per-request shadow block allotment."""
    shadow_graph: bool = False
    """SM70 NVFP4: shadow dequant runs the fixed-grid graph kernel."""


class FlashInferDecodeKernel(Enum):
    """Decode kernels selected inside the FlashInfer backend."""

    XQA = "xqa"
    TRTLLM_GEN = "trtllm-gen"


def _is_xqa_head_dim_supported(head_dim: int) -> bool:
    return 16 <= head_dim <= 256 and head_dim % 16 == 0


def _nvfp4_d256_prefill(
    impl,
    prefill_query: torch.Tensor,
    kv_cache: torch.Tensor,
    pre: FIPrefill,
    q_scales: tuple[float, float, float],
    out_prefill: torch.Tensor,
    causal: bool = True,
) -> None:
    """Run the QK MXFP8 / PV NVFP4 kernel for NVFP4-KV D256 prefill requests.

    One kernel call per prefill request: gather that request's paged NVFP4
    blocks into dense per-side views, convert to the kernel's MXFP8/NVFP4
    input formats, and dispatch the dense kernel. ``kv_cache`` is the raw
    (num_blocks, 2*N_kv, block_size, F) layer cache; K heads occupy the first
    N_kv head slots, V the next N_kv (page = [K_data|K_scale|V_data|V_scale]).
    Within each side the page bytes are one head-major data region
    (H_kv·N·(F·8//9)) followed by one head-major scale region
    (H_kv·N·(F- F·8//9)), as ``nvfp4_split_data_scale`` reads back for the
    FA2 path; the row-interleaved [data | scale] rows are only the kernel's
    input form, re-assembled here.
    """
    from flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120 import (
        nvfp4_prefill_convert,
        qk_mxfp8_pv_nvfp4_attention_sm120_fwd,
    )

    q_scale, k_scale, v_scale = q_scales
    Hk = impl.num_kv_heads
    F = nvfp4_kv_cache_full_dim(impl.head_size)
    F_data = F * 8 // 9
    F_sf = F - F_data
    page_size = kv_cache.shape[2]
    qo = pre.cum_seq_lens_q
    for i in range(pre.block_tables.shape[0]):
        q0 = int(qo[i].item())
        q1 = int(qo[i + 1].item())
        kv_len = int(pre.seq_lens[i].item())
        nblocks = (kv_len + page_size - 1) // page_size
        gathered = kv_cache.index_select(0, pre.block_tables[i, :nblocks])
        # Split each side of the gathered pages into its physical data and
        # scale regions (head-major, token-minor views of the 4-D side).
        k_data, k_sf = nvfp4_split_data_scale(gathered[:, :Hk])
        v_data, v_sf = nvfp4_split_data_scale(gathered[:, Hk:])
        # Repack into the kernel's row-interleaved (Hk, S, F) rows
        # [data(h,s) | sf(h,s)], token order block-major, sliced to kv_len.
        k_side = torch.cat([
            k_data.permute(1, 0, 2, 3).contiguous().reshape(Hk, -1, F_data),
            k_sf.permute(1, 0, 2, 3).contiguous().reshape(Hk, -1, F_sf)
            .view(torch.uint8),
        ], dim=-1)[:, :kv_len].contiguous()
        v_side = torch.cat([
            v_data.permute(1, 0, 2, 3).contiguous().reshape(Hk, -1, F_data),
            v_sf.permute(1, 0, 2, 3).contiguous().reshape(Hk, -1, F_sf)
            .view(torch.uint8),
        ], dim=-1)[:, :kv_len].contiguous()
        q_bf16 = prefill_query[q0:q1].unsqueeze(0).transpose(1, 2)
        q_bf16 = q_bf16.contiguous()  # (1, Hq, M, D), convert needs C-order
        (q_fp8, k_fp8, v_fp4_t,
         q_sf, k_sf, v_sf_t) = nvfp4_prefill_convert(
            k_side, v_side, q_bf16, k_scale, v_scale, q_scale)
        q_len = q1 - q0
        out_req = qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            q_fp8, k_fp8, v_fp4_t, q_sf, k_sf, v_sf_t,
            sm_scale=impl.scale, causal=causal,
            unpadded_q_len=q_len, unpadded_k_len=kv_len,
            out_dtype=torch.bfloat16,
        )
        # fwd returns the 128-padded q extent; slice back to this request.
        out_prefill[q0:q1] = out_req[0, :, :q_len].transpose(0, 1)


@dataclass
class TRTLLMPrefill:
    """Metadata for the TRTLLM prefill pathway."""

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to prefill requests.
    Shape: [num_prefills, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to prefill requests.
    Shape: [num_prefills]
    """

    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor

    max_q_len: int
    """
    The maximum query length *among prefill requests*.
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


@dataclass
class FlashInferTrtllmAPIDecode:
    """Metadata for XQA and trtllm-gen decode."""

    kernel: FlashInferDecodeKernel

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to decode requests.
    Shape: [num_decodes, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to decode requests.
    Shape: [num_decodes]
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""

    q_len_per_req: int = 1
    """Query tokens per request for uniform speculative decode."""

    q_cu_seq_lens: torch.Tensor | None = None
    """Cumulative query lengths for ragged XQA decode."""

    mask: torch.Tensor | None = None
    """Packed XQA draft mask."""

    dcp_query_start_loc: torch.Tensor | None = None
    """Decode query boundaries for masking empty DCP shards during reduction."""


@dataclass
class FlashInferMetadata:
    num_actual_tokens: int
    """Total number of tokens in the batch (excluding padding)."""

    slot_mapping: torch.Tensor
    """Tensor for writing K/V to the cache. Shape: [num_actual_tokens]"""

    # The data types of the query for prefill and decode.
    # On SM90, these two data types may be different.
    q_data_type_prefill: torch.dtype
    q_data_type_decode: torch.dtype

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    causal: bool

    prefill: FIPrefill | TRTLLMPrefill | None
    """
    Holds the metadata for the prefill portion of the batch.
    Will be `None` if `num_prefill_tokens == 0`.
    """

    decode: FIDecode | FlashInferTrtllmAPIDecode | None
    """
    Holds the metadata for the decode portion of the batch.
    Will be `None` if `num_decode_tokens == 0`.
    """

    # --- Special Case: Cascade Attention ---

    use_cascade: bool
    """
    If True, the entire batch is a cascade attention call, and the
    `prefill` and `decode` fields will both be None.
    """

    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):
    kv_cache_spec: AttentionSpec
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.attention_config = vllm_config.attention_config
        self._workspace_buffer = None
        self._prefill_wrapper: (
            BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper | None
        ) = None  # Wrapper for prefill/append
        self._noncausal_prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = (
            None  # Wrapper for non-causal prefill (DFlash)
        )
        self._decode_wrapper = None  # Wrapper for decode (general shape)

        if envs.VLLM_BATCH_INVARIANT:
            self.decode_fixed_split_size = 2048
            self.prefill_fixed_split_size = 4096
            self.disable_split_kv = True
        else:
            self.decode_fixed_split_size = -1
            self.prefill_fixed_split_size = -1
            self.disable_split_kv = False

        # SM70 NVFP4: route q=1 decode through the varlen paged-prefill path
        # (SM70PAGED in-kernel-dequant kernel) instead of the Triton
        # in-kernel-dequant split-KV decode kernel, whose gather-based fp4
        # dequant codegen on sm_70 runs at ~1.8 GB/s effective KV bandwidth
        # (~5x below the paged-prefill path), making decode degrade far
        # faster with context. The paged-prefill kernel already services the
        # DFlash2 small-q verify (qo_len << kv_len), so this only changes
        # which kernel the pure q=1 decode step uses.
        self._sm70_varlen_decode = (
            envs.VLLM_SM70_VARLEN_DECODE
            and sm70_nvfp4_kv_enabled()
            and kv_cache_spec.kv_quant_mode != KVQuantMode.NONE
            and str(vllm_config.cache_config.cache_dtype).startswith("nvfp4")
        )

        # Persistent splitd workspaces (see _sm70_splitd_ensure_ws): one
        # shared k/v/q/out set grown geometrically, replacing per-step
        # re-allocations whose 2 x kv_len x 1KB churn pinned ~9GiB of
        # allocator pool at long context. This builder services every
        # full-attention layer of a step (build() runs once per step), so a
        # single buffer set is aliased by all of them safely.
        self._sm70_splitd_ws: tuple[torch.Tensor, torch.Tensor] | None = None
        self._sm70_splitd_ws_cap = 0
        self._sm70_splitd_q_ws: torch.Tensor | None = None
        self._sm70_splitd_out_ws: torch.Tensor | None = None
        # Persistent fp16 shadow-KV staging buffer for the NVFP4 FA2 prefill
        # path (2KB/token, sized by kv_len). Grown geometrically and reused
        # across steps instead of the legacy per-step re-allocation, whose
        # growing segments pinned ~9GiB of allocator pool at long context.
        self._sm70_shadow_kv: torch.Tensor | None = None
        self._sm70_shadow_kv_blocks = 0

        self.compilation_config = vllm_config.compilation_config
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        max_num_pages_per_req = cdiv(
            self.model_config.max_model_len, self.kv_cache_spec.block_size
        )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_reqs = max_num_reqs
        max_num_pages = max_num_reqs * max_num_pages_per_req
        # Persistent uniform masks keep stable addresses for CUDA graphs.
        self._decode_mask_cache: dict[tuple[int, bool], torch.Tensor] = {}
        speculative_config = vllm_config.speculative_config
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        self.enable_cuda_graph = (
            self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        )
        if self.enable_cuda_graph:
            # For full cudagraph capture, one `decode_wrapper` for each batch
            # size is needed for FlashInfer.
            self._decode_wrappers_cudagraph: dict[
                int, BatchDecodeWithPagedKVCacheWrapper
            ] = {}
            self._decode_cudagraph_max_bs = (1 + num_spec_tokens) * max_num_reqs
            if self.compilation_config.max_cudagraph_capture_size is not None:
                self._decode_cudagraph_max_bs = min(
                    self._decode_cudagraph_max_bs,
                    self.compilation_config.max_cudagraph_capture_size,
                )
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.dcp_kv_cache_interleave_size = (
                vllm_config.parallel_config.cp_kv_cache_interleave_size
            )
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.dcp_kv_cache_interleave_size = 1
        self.use_dcp = self.dcp_world_size > 1
        self.dcp_a2a = (
            self.use_dcp and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )

        # Compatible with models with non-uniform per-layer head counts.
        self.num_qo_heads = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or self.model_config.get_num_attention_heads(self.vllm_config.parallel_config)

        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        self.page_size = self.kv_cache_spec.block_size

        if self.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE:
            self.cache_dtype = self.cache_config.cache_dtype
            # Cannot use self.kv_cache_spec.dtype here because kv_cache_spec
            # storage dtype may not be the same as the op dtype (uint8 vs fp8_e4m3)
            self.is_kvcache_nvfp4 = self.cache_dtype.startswith("nvfp4")
            self.sm70_nvfp4_shadow = False
            self.sm70_fp8_shadow = False
            if self.is_kvcache_nvfp4:
                if sm70_nvfp4_kv_enabled():
                    # SM70 NVFP4 KV-cache: store path + native paged attention
                    # read. Query/output stay BF16/FP16 (no FP8); the packed KV
                    # is uint8 (flashinfer maps "nvfp4" -> uint8).
                    self.kv_cache_dtype = FlashInferBackend.get_dtype_for_flashinfer(self.cache_dtype)
                    # The FA2 paged reader cannot consume NVFP4 KV on sm_70
                    # (its in-loop fp4 dequant needs ldmatrix, sm_75+), so the
                    # read side dequantizes each layer's cache to an fp16
                    # shadow paged buffer and runs the mature fp16 paged
                    # kernels (see vllm/v1/attention/ops/sm70_nvfp4_dequant.py).
                    self.sm70_nvfp4_shadow = True
                else:
                    # SM100/SM103: NVFP4 KV cache is serviced by the trtllm-gen
                    # prefill/decode kernels, so both phases must be available.
                    # SM12x: NVFP4 KV cache is serviced by the FlashInfer XQA
                    # decode and native (FA2) prefill paths; no TRTLLM gate.
                    sm100_trtllm_gen_path = current_platform.is_device_capability_family(
                        100
                    )
                    if sm100_trtllm_gen_path and (
                        force_use_trtllm_attention() is False
                        or not supports_trtllm_attention(is_prefill=True)
                        or not supports_trtllm_attention(is_prefill=False)
                    ):
                        raise ValueError(
                            f"--kv-cache-dtype {self.cache_dtype} requires the "
                            "SM100 trtllm-gen "
                            "FlashInfer path."
                        )
                    # The scale search only affects the store kernel. FlashInfer
                    # reads both variants using the same NVFP4 layout.
                    if sm100_trtllm_gen_path:
                        # SM100 trtllm-gen paths never run a native wrapper plan
                        # (prefill uses TRTLLMPrefill, decode uses the trtllm API),
                        # so the dtype name is only ever logged here.
                        self.kv_cache_dtype = "nvfp4"
                    else:
                        # SM12x NVFP4 is serviced by the native FA2/DCP wrapper
                        # plans, which identify NVFP4 KV by its storage dtype
                        # (torch.uint8); the "nvfp4" name is not a torch dtype.
                        self.kv_cache_dtype = (
                            FlashInferBackend.get_dtype_for_flashinfer(self.cache_dtype)
                        )
            else:
                self.kv_cache_dtype = FlashInferBackend.get_dtype_for_flashinfer(
                    self.cache_dtype
                )
                # SM70 FP8 KV-cache: V100 has no FP8 hardware and the
                # flashinfer fp8 paged kernels are not sm_70-safe (the fp8
                # instantiation of BatchPrefill/DecodeWithPagedKVCache fails
                # at launch). Mirror the NVFP4 shadow architecture: the read
                # side dequantizes each layer's fp8 cache to an fp16 shadow
                # paged buffer and runs the mature fp16 paged kernels
                # (see sm70_fp8_paged_dequant in sm70_nvfp4_dequant.py).
                self.sm70_fp8_shadow = self.cache_dtype.startswith(
                    "fp8"
                ) and sm70_flashinfer_attn_enabled()
        else:
            self.cache_dtype = "auto"
            self.is_kvcache_nvfp4 = False
            self.sm70_nvfp4_shadow = False
            self.sm70_fp8_shadow = False
            assert self.kv_cache_spec.dtype == self.model_config.dtype
            self.kv_cache_dtype = self.kv_cache_spec.dtype

        # SM70 splitd D256 paged prefill (1Cat port): exact-tile WMMA kernel
        # on an fp16 NHD KV stage, replacing the FA2 prefill run for steps
        # that pass sm70_splitd_admit. Gate off by default; the op probe is
        # the last term so a missing extension never changes the gate-off
        # behavior.
        self.splitd_enabled = (
            envs.VLLM_SM70_SPLITD_PREFILL
            and self.sm70_nvfp4_shadow
            and self.head_dim == 256
            and self.num_kv_heads == 2
            and self.num_qo_heads // self.num_kv_heads == 6
            and not self.use_dcp
            and bool(_get_sm70_splitd_op())
        )
        if envs.VLLM_SM70_SPLITD_PREFILL and not self.splitd_enabled:
            logger.info(
                "SM70 splitd prefill gate on but disabled: shadow=%s "
                "head_dim=%s kv_heads=%s qo_heads=%s dcp=%s",
                self.sm70_nvfp4_shadow, self.head_dim, self.num_kv_heads,
                self.num_qo_heads, self.use_dcp,
            )

        # Builder-side copy of the impl's flashdecode gate (used by the
        # per-step fd_verify admission; mirrors FlashInferImpl.__init__).
        self._sm70_flashdecode = (
            envs.VLLM_SM70_FLASHDECODE
            and self.sm70_nvfp4_shadow
            and self.head_dim == 256
            and self.num_kv_heads == 2
            and self.num_qo_heads == 12
            and bool(_get_sm70_flashdecode_op())
        )
        # Static verify kv-len buffer (fd_verify): the pointer must stay
        # stable across cudagraph replays; refreshed per step in build().
        self._sm70_fd_verify_kvlens = torch.zeros(
            1, dtype=torch.int32, device=device
        )

        if (
            self.is_kvcache_nvfp4
            and get_flashinfer_layout_string(self.kv_cache_layout) != "HND"
        ):
            # The NVFP4 per-side [data | scale] carve is only byte-coherent
            # under the head-major HND layout (see
            # FlashInferBackend.supported_kv_cache_layouts). NHD would
            # silently corrupt the cache, so fail at init instead.
            raise ValueError(
                "NVFP4 KV cache requires the HND KV cache layout; resolved "
                f"layout is {get_flashinfer_layout_string(self.kv_cache_layout)!r}. "
                "Unset VLLM_KV_CACHE_LAYOUT or set it to 'HND'."
            )

        # Compute per-phase Q dtype.  On SM90 (XQA decode), the prefill and
        # decode phases require different Q dtypes when the KV cache is FP8
        # (FP8-Q for the FI native prefill, BF16/FP16-Q for XQA decode),
        # so both values must be tracked independently.
        self.q_data_type_prefill = self.get_q_data_type(is_prefill=True)
        self.q_data_type_decode = self.get_q_data_type(is_prefill=False)

        # Prefer TRTLLM/XQA for decoding whenever supported. The decode kernel
        # must be selected statically for FULL cudagraph capture.
        can_use_xqa_or_trtllm_gen_decode = can_use_trtllm_attention(
            self.num_qo_heads, self.num_kv_heads, is_prefill=False
        )
        # Page sizes >= 128 require the trtllm-gen GQA/MQA path (guaranteed by
        # get_supported_kernel_block_sizes).
        assert self.page_size <= 64 or (
            current_platform.is_device_capability_family(100)
            and can_use_xqa_or_trtllm_gen_decode
            and self.num_qo_heads // self.num_kv_heads > 1
        ), f"Unexpected FlashInfer page size {self.page_size} without trtllm-gen GQA"
        self.use_trtllm_decode_attention = can_use_xqa_or_trtllm_gen_decode
        self.flashinfer_trtllm_api_decode_kernel: FlashInferDecodeKernel | None = (
            self._get_flashinfer_trtllm_api_decode_kernel()
            if can_use_xqa_or_trtllm_gen_decode
            else None
        )
        # The dedicated FlashInfer XQA API accepts head dimensions in
        # [16, 256] that are divisible by 16. Some hybrid-attention models
        # (for example Gemma 4) use XQA-compatible sliding-attention groups
        # alongside global-attention groups with head_dim=512. Resolve the
        # backend per KV-cache group so the eligible groups still use XQA and
        # the wider groups fall back to native FlashInfer decode.
        if (
            self.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.XQA
            and not _is_xqa_head_dim_supported(self.head_dim)
        ):
            logger.warning_once(
                "FlashInfer XQA decode does not support head_dim=%d; "
                "reverting this KV-cache group to native FlashInfer decode.",
                self.head_dim,
            )
            self.use_trtllm_decode_attention = False
            self.flashinfer_trtllm_api_decode_kernel = None
        if (
            self.use_dcp
            and self.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.XQA
        ):
            logger.warning_once(
                "FlashInfer XQA decode does not support returning LSE and "
                "therefore does not support DCP, reverting to native FlashInfer "
                "decode."
            )
            self.use_trtllm_decode_attention = False
            self.flashinfer_trtllm_api_decode_kernel = None
        self.use_xqa = (
            self.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.XQA
        )
        # Adaptive verification trims drafts on device, so decode query lengths
        # must come from the device qo_indptr; only trtllm-gen supports that
        # (the selector already rejects the other configurations).
        self.use_trtllm_gen_varlen_decode = (
            speculative_config is not None
            and speculative_config.enable_adaptive_verification
            and self.flashinfer_trtllm_api_decode_kernel
            == FlashInferDecodeKernel.TRTLLM_GEN
            and not self.use_dcp
        )
        self._init_reorder_batch_threshold(
            1,
            supports_spec_as_decode=(
                self.flashinfer_trtllm_api_decode_kernel is not None
            ),
            # trtllm-gen decode receives no cp_rank/global-seq-len information,
            # so its end-aligned causal mask is wrong for q_len > 1 over the
            # DCP-interleaved local KV shard (spec token i misses up to
            # (dcp_world_size - 1) * (q_len - 1 - i) KV entries, including its
            # own). Keep the threshold at 1 under DCP so spec queries take the
            # DCP-aware prefill path, until the kernel is CP-aware (compare
            # flash_attn_varlen_func's cp_world_size/cp_rank/cp_tot_seqused_k).
            supports_dcp_with_varlen=False,
        )

        self._cascade_wrapper = None  # Wrapper for cascade attention

        # Global hyperparameters shared by all attention layers
        # TODO: discard this for trtllm-gen backend
        per_layer_parameters = get_per_layer_parameters(
            vllm_config, layer_names, FlashInferImpl
        )
        self.global_hyperparameters = infer_global_hyperparameters(per_layer_parameters)
        self.sm_scale = self.global_hyperparameters.sm_scale
        self.window_left = self.global_hyperparameters.window_left
        self.logits_soft_cap = self.global_hyperparameters.logits_soft_cap
        self.has_sinks = self.global_hyperparameters.has_sinks
        if self.has_sinks and not FlashInferBackend.supports_sink():
            raise NotImplementedError(
                "FlashInfer backend currently does not support attention "
                "sinks, please use trtllm on blackwell or flash attention on "
                "earlier GPUs."
            )
        capability = current_platform.get_device_capability()
        arch = f"sm{capability.major}{capability.minor}" if capability else "unknown"
        decode_backend = (
            self.flashinfer_trtllm_api_decode_kernel.value
            if self.flashinfer_trtllm_api_decode_kernel is not None
            else "flashinfer-native"
        )
        logger.info_once(
            "FlashInfer resolved query dtypes: prefill=%s, decode=%s, "
            "decode_backend=%s, kv_cache_dtype=%s, arch=%s",
            self.q_data_type_prefill,
            self.q_data_type_decode,
            decode_backend,
            self.kv_cache_dtype,
            arch,
        )
        # Preparing persistent buffers
        # Pinned at allocation: the per-step `.pin_memory()` on slices of
        # these buffers is then a no-op instead of a fresh cudaHostAlloc.
        self.paged_kv_indptr = CpuGpuBuffer(
            max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
            pin_memory=PIN_MEMORY,
        )
        self.paged_kv_indices = torch.zeros(
            max_num_pages, dtype=torch.int32, device=self.device
        )
        self.paged_kv_last_page_len = CpuGpuBuffer(
            max_num_reqs, dtype=torch.int32, device=self.device, pin_memory=PIN_MEMORY
        )

        # SM70 NVFP4 decode: the in-kernel-dequant split-KV paged decode
        # (sm70_nvfp4_paged_decode) reads the packed NVFP4 cache directly —
        # no fp16 shadow buffer (which cost 4 GiB/GPU at 262K context), and
        # no FA2 decode plan. Under full CUDA graphs the only per-step state
        # is the pinned->device kv-lens copy (replay-invariant launch). The
        # split-KV partial-state workspace is the only persistent allocation
        # (~1 MB): the grid is (R, H_kv, splits) with
        # R*H_kv*splits <= SM_COUNT + R*H_kv, bounded below.
        if self.sm70_nvfp4_shadow:
            self._sm70_sm_count = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count
            _ratio = self.num_qo_heads // self.num_kv_heads
            _rp = 1 << (_ratio - 1).bit_length()
            # The split count is ceil(SM*SF/(R*H)): the small-R regime needs
            # ~SM*SF entries (plus rounding slack R*H <= max_num_reqs*H), so
            # the sm_count term must be scaled by the split factor (same
            # sizing as _sm70_fd_ws below).
            self._sm70_ws = torch.empty(
                (
                    self._sm70_sm_count * _SM70_FD_SPLIT_FACTOR
                    + max_num_reqs * self.num_kv_heads
                )
                * _rp
                * (self.head_dim + 2),
                dtype=torch.float32,
                device=self.device,
            )
            if self.enable_cuda_graph:
                self._sm70_kvlens_pinned = torch.zeros(
                    max_num_reqs, dtype=torch.int32, pin_memory=True
                )
                self._sm70_kvlens_dev = torch.zeros(
                    max_num_reqs, dtype=torch.int32, device=self.device
                )

        # SM70 NVFP4 shadow decode keeps all per-step state device-side
        # (kv lens buffer + static block table), so draft steps can refresh
        # it in place instead of rebuilding metadata per step.
        self.supports_draft_decode_metadata_update = (
            self.sm70_nvfp4_shadow
            and not self.use_dcp
            and not self.use_trtllm_decode_attention
        )

    def update_draft_decode_metadata(self, metadata: FlashInferMetadata) -> None:
        decode = metadata.decode
        if (
            decode is None
            or decode.seq_lens_source is None
            or decode.shadow_kv_lens_dev is None
        ):
            return
        n = decode.shadow_num_reqs
        # input_buffers.seq_lens is advanced GPU-side between draft steps;
        # mirror it into the kv-lens the kernels read. For the graph-backed
        # builder that is the static capture buffer (D2D refresh keeps the
        # captured pointer); otherwise swap in a fresh device slice cast.
        if hasattr(self, "_sm70_kvlens_dev"):
            self._sm70_kvlens_dev[:n].copy_(decode.seq_lens_source[:n])
        else:
            decode.shadow_kv_lens_dev = decode.seq_lens_source[:n].to(
                torch.int32
            )

    @property
    def kv_cache_layout(self) -> KVCacheLayout:
        return self.cache_config.get_resolved_kv_cache_layout()

    # Keep SM90 prefill/decode Q dtype selection in one place.
    def get_q_data_type(self, is_prefill: bool) -> torch.dtype:
        # The user sets --attention-config.disable_flashinfer_q_quantization
        # to 1 explicitly, use model dtype for query.
        if self.vllm_config.attention_config.disable_flashinfer_q_quantization:
            return self.model_config.dtype

        # self.cache_dtype is resolved per KV-cache group: it is "auto" when
        # this group is unquantized (e.g. --kv-cache-dtype-skip-layers), even
        # if cache_config requests a quantized dtype globally.
        cache_dtype = self.cache_dtype

        # On SM90/SM12x, XQA decode requires BF16/FP16-Q even with FP8 KV cache.
        # FI native prefill on SM90 still uses FP8-Q in that case; SM12x prefill
        # is fa2-only and keeps the model dtype (handled below).
        if (
            (
                current_platform.is_device_capability(90)
                or current_platform.is_device_capability_family(120)
            )
            and not is_prefill
            and force_use_trtllm_attention() is not False
            and cache_dtype.startswith("fp8")
        ):
            return self.model_config.dtype

        # Otherwise, match Q dtype to the KV cache dtype.
        if cache_dtype.startswith("fp8"):
            # FP8-Q requires an fp8 tensor-core attention path. SM90 uses
            # native FA3 for prefill; XQA decode on SM90/SM12x is handled
            # above with model-dtype Q, while SM100 uses trtllm-gen.
            # Architectures with only fa2 (e.g. SM89, SM120) cannot
            # consume FP8 queries, so keep the model dtype for Q there.
            if current_platform.is_device_capability(
                90
            ) or current_platform.is_device_capability_family(100):
                return FlashInferBackend.get_dtype_for_flashinfer(cache_dtype)
            return self.model_config.dtype
        if cache_dtype.startswith("nvfp4"):
            # SM100/SM103 NVFP4 prefill/decode runs on trtllm-gen, which can
            # consume FP8 queries. SM70 (opt-in) and SM12x NVFP4 are serviced
            # by native paged paths, which cannot consume FP8 queries.
            if current_platform.is_device_capability_family(100):
                return FlashInferBackend.get_dtype_for_flashinfer("fp8_e4m3")
            return self.model_config.dtype
        return self.kv_cache_spec.dtype

    @override  # type: ignore[misc]
    @classmethod
    def get_cudagraph_support(
        cls: type["FlashInferMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        """Get the cudagraph support level for FlashInfer attention."""
        # XQA lacks LSE for DCP; DCP also cannot graph variable-length trtllm-gen.
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

        kv_specs = iter_layer_specs(kv_cache_spec)
        num_qo_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        is_xqa_arch = current_platform.is_device_capability(
            90
        ) or current_platform.is_device_capability_family(120)
        has_uniform_batch_support: bool = len(kv_specs) > 0
        for spec in kv_specs:
            if not isinstance(spec, AttentionSpec):
                # FlashInfer only applies to attention, so we don't consider other types
                # of KV spec (e.g. Mamba) here. This is mostly for type checking.
                continue
            if not can_use_trtllm_attention(
                num_qo_heads=num_qo_heads,
                num_kv_heads=spec.num_kv_heads,
                is_prefill=False,
            ) or (is_xqa_arch and not _is_xqa_head_dim_supported(spec.head_size)):
                has_uniform_batch_support = False
                break

        use_non_causal = vllm_config.attention_config.use_non_causal
        if has_uniform_batch_support and (not use_non_causal or is_xqa_arch):
            return AttentionCGSupport.UNIFORM_BATCH
        else:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            buffer_size = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
            if envs.VLLM_BATCH_INVARIANT:
                buffer_size = FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT
            else:
                # FlashInfer prefill temp buffers (batch_prefill_tmp_v, ...)
                # scale with the prefill chunk and query-head footprint, NOT
                # context length. The fixed ~394 MiB default is too small for
                # wide-head models at the default 8192-token chunk on some
                # archs (e.g. sm_120), where FlashInfer hard-errors instead of
                # growing. Size to the batch's head footprint; never shrink
                # below the configured default.
                est = (
                    self.max_num_batched_tokens
                    * self.num_qo_heads
                    * self.head_dim
                    * FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM
                )
                buffer_size = max(buffer_size, est)
            self._workspace_buffer = torch.zeros(
                buffer_size, dtype=torch.uint8, device=self.device
            )
        return self._workspace_buffer

    def set_workspace_buffer(self, workspace_buffer: torch.Tensor):
        self._workspace_buffer = workspace_buffer

    @staticmethod
    def _get_flashinfer_trtllm_api_decode_kernel() -> FlashInferDecodeKernel:
        if current_platform.is_device_capability(
            90
        ) or current_platform.is_device_capability_family(120):
            return FlashInferDecodeKernel.XQA
        assert current_platform.is_device_capability_family(100)
        return FlashInferDecodeKernel.TRTLLM_GEN

    def _compute_decode_query_lens(
        self,
        qo_indptr: torch.Tensor,
        qo_indptr_cpu: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
    ) -> tuple[int, torch.Tensor | None, list[int] | None]:
        """Return the query width, ragged offsets, and effective query lengths."""
        if num_decodes == 0 or num_decode_tokens == 0:
            return 1, None, None

        if not self.use_xqa:
            decode_q_lens = (
                qo_indptr_cpu[1 : num_decodes + 1] - qo_indptr_cpu[:num_decodes]
            )
            q_len_per_req = num_decode_tokens // num_decodes
            if num_decode_tokens % num_decodes == 0 and bool(
                (decode_q_lens == q_len_per_req).all().item()
            ):
                return q_len_per_req, None, None

            # CUDA-graph padding can leave zero-length requests after a
            # uniform speculative decode batch. Keep the packed query and
            # describe it through the trtllm-gen varlen API.
            max_q_len = int(decode_q_lens.max().item())
            if max_q_len > 1:
                return max_q_len, qo_indptr[: num_decodes + 1], None
            return 1, None, None

        decode_q_lens = qo_indptr_cpu[1 : num_decodes + 1] - qo_indptr_cpu[:num_decodes]
        nonzero = decode_q_lens[decode_q_lens > 0]
        q_len_per_req = int(nonzero.max().item()) if nonzero.numel() > 0 else 1
        uniform = nonzero.numel() <= 1 or bool((nonzero == nonzero[0]).all().item())
        if uniform and num_decode_tokens == num_decodes * q_len_per_req:
            return q_len_per_req, None, None

        if q_len_per_req <= 1:
            return 1, None, None

        q_lens = decode_q_lens.tolist()
        return (
            q_len_per_req,
            qo_indptr[: num_decodes + 1],
            q_lens,
        )

    def _get_decode_mask(
        self,
        q_len_per_req: int,
        ragged_q_lens: list[int] | None,
        num_decodes: int,
        causal: bool,
    ) -> torch.Tensor | None:
        """Return the packed XQA draft mask for speculative decode."""
        if q_len_per_req <= 1:
            return None

        if ragged_q_lens is None:
            key = (q_len_per_req, causal)
            buf = self._decode_mask_cache.get(key)
            if buf is None:
                per_req = _make_xqa_draft_block_mask(q_len_per_req, causal, self.device)
                buf = (
                    per_req.unsqueeze(0).expand(self.max_num_reqs, -1, -1).contiguous()
                )
                self._decode_mask_cache[key] = buf
            return buf[:num_decodes]

        return _make_xqa_ragged_draft_block_mask(
            ragged_q_lens, q_len_per_req, causal, self.device
        )

    def _get_prefill_wrapper(
        self,
        causal: bool = True,
    ) -> BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper:
        if not causal:
            if self.use_dcp:
                raise NotImplementedError(
                    "FlashInfer non-causal prefill is not supported with DCP yet."
                )
            if self.is_kvcache_nvfp4 and not self.sm70_nvfp4_shadow:
                raise NotImplementedError(
                    "FlashInfer non-causal attention is not supported with "
                    "NVFP4 KV cache."
                )
            # SM70 NVFP4 shadow: non-causal (e.g. DFlash draft) runs on the
            # fp16 shadow KV; the plan below passes the shadow kv dtype.
            if self._noncausal_prefill_wrapper is None:
                if self.has_sinks and (
                    current_platform.is_device_capability(90)
                    or current_platform.is_device_capability_family(120)
                ):
                    self._noncausal_prefill_wrapper = (
                        BatchAttentionWithAttentionSinkWrapper(
                            self._get_workspace_buffer(),
                            get_flashinfer_layout_string(self.kv_cache_layout),
                            backend="auto",
                            q_data_type=self.q_data_type_prefill,
                            kv_data_type=self.kv_cache_dtype,
                            head_dim_qk=self.head_dim,
                            head_dim_vo=self.head_dim,
                            window_left=self.window_left,
                        )
                    )
                else:
                    self._noncausal_prefill_wrapper = (
                        BatchPrefillWithPagedKVCacheWrapper(
                            self._get_workspace_buffer(),
                            get_flashinfer_layout_string(self.kv_cache_layout),
                            backend="auto",
                        )
                    )
            return self._noncausal_prefill_wrapper

        if self._prefill_wrapper is None:
            if self.use_dcp:
                self._prefill_wrapper = BatchDCPPrefillWrapper(
                    kv_layout=get_flashinfer_layout_string(self.kv_cache_layout),
                    workspace_buffer=self._get_workspace_buffer(),
                    dcp_a2a=self.dcp_a2a,
                )
            else:
                if self.has_sinks and (
                    current_platform.is_device_capability(90)
                    or current_platform.is_device_capability_family(120)
                ):
                    assert not self.is_kvcache_nvfp4
                    self._prefill_wrapper = BatchAttentionWithAttentionSinkWrapper(
                        self._get_workspace_buffer(),
                        get_flashinfer_layout_string(self.kv_cache_layout),
                        backend="auto",
                        q_data_type=self.q_data_type_prefill,
                        kv_data_type=self.kv_cache_dtype,
                        head_dim_qk=self.head_dim,
                        head_dim_vo=self.head_dim,
                        window_left=self.window_left,
                    )
                else:
                    # NVFP4 KV cache: SM100/SM103 uses trtllm-gen; SM70
                    # (opt-in) and SM12x read it via the native paged path.
                    backend = _nvfp4_kv_backend(self.is_kvcache_nvfp4)
                    self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                        self._get_workspace_buffer(),
                        get_flashinfer_layout_string(self.kv_cache_layout),
                        backend=backend,
                    )
        assert self._prefill_wrapper is not None
        return self._prefill_wrapper

    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False):
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                paged_kv_indices = self.paged_kv_indices
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
            else:
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_last_page_len = None
            # NVFP4 KV cache: SM100/SM103 uses trtllm-gen; SM70 (opt-in)
            # and SM12x read it via the native paged path.
            backend = _nvfp4_kv_backend(self.is_kvcache_nvfp4)
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                get_flashinfer_layout_string(self.kv_cache_layout),
                use_cuda_graph=use_cudagraph,
                paged_kv_indptr_buffer=paged_kv_indptr,
                paged_kv_indices_buffer=paged_kv_indices,
                paged_kv_last_page_len_buffer=paged_kv_last_page_len,
                # Tensor cores are enabled by default because the perf would be
                # at least as good as cuda cores for all attention ops in latest
                # gpus.
                use_tensor_cores=True,
                backend=backend,
            )

            # save the decode wrapper
            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper
            else:
                self._decode_wrapper = decode_wrapper

        return decode_wrapper

    def _get_cascade_wrapper(self):
        if self._cascade_wrapper is None:
            self._cascade_wrapper = MultiLevelCascadeAttentionWrapper(
                2,
                self._get_workspace_buffer(),
                get_flashinfer_layout_string(self.kv_cache_layout),
            )
        return self._cascade_wrapper

    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np: np.ndarray,
        seq_lens_np: np.ndarray,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
        page_size: int,
    ) -> torch.Tensor:
        """
        Compute paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len for FlashInfer
        attention.

        Results are stored in self.paged_kv_indptr,
        self.paged_kv_indices, self.paged_kv_last_page_len buffers.

        Returns paged_kv_indices, a GPU tensor with shape [num_actual_pages].
        """
        # write self.paged_kv_indptr_cpu inplace (0-index is always 0)
        np.cumsum(
            num_blocks_np,
            dtype=np.int32,
            out=self.paged_kv_indptr.np[1 : num_reqs + 1],
        )
        paged_kv_indptr = self.paged_kv_indptr.gpu[: num_reqs + 1]
        paged_kv_indptr_cpu = self.paged_kv_indptr.cpu[: num_reqs + 1]
        if PIN_MEMORY:
            paged_kv_indptr_cpu = paged_kv_indptr_cpu.pin_memory()
        paged_kv_indptr.copy_(paged_kv_indptr_cpu, non_blocking=True)

        # write self.paged_kv_indices inplace
        num_actual_pages = self.paged_kv_indptr.np[num_reqs]
        paged_kv_indices = self.paged_kv_indices[:num_actual_pages]
        _copy_page_indices_kernel[(num_reqs,)](
            paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )

        # write self.paged_kv_last_page_len_cpu inplace
        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(
            (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),
            page_size,
            paged_kv_last_page_len_np,
        )
        paged_kv_last_page_len_cpu = self.paged_kv_last_page_len.cpu[:num_reqs]
        if PIN_MEMORY:
            paged_kv_last_page_len_cpu = paged_kv_last_page_len_cpu.pin_memory()
        self.paged_kv_last_page_len.gpu[:num_reqs].copy_(
            paged_kv_last_page_len_cpu, non_blocking=True
        )
        return paged_kv_indices

    def _sm70_splitd_ensure_ws(
        self,
        kv_len: int,
        q_len: int,
        dev: torch.device,
        stage_kv: bool = True,
    ) -> tuple:
        """Persistent splitd workspaces, shared by every full-attention
        layer of a step (build() runs once per step and its metadata is
        consumed by all full-attn layers sequentially on one stream, so
        aliasing one buffer set is safe). Grows geometrically up to the
        policy's 262144-kv cap (sm70_splitd_admit), ~0.6GiB at cap —
        replacing the legacy per-step allocations (k/v sized 2 x kv_len x
        1KB, q/out 2 x q_len x 12 x 1KB) that churned and pinned allocator
        pool at long context.

        ``stage_kv=False`` (NVFP4 in-kernel-dequant path) skips the fp16
        K/V staging buffers entirely: the kernel reads the packed cache
        in-kernel.

        Returns (k_ws, v_ws, q_ws, out_ws) views exact-sized for this step
        (k_ws/v_ws are None when stage_kv=False).
        """
        page = self.page_size
        if stage_kv and (
            self._sm70_splitd_ws is None or kv_len > self._sm70_splitd_ws_cap
        ):
            new_cap = min(max(kv_len, 65536) * 2, 262144)
            m_cap = (new_cap + page - 1) // page
            k = torch.zeros(
                m_cap,
                page,
                self.num_kv_heads,
                self.head_dim,
                dtype=torch.float16,
                device=dev,
            )
            self._sm70_splitd_ws = (k, torch.zeros_like(k))
            self._sm70_splitd_ws_cap = new_cap
        if self._sm70_splitd_q_ws is None:
            q = torch.empty(
                1,
                8192,
                self.num_qo_heads,
                self.head_dim,
                dtype=torch.float16,
                device=dev,
            )
            self._sm70_splitd_q_ws = q
            self._sm70_splitd_out_ws = torch.empty_like(q)
        if stage_kv:
            k, v = self._sm70_splitd_ws
            k_ws, v_ws = k, v
            m = (kv_len + page - 1) // page
        else:
            k_ws, v_ws = None, None
        q_padded = ((q_len + 63) // 64) * 64
        return (
            k_ws[:m] if stage_kv else None,
            v_ws[:m] if stage_kv else None,
            self._sm70_splitd_q_ws[:, :q_padded],
            self._sm70_splitd_out_ws[:, :q_padded],
        )

    def _sm70_ensure_shadow_kv(
        self, num_blocks: int, dev: torch.device
    ) -> torch.Tensor:
        """Persistent fp16 shadow-KV staging buffer for the NVFP4 FA2
        prefill path. Grown geometrically up to max_model_len and reused
        across steps (the 16 full-attn layers of a step write into it
        sequentially), replacing the legacy per-step re-allocation whose
        growing per-chunk segments accumulated ~9GiB of allocator pool at
        long context.

        Returns a (num_blocks, num_kv_heads, page_size, 2*head_dim) view.
        """
        page = self.page_size
        if self._sm70_shadow_kv is None or num_blocks > self._sm70_shadow_kv_blocks:
            cap_blocks = (self.model_config.max_model_len + page - 1) // page
            new_blocks = min(max(num_blocks * 2, 1), cap_blocks)
            new_blocks = max(new_blocks, num_blocks)
            self._sm70_shadow_kv = torch.empty(
                new_blocks,
                self.num_kv_heads,
                page,
                2 * self.head_dim,
                dtype=torch.float16,
                device=dev,
            )
            self._sm70_shadow_kv_blocks = new_blocks
        return self._sm70_shadow_kv[:num_blocks]

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        causal = common_attn_metadata.causal
        route_decode = causal or self.use_xqa
        if route_decode and self._sm70_varlen_decode:
            # SM70 NVFP4: service q=1 decode as small-q varlen prefill so
            # the paged-prefill kernel (in-kernel fp4 dequant) handles it
            # instead of the slow Triton decode kernel (see the flag).
            route_decode = False
        if route_decode:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.reorder_batch_threshold,
                    require_uniform=not self.use_xqa,
                )
            )
        else:
            num_decodes = 0
            num_prefills = num_reqs
            num_decode_tokens = 0
            num_prefill_tokens = num_actual_tokens

        page_size = self.page_size
        max_seq_len = common_attn_metadata.max_seq_len
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        qo_indptr = common_attn_metadata.query_start_loc
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu

        # Step 1: Decide which dispatch modes to use:
        # - Cascade attention (distinct mode)
        # - Prefill (FI native or TRTLLM)
        # - Decode (FI native, XQA, or trtllm-gen)
        use_cascade = common_prefix_len > 0
        uses_spec_reorder = self.reorder_batch_threshold > 1
        # Page sizes >= 128 must use trtllm-gen; force it for prefill too.
        prefill_force_trtllm = (
            True if page_size >= 128 else self.attention_config.use_trtllm_attention
        )
        prefill_use_trtllm = causal and use_trtllm_attention(
            self.num_qo_heads,
            self.num_kv_heads,
            num_prefill_tokens,
            max_seq_len,
            self.dcp_world_size,
            self.cache_dtype,
            self.q_data_type_prefill,
            is_prefill=True,
            force_use_trtllm=prefill_force_trtllm,
            has_sinks=self.has_sinks,
            has_spec=uses_spec_reorder,
        )
        decode_with_flashinfer_trtllm_api = self.use_trtllm_decode_attention and (
            causal or self.use_xqa
        )

        if not causal and self.use_dcp:
            raise NotImplementedError(
                "FlashInfer non-causal prefill is not supported with DCP yet."
            )
        if not causal and self.use_trtllm_decode_attention:
            logger.warning_once(
                "Using FlashInfer for draft model non-causal attention; TRTLLM "
                "can still be used for target model causal attention."
            )
        all_uses_trtllm = causal and (
            (num_prefills == 0 or prefill_use_trtllm)
            and (num_decodes == 0 or decode_with_flashinfer_trtllm_api)
        )

        if not all_uses_trtllm:
            sinks_would_be_dropped = self.use_dcp or use_cascade
            if self.has_sinks and sinks_would_be_dropped:
                raise NotImplementedError(
                    "FlashInfer BatchDCPPrefillWrapper and "
                    "MultiLevelCascadeAttentionWrapper do not support attention sinks. "
                    "Please use TRTLLM on Blackwell or FlashAttention on earlier GPUs."
                )

            if not self.global_hyperparameters.has_same_window_lefts:
                raise ValueError(
                    "Window left is not the same for all layers. "
                    "One potential fix is to set disable_sliding_window=True"
                )

            assert self.global_hyperparameters.has_same_all_params, (
                "FlashInfer backend currently only supports models in which "
                "all layers share the same values for the following "
                "hyperparameters: `window_left`, `logits_soft_cap`, "
                "`sm_scale`."
            )

        # Hostopt: early fd_verify admission. A single small-q (<=8) causal
        # prefill under the SM70 NVFP4 flashdecode route is served in
        # forward() by the sm70_flashdecode op, so the FA2 paged plan and
        # every CPU-side input feeding it are dead work for this step — and
        # the seq_lens D2H sync below would drain the GPU pipeline before
        # the graph launches. Admit here (CPU-only test) so the plan-free
        # path in the prefill pathway can skip them; the kv len is copied
        # GPU->GPU from the attention seq_lens buffer instead of the
        # synced CPU value.
        # The plan-free path is only taken when forward() is guaranteed to
        # route the step to the sm70_flashdecode op: fd_verify plus its
        # fp16/bf16-query and no-sinks requirements, and not diverted to
        # raw-verify or the native prefill.
        fd_verify_admit = (
            envs.VLLM_SM70_DECODE_HOSTOPT
            and self._sm70_flashdecode
            and (self.sm70_nvfp4_shadow or self.sm70_fp8_shadow)
            and os.environ.get("VLLM_SM70_RAW_VERIFY", "0") != "1"
            and not _SM70_NATIVE_PREFILL
            and not self.use_dcp
            and not use_cascade
            and num_decodes == 0
            and num_prefills == 1
            and int(qo_indptr_cpu[1] - qo_indptr_cpu[0]) <= 8
            and causal
            and not self.logits_soft_cap
            and (self.window_left is None or self.window_left <= 0)
            and not self.has_sinks
            and self.q_data_type_prefill
            in (torch.float16, torch.bfloat16)
        )
        if fd_verify_admit and self.splitd_enabled:
            # The splitd policy may take the step; its admission test needs
            # a kv len. The CPU upper bound may over-estimate, which only
            # ever rejects fd_verify (falling back to the sync path), so it
            # is safe; without a CPU bound there is no plan-free admission.
            seq_lens_cpu_ub = common_attn_metadata.seq_lens_cpu_upper_bound
            if seq_lens_cpu_ub is None:
                fd_verify_admit = False
            else:
                fd_verify_admit = not sm70_splitd_admit(
                    1,
                    int(qo_indptr_cpu[1] - qo_indptr_cpu[0]),
                    int(seq_lens_cpu_ub[0]),
                    causal,
                    self.q_data_type_prefill,
                )

        # Step 2: Initialize the output metadata
        # Leave prefill/decode/cascade_wrapper empty, to be populated
        # case by case depending on the batch contents and backend selection.
        attn_metadata = FlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            q_data_type_prefill=self.q_data_type_prefill,
            q_data_type_decode=self.q_data_type_decode,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            causal=causal,
            use_cascade=use_cascade,
            prefill=None,
            decode=None,
            cascade_wrapper=None,
        )

        # Guard access to seq_lens_cpu, which may not always be needed
        # and can be expensive to retrieve in async mode.
        # When all attention (both prefill and decode) uses TRTLLM,
        # seq_lens_cpu is not needed since TRTLLM paths use GPU tensors
        # (block_tables, seq_lens) directly.
        # SM70 hostopt: a pure shadow-decode step consumes only the device
        # kv-lens buffer (the in-kernel-dequant op reads it D2D); the CPU
        # readback here would drain the whole queued GPU pipeline once per
        # draft step of the speculator.
        sm70_shadow_decode_nosync = (
            envs.VLLM_SM70_DECODE_HOSTOPT
            and self.sm70_nvfp4_shadow
            and num_prefills == 0
            and num_decodes > 0
            and not self.use_dcp
            and not use_cascade
            and not self.use_trtllm_decode_attention
        )
        needs_seq_lens_cpu = (
            (self.use_dcp or use_cascade or not all_uses_trtllm)
            and not fd_verify_admit
            and not sm70_shadow_decode_nosync
        )
        if needs_seq_lens_cpu:
            with gpu_sync_allowed():
                seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        else:
            seq_lens_cpu = None

        # Adjust seq_lens_cpu for DCP
        if self.use_dcp:
            assert seq_lens_cpu is not None
            if num_prefills > 0:
                # Other attention groups may reuse the same common metadata.
                seq_lens_cpu = seq_lens_cpu.clone()
                qo_indptr_prefill_cpu = (
                    qo_indptr_cpu[num_decodes:] - qo_indptr_cpu[num_decodes]
                )
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                seq_lens_cpu[num_decodes:] = (
                    seq_lens_cpu[num_decodes:] - query_lens_prefill_cpu
                )

            seq_lens_cpu = get_dcp_local_seq_lens(
                seq_lens_cpu,
                self.dcp_world_size,
                self.dcp_rank,
                self.dcp_kv_cache_interleave_size,
            )

        # Native paged attention consumes rank-local lengths (context only
        # for DCP prefills), so derive its page counts after the conversion.
        if seq_lens_cpu is not None:
            seq_lens_np = seq_lens_cpu.numpy()
            num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size
        else:
            seq_lens_np = None
            num_blocks_np = None

        # Adjust num_block_np for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
            assert common_prefix_len % page_size == 0
            num_common_kv_blocks = common_prefix_len // page_size
            num_blocks_np -= num_common_kv_blocks

        # Compute paged_kv_indices if necessary
        # paged_kv_indices is only needed for FlashInfer native paths;
        # XQA/trtllm-gen paths use block_tables directly on GPU.
        needs_native_paged_prefill = (
            num_prefills > 0 and not prefill_use_trtllm and not fd_verify_admit
        )
        needs_native_paged_decode = (
            num_decodes > 0
            and not decode_with_flashinfer_trtllm_api
            # The SM70 NVFP4 in-kernel-dequant flashdecode op reads the
            # packed cache with its own block table and consumes only the
            # kv lens; the native FlashInfer paged buffers are never used
            # on that path (pure decode only — mixed batches still need
            # them for the prefill plan).
            and not (self.sm70_nvfp4_shadow and num_prefills == 0)
        )
        needs_paged_kv_indices = (
            use_cascade or needs_native_paged_prefill or needs_native_paged_decode
        )
        if needs_paged_kv_indices:
            assert num_blocks_np is not None
            assert seq_lens_np is not None
            paged_kv_indices = self._compute_flashinfer_kv_metadata(
                num_blocks_np,
                seq_lens_np,
                block_table_tensor,
                num_reqs,
                page_size,
            )
        else:
            paged_kv_indices = None

        # Early-out for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
            # Grab the blocks of the shared prefix from the first request.
            num_common_kv_blocks = common_prefix_len // page_size

            # Create CPU versions directly for cascade (no GPU versions needed)
            shared_qo_indptr_cpu = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indptr_cpu = torch.tensor(
                [0, num_common_kv_blocks], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indices_cpu = block_table_tensor[0, :num_common_kv_blocks]
            shared_kv_last_page_len_cpu = torch.tensor(
                [page_size], dtype=torch.int32, device="cpu"
            )

            # Remove the blocks of the shared prefix from all requests.
            block_table_tensor = block_table_tensor[:, num_common_kv_blocks:]
            num_blocks_np -= num_common_kv_blocks

            assert paged_kv_indices is not None
            paged_kv_indptr_cpu = self.paged_kv_indptr.cpu[: 1 + num_reqs]
            paged_kv_last_page_len_cpu = self.paged_kv_last_page_len.cpu[:num_reqs]

            attn_metadata.cascade_wrapper = self._get_cascade_wrapper()
            # Cascade attention must use the same q dtype for prefill and decode
            # because it does not support FP8 kv-cache or FP8 query yet.
            assert self.q_data_type_prefill == self.q_data_type_decode
            attn_metadata.cascade_wrapper.plan(
                qo_indptr_arr=[shared_qo_indptr_cpu, qo_indptr_cpu],
                paged_kv_indptr_arr=[shared_kv_page_indptr_cpu, paged_kv_indptr_cpu],
                paged_kv_indices_arr=[shared_kv_page_indices_cpu, paged_kv_indices],
                paged_kv_last_page_len=[
                    shared_kv_last_page_len_cpu,
                    paged_kv_last_page_len_cpu,
                ],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                causal=True,
                sm_scale=self.sm_scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                q_data_type=self.q_data_type_prefill,
                kv_data_type=self.kv_cache_dtype,
            )
            return attn_metadata

        # Step 3: Handle prefill and decode pathways case by case
        ## PREFILL PATHWAY
        if num_prefills > 0:
            # Slices for shared prefill metadata
            prefill_start = num_decodes
            qo_indptr_prefill_cpu = (
                qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]
            )
            assert qo_indptr_prefill_cpu.shape[0] == num_prefills + 1

            if prefill_use_trtllm:
                # TRTLLM prefill has no cross-rank combine for DCP-sharded KV;
                # use_trtllm_attention never selects it when DCP is enabled.
                assert not self.use_dcp
                # Create GPU versions
                qo_indptr_prefill_gpu = (
                    qo_indptr[prefill_start:] - qo_indptr[prefill_start]
                )
                # Compute cum_seq_lens_kv on GPU to avoid CPU sync.
                # This is the cumulative sum of the number of KV cache
                # blocks per prefill request.
                prefill_seq_lens = seq_lens[prefill_start:]
                num_blocks_per_req = (prefill_seq_lens + page_size - 1) // page_size
                paged_kv_indptr_prefill_gpu = self.paged_kv_indptr.gpu[
                    prefill_start : num_reqs + 1
                ]
                # Assign to slice to avoid cpu sync.
                paged_kv_indptr_prefill_gpu[:1] = 0
                torch.cumsum(
                    num_blocks_per_req,
                    dim=0,
                    out=paged_kv_indptr_prefill_gpu[1:],
                )
                # Compute max_q_len for prefill requests
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                max_q_len_prefill = int(query_lens_prefill_cpu.max().item())
                attn_metadata.prefill = TRTLLMPrefill(
                    block_tables=block_table_tensor[prefill_start:],
                    seq_lens=prefill_seq_lens,
                    cum_seq_lens_q=qo_indptr_prefill_gpu,
                    cum_seq_lens_kv=paged_kv_indptr_prefill_gpu,
                    max_q_len=max_q_len_prefill,
                    max_seq_len=max_seq_len,
                )
            else:
                # SM70 NVFP4 native paged prefill (in-kernel dequant) needs
                # no FA2 wrapper/plan; everything below is plan-free for it.
                native_prefill = self.sm70_nvfp4_shadow and _SM70_NATIVE_PREFILL
                prefill_wrapper = (
                    None if native_prefill
                    else self._get_prefill_wrapper(causal=attn_metadata.causal)
                )
                if fd_verify_admit:
                    # Hostopt plan-free path: forward() routes this step to
                    # the sm70_flashdecode op, which reads the block table
                    # and a static GPU kv-len buffer — the FA2 wrapper is
                    # never planned or run, so none of the CPU paged buffers
                    # (or the seq_lens D2H sync that feeds them) are needed.
                    assert prefill_wrapper is not None
                    self._sm70_fd_verify_kvlens.copy_(
                        seq_lens[prefill_start : prefill_start + 1],
                        non_blocking=True,
                    )
                    attn_metadata.prefill = FIPrefill(
                        wrapper=prefill_wrapper,
                        block_tables=block_table_tensor[prefill_start:],
                        seq_lens=seq_lens[prefill_start:],
                        cum_seq_lens_q=qo_indptr[prefill_start:],
                        fd_verify=True,
                        fd_kv_lens_dev=self._sm70_fd_verify_kvlens,
                    )
                    return attn_metadata
                # Slicing CPU buffers that are only needed for FI native prefills
                paged_kv_last_page_len_prefill_cpu = self.paged_kv_last_page_len.cpu[
                    prefill_start:num_reqs
                ]
                assert paged_kv_last_page_len_prefill_cpu.shape[0] == num_prefills
                paged_kv_indptr_prefill_cpu = self.paged_kv_indptr.cpu[
                    prefill_start : num_reqs + 1
                ]
                assert paged_kv_indptr_prefill_cpu.shape[0] == num_prefills + 1
                # plan() copies these to the GPU with non_blocking=True;
                # stage them in pinned memory so the copies stay async
                # (the reused buffers themselves are intentionally not pinned).
                if PIN_MEMORY:
                    qo_indptr_prefill_cpu = qo_indptr_prefill_cpu.pin_memory()
                    paged_kv_indptr_prefill_cpu = (
                        paged_kv_indptr_prefill_cpu.pin_memory()
                    )
                    paged_kv_last_page_len_prefill_cpu = (
                        paged_kv_last_page_len_prefill_cpu.pin_memory()
                    )
                # plan() also copies kv_lens to the GPU with non_blocking=True,
                # deriving them via get_seq_lens() into unpinned memory when
                # seq_lens is not passed; pass an explicit pinned copy of the
                # same values so that copy stays async too.
                kv_lens_prefill_cpu = get_seq_lens(
                    paged_kv_indptr_prefill_cpu,
                    paged_kv_last_page_len_prefill_cpu,
                    page_size,
                )
                if PIN_MEMORY:
                    kv_lens_prefill_cpu = kv_lens_prefill_cpu.pin_memory()
                shadow_kv = None
                splitd = None
                fd_verify = False
                shadow_kv_lens = None
                shadow_kv_windowed = False
                shadow_bt_start = 0
                if self.use_dcp:
                    assert not native_prefill
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                    prefill_wrapper.plan(
                        qo_indptr_cpu=qo_indptr_prefill_cpu,
                        paged_kv_indptr_cpu=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len_cpu=paged_kv_last_page_len_prefill_cpu,
                        kv_lens_cpu=kv_lens_prefill_cpu,
                        page_size=self.page_size,
                        num_qo_heads=self.num_qo_heads,
                        dcp_world_size=self.dcp_world_size,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type_prefill,
                        kv_cache_dtype=self.kv_cache_dtype,
                        prefill_fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                    )
                else:
                    if prefill_wrapper is not None:
                        assert isinstance(
                            prefill_wrapper,
                            BatchPrefillWithPagedKVCacheWrapper,
                        )
                    # The SM100 trtllm-gen NVFP4 kernel only supports FP8
                    # output. SM70 (opt-in) and SM12x NVFP4 are serviced by
                    # the native FA2 path, which writes model-dtype output
                    # directly.
                    o_dtype = _nvfp4_o_dtype(
                        self.is_kvcache_nvfp4, self.model_config.dtype
                    )
                    raw_disable_split = False
                    n_pre = qo_indptr_prefill_cpu.shape[0] - 1
                    q_len0 = int(
                        (qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1])[0]
                    )
                    # SM70 NVFP4 small-q verify/decode: plan the paged
                    # prefill wrapper directly against the packed NVFP4
                    # cache. The SM70PAGED kernel dequantizes in-kernel, so
                    # no fp16 shadow staging (a full-context pass per step
                    # that scales the decode step with context). Covers spec
                    # verify (q=6) and pure decode (q=1); full-context only
                    # (windowed layers keep the windowed-shadow path).
                    # Disabled by default: the current flashinfer SM70PAGED
                    # fp4 prefill kernel faults at every kv length (Xid 43;
                    # repro /root/run/test_sm70paged_hang.py) — enable only
                    # after that kernel is fixed.
                    _sm70_raw_verify = (
                        os.environ.get("VLLM_SM70_RAW_VERIFY", "0") == "1"
                        and self.sm70_nvfp4_shadow
                        and not self.sm70_fp8_shadow
                        and n_pre == 1
                        and q_len0 <= 8
                        and (self.window_left is None or self.window_left <= 0)
                    )
                    if (
                        (self.sm70_nvfp4_shadow or self.sm70_fp8_shadow)
                        and not _sm70_raw_verify
                    ):
                        if native_prefill:
                            # Native in-kernel-dequant prefill: the flash
                            # kernel reads the packed NVFP4 cache directly —
                            # device metadata only, no shadow buffer, no
                            # FA2 plan.
                            qo_indptr_prefill_dev = qo_indptr_prefill_cpu.to(
                                self.device,
                                dtype=torch.int32,
                                non_blocking=True,
                            )
                            kv_lens_prefill_dev = kv_lens_prefill_cpu.to(
                                self.device,
                                dtype=torch.int32,
                                non_blocking=True,
                            )
                            native_max_q_len = int(
                                (
                                    qo_indptr_prefill_cpu[1:]
                                    - qo_indptr_prefill_cpu[:-1]
                                ).max().item()
                            )
                            plan_kv_dtype = None
                        else:
                            # The fp16 shadow buffer receives the
                            # dequantized KV per layer at forward time; plan
                            # the fp16 paged kernel against its identity
                            # block table.
                            # Sliding-window layers only attend to the last
                            # window_left tokens, so the fp16 shadow buffer
                            # only needs to hold those (a full-context shadow
                            # wastes ~0.5GiB/GPU at 260K). Slicing is only
                            # safe for a single prefill request.
                            shadow_kv_windowed = (
                                self.window_left is not None
                                and self.window_left > 0
                                and qo_indptr_prefill_cpu.shape[0] - 1 == 1
                            )
                            if shadow_kv_windowed:
                                # Token-aligned window tail: the shadow holds
                                # the pages covering
                                # [max(0, kv_len - window_left), kv_len).
                                # FA2's window_left then keeps exactly the last
                                # window_left tokens. Holding a whole page per
                                # side avoids sub-page misalignment (the
                                # window start need not be page-aligned).
                                _win_start_tok = torch.clamp_min(
                                    kv_lens_prefill_cpu - self.window_left, 0
                                )
                                # Page index of the window start; the shadow
                                # holds from that page's first token onward.
                                _win_start_page = (
                                    _win_start_tok // self.page_size
                                )
                                shadow_bt_start = int(_win_start_page[0])
                                shadow_kv_lens = (
                                    kv_lens_prefill_cpu
                                    - _win_start_page * self.page_size
                                )
                            else:
                                shadow_kv_lens = kv_lens_prefill_cpu
                            (
                                plan_indptr,
                                plan_indices,
                                plan_last,
                                _,
                                shadow_blocks,
                            ) = sm70_shadow_plan_tensors(
                                shadow_kv_lens, self.page_size
                            )
                            # Skip the fp16 shadow buffer when splitd will
                            # handle this step (it stages its own k/v and never
                            # reads shadow_kv — see _sm70_splitd_run). Saves
                            # ~0.5GiB/GPU on single-request prefills.
                            n_pre = qo_indptr_prefill_cpu.shape[0] - 1
                            qo_diff = (
                                qo_indptr_prefill_cpu[1:]
                                - qo_indptr_prefill_cpu[:-1]
                            )
                            splitd_admit = (
                                self.splitd_enabled
                                and n_pre == 1
                                and sm70_splitd_admit(
                                    1,
                                    int(qo_diff[0]),
                                    int(kv_lens_prefill_cpu[0]),
                                    attn_metadata.causal,
                                    self.q_data_type_prefill,
                                )
                            )
                            # SM70 flashdecode verify: a single small-q
                            # prefill (spec verify, q_len <= 8) routes to
                            # the in-kernel-dequant flashdecode op in
                            # forward — 1x NVFP4 KV reads, no fp16 shadow
                            # staging (staging the full ctx per step is
                            # the long-ctx decode slope).
                            fd_verify = (
                                self._sm70_flashdecode
                                and n_pre == 1
                                and int(qo_diff[0]) <= 8
                                and attn_metadata.causal
                                and not self.logits_soft_cap
                                and not splitd_admit
                            )
                            if fd_verify:
                                shadow_kv = None
                                self._sm70_fd_verify_kvlens.copy_(
                                    kv_lens_prefill_cpu[:1].to(torch.int32),
                                    non_blocking=True,
                                )
                            elif splitd_admit:
                                shadow_kv = None
                            else:
                                shadow_kv = self._sm70_ensure_shadow_kv(
                                    shadow_blocks,
                                    self.paged_kv_indptr.gpu.device,
                                )
                            plan_kv_dtype = torch.float16
                            if self.splitd_enabled:
                                # SM70 splitd D256 paged prefill admission
                                # (per step; a miss keeps the FA2 path):
                                # single prefill request, exact-tile q/kv.
                                n_pre = qo_indptr_prefill_cpu.shape[0] - 1
                                qo_diff = (
                                    qo_indptr_prefill_cpu[1:]
                                    - qo_indptr_prefill_cpu[:-1]
                                )
                                if n_pre == 1 and sm70_splitd_admit(
                                    1,
                                    int(qo_diff[0]),
                                    int(kv_lens_prefill_cpu[0]),
                                    attn_metadata.causal,
                                    self.q_data_type_prefill,
                                ):
                                    kv_len = int(kv_lens_prefill_cpu[0])
                                    q_len = int(qo_diff[0])
                                    dev = self.paged_kv_indptr.gpu.device
                                    # Size the KV workspace in whole cache
                                    # pages covering kv_len (page_size is a
                                    # multiple of the kernel's 32-column
                                    # tile, so capacity stays %32==0 and the
                                    # padded tail rows are excluded by the
                                    # kernel's mask). The causal offset uses
                                    # the true (q_len, kv_len); the query/
                                    # output buffers pad to the 64-row tile
                                    # and forward slices the tail off.
                                    m_blocks = (
                                        (kv_len + self.page_size - 1)
                                        // self.page_size
                                    )
                                    # NVFP4 in-kernel-dequant path: the
                                    # kernel reads the packed cache
                                    # directly, so skip the fp16 K/V stage
                                    # buffers (~0.6GiB at the 260K cap).
                                    inline_nvfp4 = (
                                        _SM70_SPLITD_NVFP4_INLINE
                                        and bool(
                                            _get_sm70_splitd_nvfp4_op()
                                        )
                                    )
                                    (
                                        k_ws,
                                        v_ws,
                                        q_ws,
                                        out_ws,
                                    ) = self._sm70_splitd_ensure_ws(
                                        kv_len,
                                        q_len,
                                        dev,
                                        stage_kv=not inline_nvfp4,
                                    )
                                    splitd = (
                                        q_len,
                                        kv_len,
                                        block_table_tensor[
                                            prefill_start, :m_blocks
                                        ].contiguous(),
                                        torch.arange(
                                            m_blocks,
                                            dtype=torch.int32,
                                            device=dev,
                                        ).unsqueeze(0),
                                        k_ws,
                                        v_ws,
                                        q_ws,
                                        out_ws,
                                    )
                    else:
                        # Raw NVFP4 prefill (SM70PAGED in-kernel dequant):
                        # force the non-split-KV kernel — the split-KV
                        # worker path hangs on small-kv shapes (graph
                        # capture).
                        raw_disable_split = True
                        shadow_kv = None
                        plan_indptr = paged_kv_indptr_prefill_cpu
                        plan_indices = paged_kv_indices
                        plan_last = paged_kv_last_page_len_prefill_cpu
                        plan_kv_dtype = self.kv_cache_dtype
                    if prefill_wrapper is not None:
                        prefill_wrapper.plan(
                            qo_indptr=qo_indptr_prefill_cpu,
                            paged_kv_indptr=plan_indptr,
                            paged_kv_indices=plan_indices,
                            paged_kv_last_page_len=plan_last,
                            # The FA2 kernel indexes the paged cache by
                            # ``seq_lens``; for the NVFP4 shadow path that
                            # cache is the (possibly windowed) shadow
                            # buffer, so use its length, not the full KV.
                            seq_lens=(
                                shadow_kv_lens
                                if shadow_kv_lens is not None
                                else kv_lens_prefill_cpu
                            ),
                            num_qo_heads=self.num_qo_heads,
                            num_kv_heads=self.num_kv_heads,
                            head_dim_qk=self.head_dim,
                            page_size=self.page_size,
                            causal=attn_metadata.causal,
                            sm_scale=self.sm_scale,
                            window_left=self.window_left,
                            logits_soft_cap=self.logits_soft_cap,
                            q_data_type=self.q_data_type_prefill,
                            kv_data_type=plan_kv_dtype,
                            o_data_type=o_dtype,
                            fixed_split_size=self.prefill_fixed_split_size,
                            disable_split_kv=(
                                self.disable_split_kv or raw_disable_split
                            ),
                        )
                qo_indptr_prefill_gpu = (
                    qo_indptr[prefill_start:] - qo_indptr[prefill_start]
                )
                attn_metadata.prefill = FIPrefill(
                    wrapper=prefill_wrapper,
                    block_tables=block_table_tensor[prefill_start:],
                    seq_lens=seq_lens[prefill_start:],
                    cum_seq_lens_q=qo_indptr_prefill_gpu,
                    shadow_kv=shadow_kv,
                    shadow_kv_lens=shadow_kv_lens,
                    shadow_kv_windowed=shadow_kv_windowed,
                    shadow_bt_start=shadow_bt_start,
                    native_qo_indptr=(
                        qo_indptr_prefill_dev if native_prefill else None
                    ),
                    native_kv_lens=(
                        kv_lens_prefill_dev if native_prefill else None
                    ),
                    native_max_q_len=native_max_q_len if native_prefill else 0,
                    splitd_admit=splitd is not None,
                    splitd_q_len=splitd[0] if splitd else 0,
                    splitd_kv_len=splitd[1] if splitd else 0,
                    splitd_bt_row=splitd[2] if splitd else None,
                    splitd_bt=splitd[3] if splitd else None,
                    splitd_k_ws=splitd[4] if splitd else None,
                    splitd_v_ws=splitd[5] if splitd else None,
                    splitd_q_ws=splitd[6] if splitd else None,
                    splitd_out_ws=splitd[7] if splitd else None,
                    fd_verify=fd_verify,
                    fd_kv_lens_dev=(
                        self._sm70_fd_verify_kvlens if fd_verify else None
                    ),
                )

        ## DECODE PATHWAY
        if num_decodes > 0:
            if decode_with_flashinfer_trtllm_api:
                assert self.flashinfer_trtllm_api_decode_kernel is not None
                seq_lens_decode = seq_lens[:num_decodes]
                if self.use_dcp:
                    assert common_attn_metadata.dcp_local_seq_lens is not None
                    seq_lens_decode = common_attn_metadata.dcp_local_seq_lens[
                        :num_decodes
                    ]
                q_len_per_req, q_cu_seq_lens, ragged_q_lens = (
                    self._compute_decode_query_lens(
                        qo_indptr, qo_indptr_cpu, num_decodes, num_decode_tokens
                    )
                )
                decode_mask = None
                if self.use_xqa:
                    decode_mask = self._get_decode_mask(
                        q_len_per_req,
                        ragged_q_lens,
                        num_decodes,
                        bool(causal),
                    )
                elif self.use_trtllm_gen_varlen_decode:
                    # CPU lens are only an upper bound, device qo_indptr is
                    # the truth.
                    decode_q_lens = (
                        qo_indptr_cpu[1 : num_decodes + 1] - qo_indptr_cpu[:num_decodes]
                    )
                    max_q_upper = int(decode_q_lens.max())
                    if max_q_upper > 1:
                        q_len_per_req = max_q_upper
                        q_cu_seq_lens = qo_indptr[: num_decodes + 1]
                attn_metadata.decode = FlashInferTrtllmAPIDecode(
                    kernel=self.flashinfer_trtllm_api_decode_kernel,
                    block_tables=block_table_tensor[:num_decodes],
                    seq_lens=seq_lens_decode,
                    max_seq_len=max_seq_len,
                    q_len_per_req=q_len_per_req,
                    q_cu_seq_lens=q_cu_seq_lens,
                    mask=decode_mask,
                    dcp_query_start_loc=(
                        qo_indptr[: num_decodes + 1] if self.use_dcp else None
                    ),
                )
            else:
                assert seq_lens_cpu is not None or sm70_shadow_decode_nosync
                pure_decode = num_prefills == 0
                use_cudagraph = (
                    self.enable_cuda_graph
                    and pure_decode
                    and num_decode_tokens <= self._decode_cudagraph_max_bs
                )
                num_input_tokens = num_decode_tokens

                decode_wrapper = self._get_decode_wrapper(
                    num_input_tokens, use_cudagraph
                )
                # Use the persistent buffer with padding length,
                # instead of the same address but chunked version
                # in atten_metadata when using cudagraph.
                # The SM100 trtllm-gen NVFP4 kernel only supports FP8
                # output. SM70 (opt-in) and SM12x NVFP4 are serviced by the
                # native FA2 path, which writes model-dtype output directly.
                o_dtype = _nvfp4_o_dtype(
                    self.is_kvcache_nvfp4, self.model_config.dtype
                )
                if self.sm70_nvfp4_shadow:
                    # The in-kernel-dequant flashdecode op consumes only the
                    # per-request kv lens (= seq lens for pure decode) and
                    # its own block table; skip the native FlashInfer paged
                    # staging (per-step pin/copy of indptr and last-page-len,
                    # kv-lens re-derivation) that this path never uses.
                    # Decode requests lead the batch (same order as
                    # block_table_tensor[:num_input_tokens] above).
                    kv_lens_decode_cpu = (
                        None
                        if seq_lens_cpu is None
                        else seq_lens_cpu[:num_input_tokens].to(torch.int32)
                    )
                else:
                    paged_kv_indptr_cpu = self.paged_kv_indptr.cpu[
                        : num_input_tokens + 1
                    ]
                    paged_kv_last_page_len_cpu = self.paged_kv_last_page_len.cpu[
                        :num_input_tokens
                    ]
                    # plan() copies these to the GPU with non_blocking=True;
                    # they are pinned at allocation so the copies stay async.
                    if PIN_MEMORY:
                        paged_kv_indptr_cpu = paged_kv_indptr_cpu.pin_memory()
                        paged_kv_last_page_len_cpu = (
                            paged_kv_last_page_len_cpu.pin_memory()
                        )
                    # plan() also copies kv_lens to the GPU with
                    # non_blocking=True (trtllm-gen/cute-dsl backends),
                    # deriving them via get_seq_lens() into unpinned memory
                    # when seq_lens is not passed; pass an explicit pinned
                    # copy of the same values so that copy stays async too.
                    kv_lens_decode_cpu = get_seq_lens(
                        paged_kv_indptr_cpu,
                        paged_kv_last_page_len_cpu,
                        page_size,
                    )
                    if PIN_MEMORY:
                        kv_lens_decode_cpu = kv_lens_decode_cpu.pin_memory()
                if self.sm70_fp8_shadow:
                    # SM70 FP8: the flashinfer fp8 paged decode kernel is not
                    # sm_70-safe (V100 has no FP8 hardware), so the read side
                    # dequants this layer's paged fp8 cache to an fp16 shadow
                    # buffer (per step, in the forward) and plans the mature
                    # fp16 paged decode against the shadow's identity block
                    # table. The shadow buffer holds the request's full active
                    # context (decode), sized by sm70_shadow_plan_tensors.
                    assert num_input_tokens == num_decodes
                    (
                        plan_indptr,
                        plan_indices,
                        plan_last,
                        shadow_kv_lens,
                        shadow_blocks,
                    ) = sm70_shadow_plan_tensors(
                        kv_lens_decode_cpu, self.page_size
                    )
                    shadow_kv = self._sm70_ensure_shadow_kv(
                        shadow_blocks, self.paged_kv_indptr.gpu.device
                    )
                    fast_plan_decode(
                        decode_wrapper,
                        indptr_cpu=plan_indptr,
                        indices=plan_indices,
                        last_page_len_cpu=plan_last,
                        seq_lens_cpu=shadow_kv_lens,
                        num_qo_heads=self.num_qo_heads * self.dcp_world_size,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        page_size=self.page_size,
                        pos_encoding_mode="NONE",
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type_decode,
                        kv_data_type=torch.float16,
                        o_data_type=o_dtype,
                        fixed_split_size=self.decode_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                    )
                    attn_metadata.decode = FIDecode(
                        wrapper=decode_wrapper,
                        shadow_kv=shadow_kv,
                        shadow_kv_lens=shadow_kv_lens,
                        block_tables=block_table_tensor[:num_input_tokens],
                    )
                elif self.sm70_nvfp4_shadow:
                    # SM70 NVFP4: the in-kernel-dequant split-KV paged decode
                    # (sm70_nvfp4_paged_decode) reads the packed NVFP4 cache
                    # directly — no fp16 shadow buffer and no FA2 decode
                    # plan. Per-step state is only the device kv-lens buffer;
                    # graph replays refresh it via the pinned copy, so the
                    # captured launches stay replay-invariant.
                    assert num_input_tokens == num_decodes
                    if use_cudagraph:
                        if kv_lens_decode_cpu is None:
                            # hostopt: stream-ordered D2D into the static
                            # capture buffer - same value the kernels read,
                            # no CPU round trip, no pipeline drain.
                            self._sm70_kvlens_dev[:num_input_tokens].copy_(
                                common_attn_metadata.seq_lens[
                                    :num_input_tokens
                                ]
                            )
                        else:
                            kv_lens_decode_cpu = kv_lens_decode_cpu.to(
                                torch.int32
                            )
                            self._sm70_kvlens_pinned[:num_input_tokens].copy_(
                                kv_lens_decode_cpu
                            )
                            self._sm70_kvlens_dev[:num_input_tokens].copy_(
                                self._sm70_kvlens_pinned[:num_input_tokens],
                                non_blocking=True,
                            )
                        kvlens_dev = self._sm70_kvlens_dev[:num_input_tokens]
                    else:
                        kvlens_dev = (
                            common_attn_metadata.seq_lens[
                                :num_input_tokens
                            ].to(torch.int32)
                            if kv_lens_decode_cpu is None
                            else kv_lens_decode_cpu.to(
                                self.device, non_blocking=True
                            )
                        )
                    # Fill the SMs: the split kernel's grid is
                    # (num_reqs, num_kv_heads, splits).
                    rh = num_input_tokens * self.num_kv_heads
                    splits = (
                        self._sm70_sm_count * _SM70_FD_SPLIT_FACTOR
                        + rh - 1
                    ) // rh
                    attn_metadata.decode = FIDecode(
                        wrapper=decode_wrapper,
                        shadow_kv_lens=kv_lens_decode_cpu,
                        block_tables=block_table_tensor[:num_input_tokens],
                        shadow_kv_lens_dev=kvlens_dev,
                        shadow_num_reqs=num_input_tokens,
                        shadow_splits=splits,
                        shadow_ws=self._sm70_ws,
                        seq_lens_source=common_attn_metadata.seq_lens,
                    )
                else:
                    fast_plan_decode(
                        decode_wrapper,
                        indptr_cpu=paged_kv_indptr_cpu,
                        indices=paged_kv_indices,
                        last_page_len_cpu=paged_kv_last_page_len_cpu,
                        seq_lens_cpu=kv_lens_decode_cpu,
                        num_qo_heads=self.num_qo_heads * self.dcp_world_size,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        page_size=self.page_size,
                        # Disable flashinfer's pos encoding and use vllm's rope.
                        pos_encoding_mode="NONE",
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type_decode,
                        kv_data_type=self.kv_cache_dtype,
                        o_data_type=o_dtype,
                        fixed_split_size=self.decode_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                    )
                    attn_metadata.decode = FIDecode(wrapper=decode_wrapper)
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        if self.kv_cache_spec.dtype != self.vllm_config.model_config.dtype:
            # TODO: The cascade wrapper currently does not support setting
            # kv cache dtype to something different from query dtype.
            return False
        # TODO: Cascade attention doesn't work, disable it for now
        # return use_cascade_attention(*args, **kwargs)
        return False


class FlashInferImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.window_left = (
            self.sliding_window[0] if self.sliding_window is not None else -1
        )
        self.cache_dtype = kv_cache_dtype
        self.is_kvcache_nvfp4 = kv_cache_dtype.startswith("nvfp4")
        self.sm70_nvfp4_shadow = (
            self.is_kvcache_nvfp4 and sm70_nvfp4_kv_enabled()
        )
        # SM70 FP8 KV-cache: dequant to an fp16 shadow buffer and run the
        # mature fp16 paged kernels (the flashinfer fp8 paged kernels are not
        # sm_70-safe). Parallel to sm70_nvfp4_shadow.
        self.sm70_fp8_shadow = (
            kv_cache_dtype.startswith("fp8") and sm70_flashinfer_attn_enabled()
        )
        # SM70 splitd D256 paged prefill (1Cat port). The builder decides
        # per-step admission (sm70_splitd_admit); this only gates whether
        # the branch exists at all. Gate off by default.
        self.splitd_enabled = (
            envs.VLLM_SM70_SPLITD_PREFILL
            and self.sm70_nvfp4_shadow
            and self.head_size == 256
            and bool(_get_sm70_splitd_op())
        )
        self.kv_cache_dtype = "nvfp4" if self.is_kvcache_nvfp4 else kv_cache_dtype
        self.fp4_data_dim = head_size // 2 if self.is_kvcache_nvfp4 else 0
        # SM100/SM103 NVFP4 KV cache is serviced by the trtllm-gen kernels,
        # which only emit FP8 output and require FP8 queries. SM12x NVFP4 is
        # serviced by the native FA2/XQA paths, which write model-dtype
        # output directly and need no FP8 staging buffer.
        self.trtllm_gen_nvfp4 = (
            self.is_kvcache_nvfp4
            and current_platform.is_device_capability_family(100)
        )
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferImpl"
            )

        self.sinks: torch.Tensor | None = None
        # Keep the source so RL weight updates can refresh the runtime tensor.
        self._sinks_source = sinks
        if sinks is not None:
            if sinks.shape[0] != num_heads:
                raise ValueError(
                    "Sinks must have the same number of heads as the number of "
                    f"heads in the layer. Expected {num_heads}, but got "
                    f"{sinks.shape[0]}."
                )
            self.sinks = sinks

        self.supports_xqa_or_trtllm_gen_decode = can_use_trtllm_attention(
            num_heads, num_kv_heads, is_prefill=False
        )
        vllm_config = get_current_vllm_config_or_none()
        # The layout is resolved after model construction, so read it lazily.
        self.cache_config = vllm_config.cache_config if vllm_config else None
        # SM70 unified NVFP4 FlashDecoding decode/verify kernel
        # (sm70_flashdecode op): fixed-grid split-KV partition + LSE reduce
        # with in-kernel NVFP4 dequant (dim-lane SIMD, 1x KV reads). The
        # authoritative gate for the dispatch in forward(); off by default,
        # op probe last so a missing extension never changes gate-off
        # behavior. DCP and page-size are fail-closed elsewhere (dispatch
        # assert / C++ op check).
        self._sm70_flashdecode = (
            envs.VLLM_SM70_FLASHDECODE
            and self.sm70_nvfp4_shadow
            and self.head_size == 256
            and self.num_kv_heads == 2
            and self.num_heads == 12
            and bool(_get_sm70_flashdecode_op())
        )
        if envs.VLLM_SM70_FLASHDECODE and not self._sm70_flashdecode:
            logger.info(
                "SM70 flashdecode gate on but disabled: shadow=%s "
                "head_size=%s kv_heads=%s qo_heads=%s",
                self.sm70_nvfp4_shadow, self.head_size, self.num_kv_heads,
                self.num_heads,
            )
        if self._sm70_flashdecode:
            # Persistent split-KV partial-state workspace: per call the need
            # is R*H*splits*q_rows*kRatio*(D+2) fp32 with splits(R) =
            # ceil(SM*factor/(R*H)) — worst case is q_rows=8 (verify) at
            # R=1: H*splits*q_rows*kRatio*(D+2); the (SM*factor +
            # max_num_reqs*H) sizing below keeps the q_rows=1 decode margin.
            # kRatio = num_heads / num_kv_heads.
            _max_reqs = (
                vllm_config.scheduler_config.max_num_seqs
                if vllm_config is not None
                else 1
            )
            _k_ratio = self.num_heads // self.num_kv_heads
            self._sm70_fd_ws = torch.empty(
                (
                    torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    ).multi_processor_count
                    * _SM70_FD_SPLIT_FACTOR
                    + _max_reqs * self.num_kv_heads
                    # Reduce stage-1 partials (kSplitChunks=4):
                    # 4*R*H*q_rows*kRatio*(D+2), worst at R=max_reqs.
                    + 4 * _max_reqs * self.num_kv_heads
                )
                * 8  # q_rows up to 8 (spec verify)
                * _k_ratio
                * (self.head_size + 2),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            # Verify (q<=8): R=1 split count (the grid is
            # capture-invariant). The kv-len device buffer is allocated
            # by the metadata builder and carried via FIPrefill.
            # The merged v4 kernel (2..4 q-rows per CTA) halves verify
            # CTAs, so it wants 1.5x the per-row splits: measured
            # splits=120 beats the old 80 by 15% at every ctx
            # (135K: 1.43 vs 1.69 ms/layer-call). 6*SM/(2*H_kv) = 120.
            self._sm70_fd_verify_splits = (
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).multi_processor_count
                * _SM70_FD_VERIFY_FACTOR
                + 2 * self.num_kv_heads
                - 1
            ) // (2 * self.num_kv_heads)
        # Query pre-quantization needs a single dtype for the whole query tensor.
        # SM90 XQA needs BF16/FP16-Q for decode and FP8 for prefill,
        # so only enable this for SM100 trtllm-gen where both use FP8-Q.
        self.supports_quant_query_input = (
            self.supports_xqa_or_trtllm_gen_decode
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and current_platform.is_device_capability_family(100)
            and vllm_config is not None
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        )
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
        self.o_sf_scale: float | None = None

        # Pre-allocated FP8 output buffer for NVFP4 without fused output quant.
        # Only the SM100 trtllm-gen NVFP4 kernels emit FP8 output; the SM12x
        # native paths write model-dtype output directly.
        if self.trtllm_gen_nvfp4 and vllm_config is not None:
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self._nvfp4_fp8_out = torch.empty(
                (max_num_tokens, num_heads, head_size),
                dtype=FP8_DTYPE,
                device="cuda",
            )
        else:
            self._nvfp4_fp8_out = None

        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        if dcp_a2a:
            self.dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self.dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)

    @property
    def kv_cache_layout(self) -> KVCacheLayout:
        assert self.cache_config is not None
        return self.cache_config.get_resolved_kv_cache_layout()

    def fused_output_quant_supported(self, quant_key: QuantKey):
        if quant_key == kNvfp4Dynamic and self.is_kvcache_nvfp4:
            logger.warning_once(
                "fuse_attn_quant: not registering the NVFP4-output attention "
                "fusion with an NVFP4 KV cache (no such trtllm-gen kernel). "
                "Layers whose o_proj takes NVFP4 input keep the unfused path "
                "(FP8 -> BF16 -> o_proj quant); FP8-output fusion is unaffected."
            )
            return False
        # XQA does not support FP8/NVFP4 output, so require trtllm-gen
        # (SM100+) here.  Without that we cannot fuse the output quant.
        return (
            self.supports_xqa_or_trtllm_gen_decode
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and current_platform.is_device_capability_family(100)
            and quant_key in (kFp8StaticTensorSym, kNvfp4Dynamic)
        )

    # FlashInfer requires attention sinks to be float32
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        source_sinks = self._sinks_source
        if source_sinks is None:
            return
        if source_sinks.dtype == torch.float32:
            self.sinks = source_sinks
        elif self.sinks is None or self.sinks.dtype != torch.float32:
            self.sinks = source_sinks.to(torch.float32)
        else:
            self.sinks.copy_(source_sinks)

    def get_xqa_bmm1_scale(self, layer: torch.nn.Module, q_data_type: torch.dtype):
        bmm1_scale = self.scale
        if is_quantized_kv_cache(self.kv_cache_dtype):
            if q_data_type in (torch.float8_e4m3fn, torch.float8_e5m2):
                bmm1_scale *= layer._q_scale_float
            bmm1_scale *= layer._k_scale_float
        return bmm1_scale

    # SM90 may need FP8-Q for native prefill and BF16/FP16-Q for XQA decode,
    # so quantize only the slice whose target dtype differs.
    def maybe_quant_query(
        self,
        query: torch.Tensor,
        q_data_type: torch.dtype,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if query.dtype != q_data_type:
            assert query.dtype in [torch.float16, torch.bfloat16]
            assert q_data_type in [torch.float8_e4m3fn, torch.float8_e5m2]
            assert query.dim() == 3
            num_tokens = query.shape[0]
            num_heads = query.shape[1]
            head_size = query.shape[2]
            assert query.stride(2) == 1 and query.stride(1) == head_size
            query_quantized, _ = custom_ops.scaled_fp8_quant(
                query.view(num_tokens, num_heads * head_size), scale=scale
            )
            return query_quantized.view(num_tokens, num_heads, head_size)

        return query

    def _sm70_splitd_run(
        self,
        prefill_query: torch.Tensor,
        kv_cache: torch.Tensor,
        pre: FIPrefill,
        k_scale: float,
        v_scale: float,
        out: torch.Tensor,
    ) -> None:
        """Run the splitd paged prefill op for the admitted step.

        Stages this layer's NVFP4 KV into the NHD fp16 workspaces (allocated
        in build) and launches the 1Cat splitd kernel once, writing the
        layer output rows directly.
        """
        q_len = pre.splitd_q_len
        kv_len = pre.splitd_kv_len
        H, D = prefill_query.shape[1], prefill_query.shape[2]
        if (
            _SM70_SPLITD_NVFP4_INLINE
            and kv_cache.dtype == torch.uint8
            and bool(_get_sm70_splitd_nvfp4_op())
        ):
            # NVFP4 in-kernel-dequant path: the kernel reads the packed
            # paged cache directly — no staging pass, no fp16 workspaces.
            q_ws = pre.splitd_q_ws
            out_ws = pre.splitd_out_ws
            q_ws[:, :q_len].copy_(prefill_query.view(1, q_len, H, D))
            q_ws[:, q_len:].zero_()
            _get_sm70_splitd_nvfp4_op()(
                q_ws,
                kv_cache,
                pre.splitd_bt_row,
                out_ws,
                kv_len,
                q_len,
                k_scale,
                v_scale,
                self.scale,
                True,
            )
            out.copy_(out_ws[0, :q_len])
            return
        sm70_nvfp4_paged_stage_nhd(
            kv_cache,
            pre.splitd_bt_row,
            kv_len,
            pre.splitd_k_ws,
            pre.splitd_v_ws,
            k_scale,
            v_scale,
        )
        q_ws = pre.splitd_q_ws
        out_ws = pre.splitd_out_ws
        q_ws[:, :q_len].copy_(prefill_query.view(1, q_len, H, D))
        q_ws[:, q_len:].zero_()
        _get_sm70_splitd_op()(
            q_ws,
            pre.splitd_k_ws,
            pre.splitd_v_ws,
            pre.splitd_bt,
            out_ws,
            kv_len,
            q_len,
            self.scale,
            True,
        )
        # out is the layer-output slice [q_len, H, D].
        out.copy_(out_ws[0, :q_len])

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashInfer.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size], or None for a
                KV-sharing decoder layer.
            value: shape = [num_tokens, num_kv_heads, head_size], or None for a
                KV-sharing decoder layer.
            kv_cache: [num_blocks, num_kv_heads, block_size, 2*head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        if (
            os.environ.get("VLLM_SM70_SHADOW_DEBUG") == "1"
            and "mtp" in layer.layer_name
        ):
            try:
                _pd = attn_metadata.prefill
                print(
                    f"[FWDTOP] layer={layer.layer_name} "
                    f"num_prefills={attn_metadata.num_prefills} "
                    f"num_decodes={attn_metadata.num_decodes} "
                    f"n_pre_tok={attn_metadata.num_prefill_tokens} "
                    f"n_dec_tok={attn_metadata.num_decode_tokens} "
                    f"fd_verify={getattr(_pd, 'fd_verify', None)} "
                    f"shadow_kv={'Y' if getattr(_pd, 'shadow_kv', None) is not None else 'N'} "
                    f"q={tuple(query.shape)}",
                    flush=True,
                )
            except Exception as _e:  # pragma: no cover
                print(f"[FWDTOP] err {_e}", flush=True)

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._v_scale_float

        prefill_use_trtllm = isinstance(attn_metadata.prefill, TRTLLMPrefill)
        decode_kernel = (
            attn_metadata.decode.kernel
            if isinstance(attn_metadata.decode, FlashInferTrtllmAPIDecode)
            else None
        )
        decode_with_xqa = decode_kernel == FlashInferDecodeKernel.XQA
        decode_with_trtllm_gen = decode_kernel == FlashInferDecodeKernel.TRTLLM_GEN
        decode_with_flashinfer_trtllm_api = decode_with_xqa or decode_with_trtllm_gen

        # The attn+quant fusion happens when output_scale is provided.
        if output_scale is None:
            assert output_block_scale is None, (
                "output_block_scale is not supported when fusion has not happened"
            )
        else:
            assert attn_metadata.q_data_type_prefill == FP8_DTYPE, (
                "Query must be FP8 when attn+quant fusion happened for prefill."
            )
            assert attn_metadata.q_data_type_decode == FP8_DTYPE, (
                "Query must be FP8 when attn+quant fusion happened for decode."
            )
            assert (attn_metadata.num_prefills == 0 or prefill_use_trtllm) and (
                attn_metadata.num_decodes == 0 or decode_with_trtllm_gen
            ), "Output quant fusion requires TRTLLM prefill/trtllm-gen decode"

            if output.dtype == FP8_DTYPE:
                assert output_block_scale is None, (
                    "output_block_scale should not be provided for fp8 output"
                )
            elif output.dtype == FP4_DTYPE:
                assert output_block_scale is not None, (
                    "output_block_scale is required for nvfp4 output"
                )
            else:
                raise ValueError(f"Unsupported output dtype: {output.dtype}")

            # TRTLLM attn kernel requires to scale to pass as a host scalar,
            # store the o scale as a host scalar in warmup run with cuda graph
            # not enabled
            if layer._o_scale_float is None:
                layer._o_scale_float = output_scale.cpu().item()
                if output.dtype == FP8_DTYPE:
                    self.bmm2_scale = self.bmm2_scale / layer._o_scale_float
                elif output.dtype == FP4_DTYPE:
                    self.o_sf_scale = layer._o_scale_float

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # FlashInfer treats uint8 KV cache as NVFP4. vLLM stores FP8 KV cache
        # as uint8 bytes, so pass FP8 caches with their logical dtype.
        if not self.is_kvcache_nvfp4 and kv_cache.dtype == torch.uint8:
            fp8_view_dtype = None
            if self.kv_cache_dtype in ("fp8", "fp8_e4m3", torch.float8_e4m3fn):
                fp8_view_dtype = torch.float8_e4m3fn
            elif self.kv_cache_dtype in ("fp8_e5m2", torch.float8_e5m2):
                fp8_view_dtype = torch.float8_e5m2
            if fp8_view_dtype is not None:
                kv_cache = kv_cache.view(fp8_view_dtype)

        # Inputs and outputs may be padded for CUDA graphs. Keep the original
        # query for single-token XQA, whose non-ragged API requires one row for
        # every (including padded) decode request.
        query_padded = query
        query = query[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        if attn_metadata.use_cascade:
            # Cascade attention (rare case).
            assert attn_metadata.cascade_wrapper is not None
            stride_order = self.kv_cache_layout.layer_view_order
            if self.is_kvcache_nvfp4:
                kv_cache_views = tuple(
                    cache.permute(*stride_order)
                    for cache in kv_cache.split(self.num_kv_heads, dim=1)
                )
            else:
                kv_perm = kv_cache.permute(*stride_order)
                kv_cache_views = kv_perm.split(self.head_size, dim=-1)
            kv_tuple = tuple(
                canonicalize_singleton_dim_strides(cache) for cache in kv_cache_views
            )
            output.copy_(attn_metadata.cascade_wrapper.run(query, kv_tuple))
            return output

        # When using spec decoding, num_decodes can be < num_decode_tokens
        # because some decode requests may have more than one query token.
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        stride_order = self.kv_cache_layout.layer_view_order
        kv_cache_permute = kv_cache.permute(*stride_order)  # HND and contiguous
        # Fix degenerate strides on any size-1 dimension (e.g. num_kv_heads=1
        # with TP=8).  PyTorch permits non-canonical strides on size-1 dims;
        # CUDA TMA requires ≥16-byte alignment on all non-outermost strides.
        # canonicalize_singleton_dim_strides patches metadata via as_strided —
        # zero-copy.  See vllm.utils.torch_utils.
        fixed = canonicalize_singleton_dim_strides(kv_cache_permute)
        if fixed is not kv_cache_permute:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashInfer): "
                "shape=%s, strides before=%s, strides after=%s",
                kv_cache_permute.shape,
                kv_cache_permute.stride(),
                fixed.stride(),
            )
        kv_cache_permute = fixed

        # Split K/V — zero-copy views. NVFP4 stores K/V as separate head
        # groups; other dtypes pack K/V in the content dim.
        hs = self.head_size
        nvfp4_kv_data = None
        nvfp4_kv_block_scales = None
        if self.is_kvcache_nvfp4:
            k_cache, v_cache = kv_cache.split(self.num_kv_heads, dim=1)
            kv_cache_tuple = (
                canonicalize_singleton_dim_strides(k_cache.permute(*stride_order)),
                canonicalize_singleton_dim_strides(v_cache.permute(*stride_order)),
            )
            k_data, k_sf = nvfp4_split_data_scale(kv_cache_tuple[0])
            v_data, v_sf = nvfp4_split_data_scale(kv_cache_tuple[1])
            nvfp4_kv_data = (k_data, v_data)
            nvfp4_kv_block_scales = (k_sf, v_sf)
        else:
            kv_cache_tuple = kv_cache_permute.split(hs, dim=-1)

        use_dcp = self.dcp_world_size > 1
        if decode_with_xqa:
            assert not use_dcp
            assert self.o_sf_scale is None
            assert output.dtype != FP4_DTYPE

        # Regular attention (common case).
        # Decodes are at the front and prefills are at the back.
        if num_prefill_tokens > 0:
            prefill_query = query[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens

            # Convert query to the expected dtype for prefill if needed.
            prefill_query = self.maybe_quant_query(
                prefill_query,
                attn_metadata.q_data_type_prefill,
                layer._q_scale,
            )

            if not prefill_use_trtllm:
                assert isinstance(attn_metadata.prefill, FIPrefill)
                prefill_wrapper = attn_metadata.prefill.wrapper
                if (
                    self.sm70_nvfp4_shadow
                    and _SM70_NATIVE_PREFILL
                    and attn_metadata.prefill.native_qo_indptr is not None
                ):
                    # SM70 NVFP4 native paged prefill: the flash kernel
                    # with in-kernel dequant reads the packed NVFP4 cache
                    # directly — no fp16 shadow buffer, no FA2 run.
                    assert not use_dcp
                    assert self.sinks is None
                    assert not self.logits_soft_cap
                    assert output.dtype == torch.float16
                    assert prefill_query.dtype == torch.float16
                    assert prefill_query.is_contiguous()
                    assert kv_cache.dtype == torch.uint8
                    sm70_nvfp4_paged_prefill(
                        prefill_query,
                        kv_cache,
                        attn_metadata.prefill.block_tables,
                        attn_metadata.prefill.native_qo_indptr,
                        attn_metadata.prefill.native_kv_lens,
                        output[num_decode_tokens :].view(
                            num_prefill_tokens,
                            prefill_query.shape[1],
                            prefill_query.shape[2],
                        ),
                        layer._k_scale_float,
                        layer._v_scale_float,
                        self.scale,
                        self.window_left,
                        max_q_len=attn_metadata.prefill.native_max_q_len,
                    )
                    return output_padded
                if (
                    self.splitd_enabled
                    and attn_metadata.prefill.splitd_admit
                    and attn_metadata.prefill.splitd_kv_len
                ):
                    # SM70 splitd D256 paged prefill (1Cat port): stage this
                    # layer's NVFP4 KV to NHD fp16 and run the exact-tile
                    # WMMA kernel straight into the output — no FA2 wrapper.
                    assert not use_dcp
                    assert self.sinks is None
                    assert not self.logits_soft_cap
                    assert output.dtype == torch.float16
                    assert prefill_query.dtype == torch.float16
                    assert prefill_query.is_contiguous()
                    assert kv_cache.dtype == torch.uint8
                    self._sm70_splitd_run(
                        prefill_query,
                        kv_cache,
                        attn_metadata.prefill,
                        layer._k_scale_float,
                        layer._v_scale_float,
                        out=output[num_decode_tokens:],
                    )
                    _SPLITD_ROUTE_STATS["routed"] += 1
                    if _SPLITD_ROUTE_STATS["routed"] % 100 == 1:
                        logger.info(
                            "SM70 splitd D256 prefill: %d routed / %d "
                            "fallback layer runs",
                            _SPLITD_ROUTE_STATS["routed"],
                            _SPLITD_ROUTE_STATS["fallback"],
                        )
                    return output_padded
                assert prefill_wrapper is not None
                if use_dcp:
                    if key is None or value is None:
                        raise NotImplementedError(
                            "FlashInfer DCP prefill does not support KV-sharing layers"
                        )
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                    assert prefill_wrapper._context._window_left == self.window_left
                    assert prefill_wrapper._context._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._context._sm_scale == self.scale
                    assert not prefill_wrapper._context._causal
                    assert prefill_wrapper._new_tokens._window_left == self.window_left
                    assert prefill_wrapper._new_tokens._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._new_tokens._sm_scale == self.scale
                    assert prefill_wrapper._new_tokens._causal

                    prefill_wrapper.run(
                        layer,
                        prefill_query,
                        kv_cache_tuple,
                        key[num_decode_tokens:num_actual_tokens],
                        value[num_decode_tokens:num_actual_tokens],
                        out=output[num_decode_tokens:],
                    )
                else:
                    assert isinstance(
                        prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper
                    )
                    assert prefill_wrapper._window_left == self.window_left
                    assert prefill_wrapper._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._sm_scale == self.scale
                    assert prefill_wrapper._causal == attn_metadata.causal

                    if (
                        self.is_kvcache_nvfp4
                        and attn_metadata.prefill.shadow_kv is not None
                    ):
                        if self.splitd_enabled:
                            _SPLITD_ROUTE_STATS["fallback"] += 1
                        # SM70: dequantize this layer's NVFP4 paged cache into
                        # the fp16 shadow buffer and run the mature fp16
                        # paged kernel (the FA2 fp4 reader needs ldmatrix,
                        # sm_75+).
                        shadow_bt = attn_metadata.prefill.block_tables
                        if attn_metadata.prefill.shadow_kv_windowed:
                            # Sliding-window shadow buffer holds only the
                            # pages covering the last window_left tokens;
                            # dequant that tail of the block table (single
                            # request). Block tables are left-aligned with
                            # right padding, so slice the used prefix's
                            # tail — the padded tail is all zeros (block 0).
                            w_blocks = attn_metadata.prefill.shadow_kv.shape[0]
                            start = attn_metadata.prefill.shadow_bt_start
                            shadow_bt = shadow_bt[:, start : start + w_blocks]
                        if os.environ.get("VLLM_SM70_SHADOW_DEBUG") == "1":
                            try:
                                _bt_max = int(shadow_bt.max().item())
                                _bt_min = int(shadow_bt.min().item())
                            except Exception:
                                _bt_max = -999
                                _bt_min = -999
                            print(
                                f"[SHADOWDBG] layer={layer.layer_name} "
                                f"kv_cache={tuple(kv_cache.shape)} page_stride={kv_cache.stride(0)} "
                                f"bt_shape={tuple(shadow_bt.shape)} bt_min={_bt_min} bt_max={_bt_max} "
                                f"kv_lens={attn_metadata.prefill.shadow_kv_lens.tolist() if attn_metadata.prefill.shadow_kv_lens is not None else None} "
                                f"shadow_kv={tuple(attn_metadata.prefill.shadow_kv.shape)} "
                                f"windowed={attn_metadata.prefill.shadow_kv_windowed} "
                                f"kv_N={kv_cache.shape[2] if kv_cache.dim() >= 3 else '?'} "
                                f"n_prefill_tokens={num_prefill_tokens} n_decode_tokens={num_decode_tokens}",
                                flush=True,
                            )
                        sm70_nvfp4_paged_dequant(
                            kv_cache,
                            shadow_bt,
                            attn_metadata.prefill.shadow_kv_lens,
                            attn_metadata.prefill.shadow_kv,
                            layer._k_scale_float,
                            layer._v_scale_float,
                        )
                        shadow_kv = attn_metadata.prefill.shadow_kv
                        kv_cache_for_fi = (
                            shadow_kv[..., : self.head_size],
                            shadow_kv[..., self.head_size :],
                        )
                        kv_cache_sf = None
                        q_scale = k_scale = v_scale = 1.0
                    elif (
                        self.sm70_fp8_shadow
                        and attn_metadata.prefill.shadow_kv is not None
                    ):
                        # SM70 FP8: dequantize this layer's paged fp8 cache
                        # into the fp16 shadow buffer and run the mature fp16
                        # paged kernel (the flashinfer fp8 kernel is not
                        # sm_70-safe).
                        shadow_bt = attn_metadata.prefill.block_tables
                        if attn_metadata.prefill.shadow_kv_windowed:
                            w_blocks = attn_metadata.prefill.shadow_kv.shape[0]
                            start = attn_metadata.prefill.shadow_bt_start
                            shadow_bt = shadow_bt[:, start : start + w_blocks]
                        sm70_fp8_paged_dequant(
                            kv_cache_permute,
                            shadow_bt,
                            attn_metadata.prefill.shadow_kv_lens,
                            attn_metadata.prefill.shadow_kv,
                            layer._k_scale_float,
                            layer._v_scale_float,
                        )
                        shadow_kv = attn_metadata.prefill.shadow_kv
                        kv_cache_for_fi = (
                            shadow_kv[..., : self.head_size],
                            shadow_kv[..., self.head_size :],
                        )
                        kv_cache_sf = None
                        q_scale = k_scale = v_scale = 1.0
                    elif self.is_kvcache_nvfp4:
                        if os.environ.get("VLLM_SM70_SHADOW_DEBUG") == "1":
                            def _safe(_x, _default="?"):
                                try:
                                    return _x
                                except Exception:
                                    return _default
                            try:
                                _bt = attn_metadata.prefill.block_tables
                                _bt_shape = tuple(_bt.shape)
                            except Exception:
                                _bt_shape = "?"
                            try:
                                _sl = attn_metadata.prefill.seq_lens
                                _sl_l = _sl.tolist() if _sl is not None else None
                            except Exception:
                                _sl_l = "?"
                            print(
                                f"[RAWFP4DBG] layer={layer.layer_name} "
                                f"n_prefill={num_prefill_tokens} "
                                f"seq_lens={_sl_l} bt_shape={_safe(_bt_shape)} "
                                f"fd_verify={_safe(attn_metadata.prefill.fd_verify)} "
                                f"splitd_admit={_safe(getattr(attn_metadata.prefill, 'splitd_admit', '?'))} "
                                f"shadow_kv_None={attn_metadata.prefill.shadow_kv is None} "
                                f"q_dtype={prefill_query.dtype} "
                                f"d256_env={envs.VLLM_USE_NVFP4_D256_PREFILL}",
                                flush=True,
                            )
                        kv_cache_for_fi = nvfp4_kv_data
                        kv_cache_sf = nvfp4_kv_block_scales
                        q_scale = layer._q_scale_float
                        k_scale = layer._k_scale_float
                        v_scale = layer._v_scale_float
                    else:
                        kv_cache_for_fi = kv_cache_tuple
                        kv_cache_sf = None
                        q_scale = layer._q_scale_float
                        k_scale = layer._k_scale_float
                        v_scale = layer._v_scale_float

                    # The SM100 trtllm-gen NVFP4 kernel only supports FP8
                    # output. SM12x native FA2 writes model-dtype output
                    # directly, so no staging buffer is used.
                    needs_fp8_out_prefill = (
                        self.trtllm_gen_nvfp4 and output.dtype != FP8_DTYPE
                    )
                    if needs_fp8_out_prefill:
                        out_prefill = self._nvfp4_fp8_out[:num_prefill_tokens]
                    else:
                        out_prefill = output[num_decode_tokens:]

                    if (
                        attn_metadata.prefill.fd_verify
                        and prefill_query.dtype
                        in (torch.float16, torch.bfloat16)
                        and self.sinks is None
                    ):
                        # SM70 flashdecode verify: single request q<=8,
                        # in-kernel NVFP4 dequant, 1x KV reads. Graph-safe:
                        # fixed grid (splits, H, 1), static kv-len buffer,
                        # static workspace. Bypasses the per-step fp16
                        # shadow staging of the full ctx.
                        q = prefill_query.to(torch.float16).contiguous()
                        q_len, HQ_, D_ = q.shape
                        sm70_flashdecode_op(
                            q.reshape(1, q_len, HQ_, D_),
                            kv_cache,
                            attn_metadata.prefill.block_tables,
                            attn_metadata.prefill.fd_kv_lens_dev,
                            out_prefill.reshape(1, q_len, HQ_, D_),
                            layer._k_scale_float,
                            layer._v_scale_float,
                            self.scale,
                            self.window_left,
                            self._sm70_fd_verify_splits,
                            q_len,
                            self._sm70_fd_ws,
                        )
                        if needs_fp8_out_prefill:
                            output[
                                num_decode_tokens
                                : num_decode_tokens + num_prefill_tokens
                            ].copy_(out_prefill)
                        return output_padded

                    if os.environ.get("VLLM_SM70_SHADOW_DEBUG") == "1":
                        try:
                            _wr_type = type(prefill_wrapper).__name__
                        except Exception:
                            _wr_type = "?"
                        print(
                            f"[PFXBR] layer={layer.layer_name} "
                            f"n_prefill={num_prefill_tokens} "
                            f"wrapper={_wr_type} sinks={self.sinks is not None} "
                            f"raw_fp4={kv_cache_sf is not None} "
                            f"fd_verify={attn_metadata.prefill.fd_verify} "
                            f"splitd={attn_metadata.prefill.splitd_admit} "
                            f"d256={bool(envs.VLLM_USE_NVFP4_D256_PREFILL and prefill_query.dtype == torch.bfloat16)} "
                            f"q_dtype={prefill_query.dtype}",
                            flush=True,
                        )
                    if isinstance(
                        prefill_wrapper, BatchAttentionWithAttentionSinkWrapper
                    ):
                        assert self.sinks is not None
                        prefill_wrapper.run(
                            prefill_query,
                            kv_cache_for_fi,
                            self.sinks,
                            self.scale * layer._q_scale_float * layer._k_scale_float,
                            v_scale=layer._v_scale_float,
                            out=out_prefill,
                        )
                    elif (
                        self.is_kvcache_nvfp4
                        and self.head_size == 256
                        and self.window_left == -1
                        and not use_dcp
                        and prefill_query.dtype == torch.bfloat16
                        and envs.VLLM_USE_NVFP4_D256_PREFILL
                    ):
                        _nvfp4_d256_prefill(
                            self,
                            prefill_query,
                            kv_cache,
                            attn_metadata.prefill,
                            (layer._q_scale_float, layer._k_scale_float,
                             layer._v_scale_float),
                            out_prefill,
                            causal=attn_metadata.causal,
                        )
                    else:
                        if os.environ.get("VLLM_SM70_SHADOW_DEBUG") == "1":
                            _sh = attn_metadata.prefill.shadow_kv
                            try:
                                _is_shadow = (
                                    _sh is not None
                                    and kv_cache_for_fi[0].data_ptr()
                                    == _sh[..., : self.head_size].data_ptr()
                                )
                            except Exception:
                                _is_shadow = "?"
                            print(
                                f"[FA2RUN] layer={layer.layer_name} "
                                f"n_prefill={num_prefill_tokens} "
                                f"raw_fp4={kv_cache_sf is not None} "
                                f"is_shadow={_is_shadow} "
                                f"fd_verify={attn_metadata.prefill.fd_verify} "
                                f"splitd={attn_metadata.prefill.splitd_admit} "
                                f"q_dtype={prefill_query.dtype}",
                                flush=True,
                            )
                        prefill_wrapper.run(
                            prefill_query,
                            kv_cache_for_fi,
                            q_scale=q_scale,
                            k_scale=k_scale,
                            v_scale=v_scale,
                            out=out_prefill,
                            kv_cache_sf=kv_cache_sf,
                        )

                    if needs_fp8_out_prefill:
                        output[
                            num_decode_tokens : num_decode_tokens + num_prefill_tokens
                        ].copy_(out_prefill)
            else:
                assert isinstance(attn_metadata.prefill, TRTLLMPrefill)
                # prefill_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                prefill_query = prefill_query.contiguous()
                prefill_query = canonicalize_singleton_dim_strides(prefill_query)
                workspace_buffer = _get_trtllm_workspace_buffer()
                block_tables_prefill = attn_metadata.prefill.block_tables
                seq_lens_prefill = attn_metadata.prefill.seq_lens

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                assert get_flashinfer_layout_string(self.kv_cache_layout) == "HND"
                assert is_strictly_contiguous(prefill_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_prefill)
                assert is_strictly_contiguous(seq_lens_prefill)

                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[num_decode_tokens:],
                        scale=output_block_scale,
                        scale_start_index=num_decode_tokens,
                        original_shape=prefill_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[num_decode_tokens:]

                # The SM100 trtllm-gen NVFP4 kernel only supports FP8 output.
                # SM12x native FA2 writes model-dtype output directly.
                needs_fp8_out = self.trtllm_gen_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_prefill_tokens]

                prefill_kv_block_scales = None
                if self.is_kvcache_nvfp4:
                    # NVFP4 trtllm-gen kernel requires FP8 query.
                    assert attn_metadata.q_data_type_prefill == FP8_DTYPE, (
                        "NVFP4 KV cache requires FP8 quantized queries for "
                        "trtllm-gen prefill. Set "
                        "disable_flashinfer_q_quantization=False."
                    )
                    mock_kv_cache = nvfp4_kv_data
                    mock_block_table = block_tables_prefill
                    prefill_kv_block_scales = nvfp4_kv_block_scales
                elif (
                    attn_metadata.q_data_type_prefill != FP8_DTYPE
                    and self.kv_cache_dtype.startswith("fp8")
                ):
                    # TRTLLM prefill attention does not support BF16 Q
                    # and fp8 kv cache. So to enable prefill attention
                    # with fp8 kv cache, we can construct a mock block
                    # and mock kv cache with BF16 KV involved in the prefill.
                    kv_cache_permute = canonicalize_singleton_dim_strides(
                        kv_cache_permute
                    )
                    kv_strides = kv_cache_permute.stride()
                    assert (
                        kv_strides[-1] == 1
                        and kv_strides[-2] == kv_cache_permute.shape[-1]
                    ), (
                        "KV cache inner dims (block_size, head_size) must be "
                        f"contiguous, got strides {kv_strides}"
                    )
                    # fp8 uses (B, H, N, 2*hs); reshape to (B, 2, H, N, hs)
                    # for the dequant kernel — zero-copy view. The dequant
                    # kernel handles the interleaved K/V block stride.
                    B_kv, H_kv, N_kv = kv_cache_permute.shape[:3]
                    kv_cache_5d = kv_cache_permute.view(B_kv, H_kv, N_kv, 2, hs)
                    kv_cache_5d = kv_cache_5d.permute(0, 3, 1, 2, 4)
                    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
                        kv_cache_5d,
                        block_tables_prefill,
                        layer._k_scale,
                        layer._v_scale,
                        attn_metadata.q_data_type_prefill,
                    )
                else:
                    mock_kv_cache = kv_cache_tuple
                    mock_block_table = block_tables_prefill

                trtllm_batch_context_with_kv_cache(
                    query=prefill_query,
                    kv_cache=mock_kv_cache,
                    workspace_buffer=workspace_buffer,
                    block_tables=mock_block_table,
                    seq_lens=seq_lens_prefill,
                    max_q_len=attn_metadata.prefill.max_q_len,
                    max_kv_len=attn_metadata.prefill.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    batch_size=attn_metadata.num_prefills,
                    cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,
                    cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    kv_cache_sf=prefill_kv_block_scales,
                )

                if needs_fp8_out:
                    output[
                        num_decode_tokens : num_decode_tokens + num_prefill_tokens
                    ].copy_(out[:num_prefill_tokens])

        if num_decode_tokens > 0:
            decode_query_tokens = num_decode_tokens
            if decode_with_xqa:
                assert isinstance(attn_metadata.decode, FlashInferTrtllmAPIDecode)
                if attn_metadata.decode.q_len_per_req == 1:
                    decode_query_tokens = attn_metadata.num_decodes
            decode_query = query_padded[:decode_query_tokens]
            assert decode_query.shape[0] == decode_query_tokens

            # Convert query to the expected dtype for decode if needed.
            decode_query = self.maybe_quant_query(
                decode_query,
                attn_metadata.q_data_type_decode,
                layer._q_scale,
            )

            if not decode_with_flashinfer_trtllm_api:
                assert isinstance(attn_metadata.decode, FIDecode)
                if self._sm70_flashdecode:
                    # SM70 unified FlashDecoding kernel: fixed-grid
                    # split-KV partition + LSE reduce, in-kernel NVFP4
                    # dequant, 1x KV reads. Reuses the per-step shadow
                    # state (kv-lens, splits) of the SM70 NVFP4 decode
                    # path; writes the output in place. Pure q=1 decode
                    # only for now (FIDecode carries no per-request q
                    # lens: total rows == request count iff every
                    # request has exactly one); verify (q>1) is routed
                    # through the varlen prefill path and lands here in
                    # a follow-up.
                    if num_decode_tokens == attn_metadata.num_decodes:
                        assert not use_dcp
                        assert self.sinks is None
                        assert not self.logits_soft_cap
                        assert output.dtype == torch.float16
                        HQ_, D_ = decode_query.shape[1], decode_query.shape[2]
                        sm70_flashdecode_op(
                            decode_query.reshape(
                                num_decode_tokens, 1, HQ_, D_
                            ),
                            kv_cache,
                            attn_metadata.decode.block_tables,
                            attn_metadata.decode.shadow_kv_lens_dev,
                            output[:num_decode_tokens].reshape(
                                num_decode_tokens, 1, HQ_, D_
                            ),
                            layer._k_scale_float,
                            layer._v_scale_float,
                            self.scale,
                            self.window_left,
                            attn_metadata.decode.shadow_splits,
                            1,
                            self._sm70_fd_ws,
                        )
                        return output_padded
                if self.sm70_fp8_shadow:
                    # SM70 FP8: dequant this layer's paged fp8 cache to the
                    # fp16 shadow buffer (built in the metadata) and run the
                    # mature fp16 paged decode (the flashinfer fp8 kernel is
                    # not sm_70-safe).
                    assert not use_dcp
                    assert self.sinks is None
                    assert not self.logits_soft_cap
                    assert attn_metadata.decode.shadow_kv is not None
                    sm70_fp8_paged_dequant(
                        kv_cache_permute,
                        attn_metadata.decode.block_tables,
                        attn_metadata.decode.shadow_kv_lens,
                        attn_metadata.decode.shadow_kv,
                        layer._k_scale_float,
                        layer._v_scale_float,
                    )
                    shadow_kv = attn_metadata.decode.shadow_kv
                    kv_cache_for_fi = (
                        shadow_kv[..., : self.head_size],
                        shadow_kv[..., self.head_size :],
                    )
                    decode_wrapper = attn_metadata.decode.wrapper
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_for_fi,
                        q_scale=1.0,
                        k_scale=1.0,
                        v_scale=1.0,
                        out=output[:num_decode_tokens],
                        kv_cache_sf=None,
                        sinks=self.sinks,
                    )
                    return output_padded
                elif self.sm70_nvfp4_shadow:
                    # SM70 NVFP4: native in-kernel-dequant split-KV paged
                    # decode (no fp16 shadow buffer, no FA2 run).
                    assert not use_dcp
                    assert self.sinks is None
                    assert not self.logits_soft_cap
                    assert output.dtype == torch.float16
                    sm70_nvfp4_paged_decode(
                        decode_query,
                        kv_cache,
                        attn_metadata.decode.block_tables,
                        attn_metadata.decode.shadow_kv_lens_dev,
                        output[:num_decode_tokens],
                        layer._k_scale_float,
                        layer._v_scale_float,
                        self.scale,
                        self.window_left,
                        attn_metadata.decode.shadow_splits,
                        attn_metadata.decode.shadow_ws,
                    )
                    return output_padded
                decode_wrapper = attn_metadata.decode.wrapper
                assert decode_wrapper is not None
                assert decode_wrapper._window_left == self.window_left
                assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)
                assert decode_wrapper._sm_scale == self.scale

                if self.is_kvcache_nvfp4:
                    kv_cache_for_fi = nvfp4_kv_data
                    kv_cache_sf = nvfp4_kv_block_scales
                    q_scale = layer._q_scale_float
                    k_scale = layer._k_scale_float
                    v_scale = layer._v_scale_float
                else:
                    kv_cache_for_fi = kv_cache_tuple
                    kv_cache_sf = None
                    q_scale = layer._q_scale_float
                    k_scale = layer._k_scale_float
                    v_scale = layer._v_scale_float

                # The SM100 trtllm-gen NVFP4 kernel only supports FP8 output.
                # SM12x native FA2 writes model-dtype output directly.
                needs_fp8_out = self.trtllm_gen_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out_decode = self._nvfp4_fp8_out[:num_decode_tokens]
                else:
                    out_decode = output[:num_decode_tokens]

                if use_dcp:
                    decode_query = get_dcp_group().all_gather(
                        decode_query.contiguous(), dim=-2
                    )
                    output_tmp = torch.empty_like(decode_query)
                    lse = torch.empty(
                        (decode_query.size(0), decode_query.size(1)),
                        dtype=torch.float32,
                        device=decode_query.device,
                    )
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_for_fi,
                        q_scale=q_scale,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        out=output_tmp,
                        lse=lse,
                        return_lse=True,
                        kv_cache_sf=kv_cache_sf,
                        sinks=self.sinks,
                    )
                    output[:num_decode_tokens] = self.dcp_combine(
                        output_tmp,
                        lse,
                        get_dcp_group(),
                    )
                else:
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_for_fi,
                        q_scale=q_scale,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        out=out_decode,
                        kv_cache_sf=kv_cache_sf,
                        sinks=self.sinks,
                    )

                if needs_fp8_out:
                    output[:num_decode_tokens].copy_(out_decode)
            else:
                assert isinstance(attn_metadata.decode, FlashInferTrtllmAPIDecode)
                # decode_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                decode_query = decode_query.contiguous()
                decode_query = canonicalize_singleton_dim_strides(decode_query)
                workspace_buffer = _get_trtllm_workspace_buffer()
                block_tables_decode = attn_metadata.decode.block_tables
                seq_lens_decode = attn_metadata.decode.seq_lens

                # trtllm-gen needs HND layout on SM100. XQA is selected
                # separately on SM90 and does not use this SM100 layout gate.
                if decode_with_trtllm_gen:
                    assert get_flashinfer_layout_string(self.kv_cache_layout) == "HND"
                else:
                    assert decode_with_xqa
                assert is_strictly_contiguous(decode_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_decode)
                assert is_strictly_contiguous(seq_lens_decode)
                kv_cache_permute = canonicalize_singleton_dim_strides(kv_cache_permute)
                kv_strides = kv_cache_permute.stride()
                assert (
                    kv_strides[-1] == 1 and kv_strides[-2] == kv_cache_permute.shape[-1]
                ), (
                    "KV cache inner dims (block_size, head_size) must be "
                    f"contiguous, got strides {kv_strides}"
                )

                if use_dcp:
                    assert decode_with_trtllm_gen
                    if output.dtype == FP4_DTYPE:
                        raise NotImplementedError(
                            "DCP decode with FlashInfer trtllm-gen does not support "
                            "FP4 attention output yet."
                        )
                    decode_query = get_dcp_group().all_gather(
                        decode_query.contiguous(), dim=-2
                    )
                    decode_query = canonicalize_singleton_dim_strides(decode_query)

                if decode_with_xqa:
                    bmm1_scale = self.get_xqa_bmm1_scale(
                        layer, attn_metadata.q_data_type_decode
                    )
                    q_len_per_req: int | None = attn_metadata.decode.q_len_per_req

                    flashinfer_xqa_batch_decode_with_kv_cache(
                        query=decode_query,
                        kv_cache=(
                            nvfp4_kv_data if self.is_kvcache_nvfp4 else kv_cache_tuple
                        ),
                        kv_cache_sf=nvfp4_kv_block_scales
                        if self.is_kvcache_nvfp4
                        else None,
                        workspace_buffer=workspace_buffer,
                        block_tables=block_tables_decode,
                        seq_lens=seq_lens_decode,
                        max_seq_len=attn_metadata.decode.max_seq_len,
                        bmm1_scale=bmm1_scale,
                        bmm2_scale=self.bmm2_scale,
                        window_left=self.window_left,
                        out=output_padded[:decode_query_tokens],
                        sinks=self.sinks,
                        kv_layout=get_flashinfer_layout_string(self.kv_cache_layout),
                        q_len_per_req=q_len_per_req,
                        mask=attn_metadata.decode.mask,
                        q_cu_seq_lens=attn_metadata.decode.q_cu_seq_lens,
                    )
                    return output_padded

                assert decode_with_trtllm_gen
                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[:num_decode_tokens],
                        scale=output_block_scale,
                        scale_start_index=0,
                        original_shape=decode_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[:num_decode_tokens]

                # The SM100 trtllm-gen NVFP4 kernel only supports FP8 output.
                # SM12x native FA2/XQA writes model-dtype output directly.
                needs_fp8_out = self.trtllm_gen_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_decode_tokens]

                q_cu_seq_lens = attn_metadata.decode.q_cu_seq_lens
                max_q_len = None
                q_len_per_req = None
                if q_cu_seq_lens is not None:
                    # q_len_per_req must stay None for the kernel's varlen path.
                    max_q_len = attn_metadata.decode.q_len_per_req
                elif num_decode_tokens % attn_metadata.num_decodes != 0:
                    # This gets triggered when the dummy_run forces
                    # attention to be initialized with q_len = 0
                    q_len_per_req = 1
                else:
                    q_len_per_req = num_decode_tokens // attn_metadata.num_decodes

                lse = None
                if use_dcp:
                    out = torch.empty(
                        decode_query.shape,
                        dtype=output.dtype,
                        device=decode_query.device,
                    )
                    lse = torch.empty(
                        (decode_query.size(0), decode_query.size(1)),
                        dtype=torch.float32,
                        device=decode_query.device,
                    )

                trtllm_batch_decode_with_kv_cache(
                    query=decode_query,
                    kv_cache=(
                        nvfp4_kv_data if self.is_kvcache_nvfp4 else kv_cache_tuple
                    ),
                    workspace_buffer=workspace_buffer,
                    block_tables=block_tables_decode,
                    seq_lens=seq_lens_decode,
                    max_seq_len=attn_metadata.decode.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    kv_layout=get_flashinfer_layout_string(self.kv_cache_layout),
                    backend=attn_metadata.decode.kernel.value,
                    q_len_per_req=q_len_per_req,
                    max_q_len=max_q_len,
                    cum_seq_lens_q=q_cu_seq_lens,
                    kv_cache_sf=(
                        nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None
                    ),
                    lse=lse,
                    return_lse=self.need_to_return_lse_for_decode,
                )

                if use_dcp:
                    assert isinstance(out, torch.Tensor)
                    assert lse is not None
                    output[:num_decode_tokens] = self.dcp_combine(
                        out,
                        lse,
                        get_dcp_group(),
                        seq_lens=seq_lens_decode,
                        query_start_loc=attn_metadata.decode.dcp_query_start_loc,
                    )
                elif needs_fp8_out:
                    output[:num_decode_tokens].copy_(out)
        return output_padded

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            if self.is_kvcache_nvfp4:
                # (B, 2*H, N, full_dim) -> ((B, N, H, full_dim),
                #                            (B, N, H, full_dim));
                # K heads first, then V heads.
                k_cache, v_cache = kv_cache.transpose(1, 2).split(
                    self.num_kv_heads, dim=-2
                )
            else:
                # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
                k_cache, v_cache = kv_cache.transpose(1, 2).split(
                    self.head_size, dim=-1
                )
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                self.cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )


def fast_plan_decode(
    self,  # decode wrapper
    indptr_cpu: torch.Tensor,
    indices: torch.Tensor,
    last_page_len_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: float | None = None,
    q_data_type: str | torch.dtype | None = "float16",
    kv_data_type: str | torch.dtype | None = None,
    o_data_type: str | torch.dtype | None = None,
    data_type: str | torch.dtype | None = None,
    sm_scale: float | None = None,
    rope_scale: float | None = None,
    rope_theta: float | None = None,
    non_blocking: bool = True,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
) -> None:
    """
    A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for
    cudagraph capture/replay, while the no cudagraph version turns back
    to the original plan.
    using original plan after passing host-side buffers:
    - only host-to-device copy of indptr and last_page_len buffers
    Modifications for cudagraph:
    - only host-to-device copy of indptr and last_page_len buffers.
    - avoid device-to-device copy of indices buffer.

    Part of the code get inspiration from the original plan from FlashInfer repo
    and the implementation of fast_decode_plan for FlashInfer in SGlang repo.
    """
    # Warm up with the original plan if it is first call, and always run the
    # original plan if we run for dynamic shape. For fixed shape (cudagraph),
    # this warm up is to generate the _cached_module for the decode wrapper.
    if not self.is_cuda_graph_enabled or getattr(self, "vllm_first_call", True):
        # plan() copies qo_indptr from flashinfer's internal cached host
        # arange buffer (_get_range_buf) with non_blocking=True; pin the
        # cache entry so the copy stays async.
        pin_host_range_buf(len(last_page_len_cpu) + 1)
        self.plan(
            indptr=indptr_cpu,
            indices=indices,
            last_page_len=last_page_len_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            block_tables=None,
            seq_lens=seq_lens_cpu,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self.vllm_first_call = False
        return

    assert self.is_cuda_graph_enabled, "Should be cudagraph only here"

    fast_decode_plan(
        self,
        indptr=indptr_cpu,
        indices=indices,
        last_page_len=last_page_len_cpu,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        data_type=data_type,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        non_blocking=non_blocking,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )


@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )
