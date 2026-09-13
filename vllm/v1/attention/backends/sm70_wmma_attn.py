# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 (Volta V100) WMMA attention backend.

Uses WMMA tensor-core paged kernels for prefill and (with FP8 KV cache)
decode. FP16 decode falls back to Triton. Subclasses TritonAttentionBackend
to reuse its metadata builder and KV cache layout.
"""

from typing import ClassVar

import os as _os
import torch
from dataclasses import replace

from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim
from vllm.v1.attention.backends.sm70_nvfp4_kv import (
    nvfp4_write as nvfp4_write_fused,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

_PROF = _os.environ.get("VLLM_SM70_PROFILE", "0") == "1"
_PROF_ATT = {"ms": 0.0, "count": 0}
_PROF_ATT_PAIRS = []
_PROF_ATT_TAG = {"name": ""}


class SM70WMMAAttentionBackend(TritonAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    ]

    @staticmethod
    def get_name() -> str:
        return "SM70_WMMA_ATTN"

    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        """NVFP4 stores K and V as separate per-head slots of packed fp4
        data plus fp8 block scales (same convention as FlashInfer)."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
            return spec
        hs = nvfp4_kv_cache_full_dim(spec.head_size)
        assert hs == nvfp4_kv_cache_full_dim(spec.head_size_v), (
            "nvfp4 with asymmetric K/V head sizes not yet supported"
        )
        return replace(
            spec,
            num_head_slots=2 * spec.num_kv_heads,
            state_content_bytes=hs,
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The custom WMMA kernels address heads/states directly and need the
        # per-layer [B, H, N, C] identity layout (LBHNC / "HND").
        return (KVCacheLayout.LBHNC,)

    @staticmethod
    def get_impl_cls() -> type["SM70WMMAAttentionImpl"]:
        return SM70WMMAAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "nvfp4":
            # K/V as separate head rows; slot = [e2m1 data (hs/2 B) |
            # fp8 block scales (hs/16 B)] — same layout as FlashInfer NVFP4.
            full_dim = head_size // 2 + head_size // 16
            return (num_blocks, 2 * num_kv_heads, block_size, full_dim)
        return TritonAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # Head-major contiguous pages: required by the nvfp4 write kernel's
        # HND path and by our WMMA kernel's region addressing.
        if include_num_layers_dimension:
            return (0, 1, 3, 2, 4)
        return (0, 1, 2, 3)

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 7

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (128, 256)

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False


class SM70WMMAAttentionImpl(TritonAttentionImpl):

    def __init__(self, *args, **kwargs):
        # kv_cache_dtype is the 7th positional arg (index 6) in the parent sig
        if "kv_cache_dtype" in kwargs:
            kv_cache_dtype = kwargs["kv_cache_dtype"]
        elif len(args) > 6:
            kv_cache_dtype = args[6]
        else:
            kv_cache_dtype = "auto"

        self._sm70_fp8_kv = str(kv_cache_dtype).startswith("fp8")
        self._sm70_fp8_e5m2 = str(kv_cache_dtype) == "fp8_e5m2"
        self._sm70_nvfp4_kv = str(kv_cache_dtype) == "nvfp4"
        if self._sm70_fp8_kv or self._sm70_nvfp4_kv:
            # Bypass Triton's quant-KV gates — our kernel dequants inline
            if "kv_cache_dtype" in kwargs:
                kwargs["kv_cache_dtype"] = "auto"
            else:
                args = list(args)
                args[6] = "auto"
            super().__init__(*args, **kwargs)
            self.kv_cache_dtype = kv_cache_dtype
        else:
            super().__init__(*args, **kwargs)
        # The WMMA kernel takes FP16 Q and dequants FP8 KV inline; do not
        # let the attention layer pre-quantize Q to FP8.
        self.supports_quant_query_input = False
        # Created at init (outside CUDA graph capture): the lazy path would
        # issue a CPU->CUDA copy during capture and abort graph capture.
        self._nvfp4_levels = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda"
        )

    def _nvfp4_write(self, layer, key, value, kv_cache, slot_mapping):
        """Quantize K/V to nvfp4 and scatter into the paged cache.

        Fused Triton path (``sm70_nvfp4_kv.nvfp4_write``); the pure-PyTorch
        port of the upstream CUDA store kernel burned ~70% of the decode
        worker's CPU in the 55-op/step path and was replaced by this.
        """
        num_tokens = key.shape[0]
        if num_tokens == 0:
            return
        if not torch.cuda.is_current_stream_capturing():
            valid = slot_mapping >= 0
            slot_mapping = slot_mapping[valid]
            key = key[valid]
            value = value[valid]
            num_tokens = key.shape[0]
            if num_tokens == 0:
                return
        else:
            # Capture placeholders may carry out-of-range slot ids; clamp so
            # the scattered values stay in-bounds (results are discarded).
            slot_mapping = slot_mapping.clamp(
                max=kv_cache.shape[0] * kv_cache.shape[2] - 1)

        hs = self.head_size
        DD = hs // 2
        SD = hs // 16
        num_kv_heads = key.shape[1]
        BS = kv_cache.shape[2]
        page = kv_cache.stride(0)
        dev = key.device
        levels = self._nvfp4_levels

        def quant(x, gs):
            xf = x.float().view(num_tokens, num_kv_heads, hs // 16, 16)
            amax = xf.abs().amax(-1, keepdim=True)
            sf = (amax / (6.0 * gs)).to(torch.float8_e4m3fn)
            y = xf / (sf.float() * gs).clamp_min(1e-30)
            idx = (y.abs().unsqueeze(-1) - levels).abs().argmin(-1)
            nib = idx.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
            nib = nib.view(num_tokens, num_kv_heads, hs)
            packed = nib[..., 0::2] | (nib[..., 1::2] << 4)
            return packed, sf.view(torch.uint8).squeeze(-1)

        # Host scalar copies avoid a device sync during CUDA graph capture.
        # _*_scale_float is a plain Python float set on every init path;
        # _*_scale_cpu may live on device when created under the CUDA
        # default-device context, so prefer the float attribute.
        gs_k = float(getattr(layer, "_k_scale_float",
                             getattr(layer, "_k_scale_cpu", layer._k_scale)))
        gs_v = float(getattr(layer, "_v_scale_float",
                             getattr(layer, "_v_scale_cpu", layer._v_scale)))
        kb, ksf = quant(key, gs_k)
        vb, vsf = quant(value, gs_v)

        flat = kv_cache.reshape(-1)
        blk = slot_mapping // BS
        off = slot_mapping % BS
        hidx = torch.arange(num_kv_heads, device=dev)
        C = DD + SD
        tok = blk[:, None, None] * page + (off * C)[:, None, None]
        h = hidx[None, :, None]

        dpos = torch.arange(DD, device=dev)
        data_idx = tok + h * (BS * C) + dpos[None, None, :]
        spos = torch.arange(SD, device=dev)
        scale_idx = tok + h * (BS * C) + DD + spos[None, None, :]

        flat[data_idx] = kb
        flat[scale_idx] = ksf
        v_side = num_kv_heads * BS * C
        flat[data_idx + v_side] = vb
        flat[scale_idx + v_side] = vsf

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self._sm70_nvfp4_kv:
            nvfp4_write_fused(layer, key, value, kv_cache, slot_mapping)
            return

        if not self._sm70_fp8_kv:
            return super().do_kv_cache_update(
                layer, key, value, kv_cache, slot_mapping
            )

        num_tokens = key.shape[0]
        if num_tokens == 0:
            return

        # Skip padded slots (-1) present in eager batches; negative indices
        # would wrap around and corrupt real blocks. During CUDA graph
        # capture vLLM pads with dummy requests that own real slots, and
        # boolean masking is not capturable anyway.
        if not torch.cuda.is_current_stream_capturing():
            valid = slot_mapping >= 0
            slot_mapping = slot_mapping[valid]
            key = key[valid]
            value = value[valid]
            if slot_mapping.shape[0] == 0:
                return

        k_scale = layer._k_scale
        v_scale = layer._v_scale

        head_size = self.head_size
        # kv_cache logical layout: [num_blocks, num_kv_heads, block_size,
        # 2*head_size]; physical layout may be NHD (non-contiguous), but
        # indexing through the logical view handles strides. Write through a
        # uint8 view: assigning uint8 into a Float8 tensor would numerically
        # cast the values instead of copying raw bytes.
        if kv_cache.dtype != torch.uint8:
            kv_cache = kv_cache.view(torch.uint8)
        key_cache = kv_cache[:, :, :, :head_size]
        value_cache = kv_cache[:, :, :, head_size:]

        # Quantize: fp16 / scale -> clamp -> fp8 -> view as uint8
        if self._sm70_fp8_e5m2:
            fp8_dtype, lim = torch.float8_e5m2, 57344.0
        else:
            fp8_dtype, lim = torch.float8_e4m3fn, 448.0
        key_fp8 = (key.float() / k_scale).clamp(-lim, lim).to(
            fp8_dtype
        ).view(torch.uint8)
        value_fp8 = (value.float() / v_scale).clamp(-lim, lim).to(
            fp8_dtype
        ).view(torch.uint8)

        # Scatter into paged cache
        block_size = kv_cache.shape[2]
        block_indices = slot_mapping // block_size
        slot_offsets = slot_mapping % block_size

        key_cache[block_indices, :, slot_offsets, :] = key_fp8
        value_cache[block_indices, :, slot_offsets, :] = value_fp8

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)
        if _PROF and not torch.cuda.is_current_stream_capturing():
            _start = torch.cuda.Event(enable_timing=True)
            _start.record()
            _PROF_ATT_TAG["name"] = (
                f"prefill_mq={attn_metadata.max_query_len} "
                f"ml={attn_metadata.max_seq_len}"
                if attn_metadata.max_query_len > 1
                else f"decode ml={attn_metadata.max_seq_len}"
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        max_query_len = attn_metadata.max_query_len

        quant_kv = self._sm70_fp8_kv or self._sm70_nvfp4_kv
        use_wmma = (
            self.head_size in (128, 256)
            and attn_metadata.causal is True
            and (quant_kv or max_query_len > 1)
            and (quant_kv or not torch.cuda.is_current_stream_capturing())
        )

        if not use_wmma:
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        q = query[:num_actual_tokens]
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        num_seqs = query_start_loc.shape[0] - 1
        out = output[:num_actual_tokens]

        # Small-query prefill (chunked prefill / partial prefill) scans the
        # full KV span through the general-purpose kernel, which is ~6x
        # slower than the partition-scan decode kernel for the same KV.
        # Route max_query_len <= BM through the decode kernel instead.
        # NOTE: num_partitions is a kernel launch constant captured by CUDA
        # graphs. Deriving it from the runtime max_seq_len makes a graph
        # captured with a small dummy seq_len replay with num_partitions=1,
        # collapsing long-context decode to a full-KV scan (13.6K ctx
        # measured at 421ms/step vs 24ms with partitions). Always use the
        # full split so capture and replay agree; empty partitions
        # write max=-inf which the log-sum-exp merge zero-weights.
        def _mk_partitions(max_seq_len):
            return int(_os.environ.get("VLLM_SM70_ATT_PARTITIONS", "64"))

        # Routing small multi-token queries (MTP verify) through the
        # partitioned decode kernel. Verified coherent end-to-end and +53%
        # long-context decode (8.7 -> 13.3 tok/s @15.4k); set to "0" to use
        # the general-purpose scan instead.
        _mq_decode = _os.environ.get("VLLM_SM70_ATT_MQ_DECODE", "1") == "1"

        if quant_kv:
            # Batched paged WMMA kernel. Reads query_start_loc / seq_lens /
            # block_table on-device, so it is CUDA-graph safe. The kernel
            # reads raw bytes, so view as uint8 (1 byte/elem).
            kv_cache_u8 = (
                kv_cache.view(torch.uint8)
                if kv_cache.dtype != torch.uint8
                else kv_cache
            )
            if self._sm70_nvfp4_kv:
                # NVFP4 layout: K slots = first num_kv_heads rows, V slots =
                # the following num_kv_heads rows in the same tensor; the
                # kernel reaches V at +num_kv_heads*stride_head.
                kv_mode = 3
            elif self._sm70_fp8_e5m2:
                kv_mode = 4
            else:
                kv_mode = 1
            # Host copies of the scales avoid device syncs during capture.
            k_scale = float(getattr(layer, "_k_scale_float",
                                    getattr(layer, "_k_scale_cpu", layer._k_scale)))
            v_scale = float(getattr(layer, "_v_scale_float",
                                    getattr(layer, "_v_scale_cpu", layer._v_scale)))
            if max_query_len == 1:
                # Parallel KV-partition decode: splits the KV span across
                # blocks and merges via log-sum-exp, avoiding the O(KV)
                # single-block scan that dominates long-context decode.
                max_seq_len = attn_metadata.max_seq_len
                # Cap partitions at the kernel's MAX_PARTITIONS buffer.
                num_partitions = _mk_partitions(max_seq_len)
                O = torch.ops._C.flash_attn_sm70_decode_partitioned(
                    q, kv_cache_u8, block_table, query_start_loc, seq_lens,
                    max_query_len, num_seqs, num_partitions, self.scale,
                    k_scale, v_scale, kv_mode, 1,
                )
            elif _mq_decode and max_query_len <= 16:
                num_partitions = _mk_partitions(attn_metadata.max_seq_len)
                O = torch.ops._C.flash_attn_sm70_decode_partitioned(
                    q, kv_cache_u8, block_table, query_start_loc, seq_lens,
                    max_query_len, num_seqs, num_partitions, self.scale,
                    k_scale, v_scale, kv_mode, 0,
                )
            else:
                O = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
                    q, kv_cache_u8, block_table, query_start_loc, seq_lens,
                    num_seqs, max_query_len, self.scale,
                    True, k_scale, v_scale, kv_mode,
                )
            out.copy_(O)
        else:
            # FP16 path: batched paged WMMA kernel (handles all sequences
            # in a single launch, reads indices on-device).
            if _mq_decode and max_query_len <= 16:
                num_partitions = _mk_partitions(attn_metadata.max_seq_len)
                O = torch.ops._C.flash_attn_sm70_decode_partitioned(
                    q, kv_cache, block_table, query_start_loc, seq_lens,
                    max_query_len, num_seqs, num_partitions, self.scale,
                    1.0, 1.0, 0, 0,
                )
            else:
                O = torch.ops._C.flash_attn_sm70_prefill_paged_batched(
                    q, kv_cache, block_table, query_start_loc, seq_lens,
                    num_seqs, max_query_len, self.scale,
                    True, 1.0, 1.0, 0,
                )
            out.copy_(O)

        if _PROF and not torch.cuda.is_current_stream_capturing():
            _end = torch.cuda.Event(enable_timing=True)
            _end.record()
            _PROF_ATT_PAIRS.append((_start, _end, _PROF_ATT_TAG["name"]))
            self._maybe_log_att_profile()
        return output

    def _maybe_log_att_profile(self):
        if not _PROF:
            return
        if len(_PROF_ATT_PAIRS) >= 100:
            torch.cuda.synchronize()
            total = sum(s.elapsed_time(e) for s, e, _ in _PROF_ATT_PAIRS)
            _PROF_ATT["ms"] += total
            _PROF_ATT["count"] += len(_PROF_ATT_PAIRS)
            pairs = _PROF_ATT_PAIRS.copy()
            _PROF_ATT_PAIRS.clear()
            from collections import Counter

            tags = Counter(t for _, _, t in pairs)
            tag_str = ", ".join(f"{t}:{c}" for t, c in tags.most_common(3))
            print(
                f"[gdn-profile] attention: {len(pairs)} calls, "
                f"{total:.1f} ms (avg {total / len(pairs):.3f} ms), "
                f"cumulative {_PROF_ATT['ms']:.1f} ms / {_PROF_ATT['count']}; "
                f"tags: {tag_str}",
                flush=True,
            )
