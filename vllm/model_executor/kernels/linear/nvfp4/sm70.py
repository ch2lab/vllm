# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sm70_nvfp4_ops import dequant, is_enabled

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


def _sm70_dequant_scaled(layer, k, n):
    """Dequant the NVFP4 weight to FP16 with the per-tensor global scale
    baked in. The layer's weight_global_scale is a MULTIPLIER (the reciprocal
    of the checkpoint's dequant divisor, e.g. 1/6171.6 = 1.6e-4), so the true
    weight = (fp4 x group_scale) * weight_global_scale. Multiplying (not
    dividing) yields the true weight magnitude (~0.4) instead of the
    pre-normalization magnitude (~2688), keeping the FP16 GEMM in range."""
    return dequant(layer.weight, layer.weight_scale,
                   layer.weight_global_scale)


class Sm70MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """Weight-only NVFP4 GEMM on SM70 (V100/Volta): dequant -> FP16.

    The Marlin NVFP4 GEMM needs a macro-N swizzled layout the checkpoint does
    not store, so the GEMM runs as a plain FP16 GEMM over an on-the-fly
    dequant of the checkpoint's natural row-major NVFP4 layout (the fused
    sm70_nvfp4_dequant kernel bakes in the per-tensor global scale). Small
    layers (output dim not a multiple of 64, e.g. the GDN in_proj_ba with
    n=96 or the standalone n=48 gating projections) have their dequant cached
    once at load; the large layers dequant on every forward (caching a 27B
    model's FP16 weights would OOM a 32 GB V100). Only supported on exact SM70
    with the VLLM_SM70_NVFP4 env switch and a registered SM70 dequant op;
    otherwise it fails closed and the upstream kernel selection is left
    untouched.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is None:
            _cc = current_platform.get_device_capability()
            compute_capability = _cc.to_int() if _cc is not None else None
        if compute_capability != 70:
            return False, "SM70 NVFP4 requires SM70 (7.0)"
        if not is_enabled():
            return False, "VLLM_SM70_NVFP4 not enabled or SM70 op unavailable"
        return True, None

    @classmethod
    def can_implement(
        cls, config: NvFp4LinearLayerConfig
    ) -> tuple[bool, str | None]:
        # Handles all n: every layer runs dequant -> FP16 GEMM; small
        # (n % 64 != 0) layers cache the dequant at load, the rest dequant
        # on-the-fly each forward.
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for attr in ("weight", "weight_scale", "weight_global_scale"):
            if not hasattr(layer, attr):
                raise AttributeError(
                    f"SM70 NVFP4 kernel requires layer.{attr} to be present"
                )

        n = layer.output_size_per_partition
        k = layer.input_size_per_partition
        if n % 64 != 0:
            # The weight is small (n % 64 != 0, e.g. n=96 / n=48), so dequant
            # it once to a dense FP16 weight and cache it; each forward then
            # runs a plain FP16 GEMM.
            W_fp16 = _sm70_dequant_scaled(layer, k, n)
            layer._sm70_nvfp4_fp16_weight = W_fp16
            layer._sm70_nvfp4_fp16_k = k
            layer._sm70_nvfp4_fp16_n = n
            logger.info_once(
                "SM70 NVFP4: output dim %d is not a multiple of 64; using the "
                "FP16 dequant fallback (k=%d).",
                n,
                k,
            )
        else:
            logger.info_once(
                "Using SM70 NVFP4 (dequant -> FP16 GEMM); weight layout read "
                "directly from the checkpoint."
            )

    def apply_weights(
        self,
        layer: torch.Tensor | torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k = layer.input_size_per_partition
        n = layer.output_size_per_partition
        # The SM70 GEMM is FP16-internal; capture the caller's dtype (FP16 or
        # BF16) and cast the result back so the output matches the model dtype.
        out_dtype = x.dtype
        a = x.reshape(-1, k).to(torch.float16)

        # SM70 NVFP4: dequant to FP16 + standard FP16 GEMM.
        #
        # Correct dequant: w_true = (fp4_codes x group_scale) x weight_global_scale
        # (the layer stores weight_global_scale as a multiplier, the reciprocal
        # of the checkpoint's dequant divisor). It is baked into the dequantized
        # FP16 weight (_sm70_dequant_scaled) so the FP16 GEMM runs on the true
        # weight magnitude (~0.4), not the pre-normalization magnitude (~2688)
        # which would overflow FP16 on activation outliers.
        #
        # Small layers (n%64!=0) have their dequant cached at load time.
        # Large layers (n%64==0) dequant on-the-fly (not cached, to avoid
        # OOM — 27B model in FP16 = 54 GB > 32 GB V100).
        W_fp16 = getattr(layer, "_sm70_nvfp4_fp16_weight", None)
        if W_fp16 is None:
            W_fp16 = _sm70_dequant_scaled(layer, k, n)
            # Only cache small weights (already pre-cached by
            # process_weights_after_loading for n%64!=0). For large layers,
            # W_fp16 is a temporary that will be freed after this call.
            if n % 64 != 0:
                layer._sm70_nvfp4_fp16_weight = W_fp16

        out = F.linear(a, W_fp16).to(torch.float32)
        if bias is not None:
            out = out + bias.to(torch.float32)

        return out.to(out_dtype).reshape(*x.shape[:-1], n)
