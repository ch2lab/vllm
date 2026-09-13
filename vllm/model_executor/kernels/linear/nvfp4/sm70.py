# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sm70_nvfp4_ops import gemm, is_enabled

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

# E2M1 (FP4) code (0..15) -> value. Sign is the high bit; the 2-bit exponent /
# 1-bit mantissa (bias 1) give magnitudes (00:0,0.5 / 01:1,1.5 / 10:2,3 /
# 11:4,6).
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequant_nvfp4_to_fp16(
    weight: torch.Tensor, weight_scale: torch.Tensor, k: int, n: int
) -> torch.Tensor:
    """Dequant the checkpoint's natural NVFP4 weight to a dense FP16 weight.

    For output dims that are not a multiple of 64 the SM70 Marlin GEMM cannot
    run (its packed macro is 64/128/256), so the layer falls back to a plain
    FP16 GEMM. This inverts the checkpoint's ``nvfp4-pack-quantized`` layout
    (the one qatfactory stores for n % 64 != 0), which is the natural row-major
    packing:

      * ``weight``       : (n, k//2) uint8 -- row n holds the k 4-bit codes for
                           output column n, 2 codes/byte (low nibble = even k,
                           high nibble = odd k).
      * ``weight_scale`` : (n, k//16) fp8_e4m3fn -- one scale per 16-wide K
                           group and output column.

    Args:
        weight: packed uint8 weight of shape (n, k//2).
        weight_scale: per-group FP8-E4M3 scale of shape (n, k//16).
        k: inner (reduction) dimension.
        n: number of output columns.

    Returns:
        A dense FP16 weight of shape (n, k) with the global scale NOT applied
        (the caller applies it in the GEMM epilogue, matching the Marlin path).
    """
    device = weight.device
    rows = weight.view(n, k // 2)
    lo = (rows & 0x0F).to(torch.int32)            # low nibble -> even k
    hi = ((rows >> 4) & 0x0F).to(torch.int32)     # high nibble -> odd k
    lut = _E2M1_LUT.to(device)
    deq = torch.empty(n, k, dtype=torch.float32, device=device)
    deq[:, 0::2] = lut[lo]
    deq[:, 1::2] = lut[hi]
    scale = weight_scale.to(torch.float32)                # (n, k//16)
    scale_exp = scale.repeat_interleave(16, dim=1)       # (n, k)
    return (deq * scale_exp).to(torch.float16)


class Sm70MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """Weight-only NVFP4 GEMM on SM70 (V100/Volta): dequant -> FP16.

    Reads the checkpoint's ``nvfp4-pack-quantized`` layout directly (no
    repack) and applies the per-tensor global scale as the GEMM epilogue alpha.
    Output dims that are a multiple of 64 use the fast Marlin GEMM kernel.
    Output dims that are not (e.g. the GDN ``in_proj_ba`` with n=96, or the
    standalone n=48 gating projections) fall back to a one-time dequant to a
    dense FP16 weight cached on the layer, so each forward is a plain FP16
    GEMM. Only supported on exact SM70 with the ``VLLM_SM70_NVFP4`` env switch
    and a registered SM70 GEMM op; otherwise it fails closed and the upstream
    kernel selection is left untouched.
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
        # Handles all n: n % 64 == 0 uses the Marlin kernel, anything else uses
        # the FP16 dequant fallback.
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
            # The Marlin kernel's packed macro is 64/128/256, so a non-64
            # multiple output dim (e.g. n=96) cannot use it. Dequant the
            # natural checkpoint layout to a dense FP16 weight once and cache
            # it; each forward then runs a plain FP16 GEMM.
            W_fp16 = dequant_nvfp4_to_fp16(layer.weight, layer.weight_scale, k, n)
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
                "Using SM70 NVFP4 GEMM (dequant -> FP16); weight layout read "
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
        m = a.shape[0]

        # SM70 NVFP4: dequant to FP16 + standard FP16 GEMM.
        #
        # Correct dequant: w_true = (fp4_codes × group_scale) / weight_global_scale
        # We dequant the NVFP4 weight to a dense FP16 tensor, run F.linear,
        # then divide by weight_global_scale. <0.1% error vs true weight.
        #
        # Small layers (n%64!=0) have their dequant cached at load time.
        # Large layers (n%64==0) dequant on-the-fly (not cached, to avoid
        # OOM — 27B model in FP16 = 54 GB > 32 GB V100).
        W_fp16 = getattr(layer, "_sm70_nvfp4_fp16_weight", None)
        if W_fp16 is None:
            W_fp16 = dequant_nvfp4_to_fp16(
                layer.weight, layer.weight_scale, k, n
            )
            # Only cache small weights (already pre-cached by
            # process_weights_after_loading for n%64!=0). For large layers,
            # W_fp16 is a temporary that will be freed after this call.
            if n % 64 != 0:
                layer._sm70_nvfp4_fp16_weight = W_fp16

        out = F.linear(a, W_fp16).to(torch.float32)
        # NVFP4: weight_global_scale is a divisor (denormalization factor)
        out = out.div_(layer.weight_global_scale.to(torch.float32))
        if bias is not None:
            out = out + bias.to(torch.float32)

        return out.to(out_dtype).reshape(*x.shape[:-1], n)
