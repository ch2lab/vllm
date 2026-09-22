# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


class Sm70Fp8DequantLinearKernel(FP8ScaledMMLinearKernel):
    """Dequantize-to-16-bit fallback for W8A8 FP8 linears on exact SM70.

    Volta (V100) has no FP8 hardware, so every native FP8 GEMM backend
    (CUTLASS, FlashInfer, Marlin, torch._scaled_mm) is unusable there. This
    kernel dequantizes the FP8 weights to the compute dtype once at load
    time and runs a plain 16-bit GEMM at apply time, keeping W8A8-style
    linear layers (e.g. ModelOpt mixed-precision checkpoints) functional on
    SM70.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        if compute_capability is None:
            cap = current_platform.get_device_capability()
            compute_capability = cap.to_int() if cap is not None else None
        if compute_capability == 70:
            return True, None
        return False, "requires exact SM70 (V100/Volta)."

    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
            and c.weight_quant_key.scale.group_shape.is_per_tensor()
        ):
            return True, None
        return False, "requires per-tensor weight and activation scales."

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence
    ) -> None:
        super().__init__(c, layer_param_names)
        self._weight_deq: torch.Tensor | None = None

    def input_quant_key(self) -> QuantKey | None:
        if self.config.activation_quant_key == kFp8StaticTensorSym:
            return kFp8StaticTensorSym
        return None

    def get_output_padding(self) -> int | None:
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w, w_s = layer.weight, layer.weight_scale
        self._weight_deq = (w.float() * w_s).to(self.config.input_dtype)

    def _get_layer_params(self, layer):
        w, w_s, x_s, x_s_ub = self.layer_param_names
        if self._weight_deq is None:
            self.process_weights_after_loading(layer)
        return (
            self._weight_deq,
            getattr(layer, w_s),
            getattr(layer, x_s, None),
            getattr(layer, x_s_ub, None),
        )

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # B is already dequantized to the compute dtype (see process hook).
        # A arrives FP8 (or pre-quantized); dequantize it to compute dtype.
        a = (A.float() * As).to(out_dtype) if A.dtype == self.fp8_dtype else A
        out = a @ B
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
