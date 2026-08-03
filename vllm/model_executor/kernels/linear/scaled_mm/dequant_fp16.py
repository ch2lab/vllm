# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


class DequantFP16ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """SM70 fallback: dequantize FP8 weights to FP16 at load time,
    then run standard F.linear at runtime."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        if compute_capability is not None:
            if compute_capability >= 75:
                return False, "only for SM70 (cc < 75)."
        elif current_platform.has_device_capability(75):
            return False, "only for SM70 (cc < 75)."
        return True, None

    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_name, w_s_name, _, _ = self.layer_param_names
        w = getattr(layer, w_name)
        w_s = getattr(layer, w_s_name)

        w_fp16 = w.to(torch.float16)
        if w_s.numel() == 1:
            w_fp16 = w_fp16 * w_s.item()
        elif w_s.dim() == 1:
            w_fp16 = w_fp16 * w_s.unsqueeze(0)
        else:
            w_fp16 = w_fp16 * w_s

        layer.weight = torch.nn.Parameter(w_fp16, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(1, dtype=torch.float32, device=w.device),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w = layer.weight
        return F.linear(x, w, bias)

    def apply_scaled_mm(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("apply_weights is overridden directly.")
