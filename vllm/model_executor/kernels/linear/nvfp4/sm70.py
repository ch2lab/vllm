# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sm70_nvfp4_ops import gemm, is_enabled

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


class Sm70MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """Weight-only NVFP4 GEMM on SM70 (V100/Volta): dequant -> FP16.

    Reads the checkpoint's ``nvfp4-pack-quantized`` layout directly (no
    repack) and applies the per-tensor global scale as the GEMM epilogue
    alpha. Only supported on exact SM70 with the ``VLLM_SM70_NVFP4`` env
    switch and a registered SM70 GEMM op; otherwise it fails closed and the
    upstream kernel selection is left untouched.
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
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # No repack: the SM70 GEMM reads the checkpoint nvfp4 layout directly.
        # Stash nothing; just validate the tensors the GEMM will consume exist.
        for attr in ("weight", "weight_scale", "weight_global_scale"):
            if not hasattr(layer, attr):
                raise AttributeError(
                    f"SM70 NVFP4 kernel requires layer.{attr} to be present"
                )
        logger.info_once(
            "Using SM70 NVFP4 GEMM (dequant -> FP16); weight layout read "
            "directly from the checkpoint."
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
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
        c = torch.empty((m, n), dtype=torch.float16, device=x.device)
        out = gemm(
            a,
            c,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            m,
            n,
            k,
            16,
        )
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.to(out_dtype).reshape(*x.shape[:-1], n)
