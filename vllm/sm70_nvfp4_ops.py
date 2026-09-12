# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM70 (Volta / V100) NVFP4 GEMM bindings.

Binds the SM70 Marlin NVFP4 GEMM op (registered as
``torch.ops._C.sm70_nvfp4_marlin_gemm`` in the vLLM C extension) and gates it
behind the ``VLLM_SM70_NVFP4`` env switch and an exact-SM70 hardware check.

The NVFP4 GEMM reads the checkpoint's ``nvfp4-pack-quantized`` layout directly
and applies ``global_scale`` internally (epilogue alpha), so no repack /
``prepare`` step is needed.
"""

import torch

from vllm import envs

__all__ = ["is_exact_sm70", "is_enabled", "gemm"]


def is_exact_sm70() -> bool:
    """Return True iff the current CUDA device is exactly SM70 (V100/Volta)."""
    if not torch.cuda.is_available():
        return False
    return tuple(torch.cuda.get_device_capability()) == (7, 0)


def _op_registered() -> bool:
    """True iff the SM70 NVFP4 GEMM op is present in the loaded C extension.

    The op is registered by the stable extension (loaded as
    ``vllm._C_stable_libtorch``) but lives in the ``torch.ops._C`` namespace.
    The lookup is guarded so an unloaded extension is treated as unavailable
    rather than raising.
    """
    try:
        return hasattr(torch.ops._C, "sm70_nvfp4_marlin_gemm")
    except Exception:
        return False


def is_enabled() -> bool:
    """Return True iff the SM70 NVFP4 GEMM should be used."""
    if not envs.VLLM_SM70_NVFP4:
        return False
    if not is_exact_sm70():
        return False
    return _op_registered()


def gemm(a, c, b_q_weight, b_scales, global_scale, m, n, k, group_size):
    """SM70 NVFP4 GEMM: ``c = a @ b_q_weight^T`` (dequantized in-kernel).

    Thin wrapper around ``torch.ops._C.sm70_nvfp4_marlin_gemm``. The kernel
    applies ``global_scale`` internally as the epilogue alpha, so it is passed
    through unchanged. ``group_size`` is 16 for NVFP4.

    Args:
        a: FP16 activation tensor of shape (m, k).
        c: FP16 output buffer of shape (m, n).
        b_q_weight: NVFP4-packed weight (uint8).
        b_scales: group-16 scales (fp8_e4m3fn).
        global_scale: per-tensor FP32 scale.
        m: number of rows of ``a`` / ``c``.
        n: number of output columns.
        k: inner (reduction) dimension.
        group_size: NVFP4 group size (16).

    Returns:
        The output tensor (``c``).
    """
    return torch.ops._C.sm70_nvfp4_marlin_gemm(
        a, c, b_q_weight, b_scales, global_scale, m, n, k, group_size
    )
