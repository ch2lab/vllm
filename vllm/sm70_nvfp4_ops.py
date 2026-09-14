# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM70 (Volta / V100) NVFP4 op bindings.

Binds the SM70 NVFP4 ops (registered in the vLLM C extension) and gates them
behind the ``VLLM_SM70_NVFP4`` env switch and an exact-SM70 hardware check:

* ``sm70_nvfp4_dequant`` -- the active weight path: a single fused pass that
  reads the checkpoint's natural row-major packed NVFP4 codes + the per-16
  group FP8-E4M3 scales and emits a dense FP16 weight (the per-tensor global
  scale is baked in). The linear layer then runs a plain FP16 GEMM.
* ``sm70_nvfp4_marlin_gemm`` -- the Marlin NVFP4 GEMM (dequant fused
  in-kernel), present for the macro-N layout-matched fast path.
"""

import torch

from vllm import envs

__all__ = ["is_exact_sm70", "is_enabled", "gemm", "dequant"]


def is_exact_sm70() -> bool:
    """Return True iff the current CUDA device is exactly SM70 (V100/Volta)."""
    if not torch.cuda.is_available():
        return False
    return tuple(torch.cuda.get_device_capability()) == (7, 0)


def _op_registered() -> bool:
    """True iff the SM70 NVFP4 ops are present in the loaded C extension.

    The ops are registered by the stable extension (loaded as
    ``vllm._C_stable_libtorch``) but live in the ``torch.ops._C`` namespace.
    The lookup is guarded so an unloaded extension is treated as unavailable
    rather than raising.
    """
    try:
        return hasattr(torch.ops._C, "sm70_nvfp4_dequant")
    except Exception:
        return False


def is_enabled() -> bool:
    """Return True iff the SM70 NVFP4 path (dequant -> FP16 GEMM) is usable."""
    if not envs.VLLM_SM70_NVFP4:
        return False
    if not is_exact_sm70():
        return False
    return _op_registered()


def gemm(a, c, b_q_weight, b_scales, global_scale, m, n, k, group_size):
    """SM70 NVFP4 GEMM: ``c = a @ b_q_weight^T`` (dequantized in-kernel).

    Thin wrapper around ``torch.ops._C.sm70_nvfp4_marlin_gemm``. The kernel
    applies ``global_scale`` internally as the epilogue alpha. The SM70
    fast/skip_flop dequant emits 2^-22x the true dequantized weight, so the
    caller's (natural-scale) ``global_scale`` is multiplied by 2^22 before the
    call -- the kernel's internal 2^-22 then nets to 1.0, giving the correct
    end-to-end output magnitude for natural-scale checkpoints. The raw C++ op
    is unchanged; this compensation lives in the wrapper. ``group_size`` is 16
    for NVFP4.

    Args:
        a: FP16 activation tensor of shape (m, k).
        c: FP16 output buffer of shape (m, n).
        b_q_weight: NVFP4-packed weight (uint8).
        b_scales: group-16 scales (fp8_e4m3fn).
        global_scale: per-tensor FP32 scale (natural-scale, not pre-scaled).
        m: number of rows of ``a`` / ``c``.
        n: number of output columns.
        k: inner (reduction) dimension.
        group_size: NVFP4 group size (16).

    Returns:
        The output tensor (``c``).
    """
    # The SM70 fast/skip_flop dequant emits 2^-22x the true dequantized weight
    # (it omits the FP4->FP16 and FP8->FP16 exponent biases). Multiply the
    # caller's scale by 2^22 so the kernel's internal 2^-22 nets to 1.0.
    #
    # NVFP4 convention: weight_global_scale is a DENORMALIZATION factor
    # (divisor), not a multiplier. The true dequantized weight is:
    #   w_true = w_fp4_code * group_scale / weight_global_scale
    # So the epilogue must use 1/weight_global_scale (the inverse).
    global_scale = (1.0 / global_scale) * 4194304.0  # (1/GS) * 2^22
    return torch.ops._C.sm70_nvfp4_marlin_gemm(
        a, c, b_q_weight, b_scales, global_scale, m, n, k, group_size
    )


def dequant(weight, weight_scale, global_scale):
    """Dequant the checkpoint's natural NVFP4 weight to a dense FP16 weight.

    Single fused pass (SM70 C++ kernel): reads the row-major packed NVFP4 codes
    + the per-16-group FP8-E4M3 scales, and writes the dense FP16 weight with
    the per-tensor global_scale (the layer's weight_global_scale multiplier)
    baked in. Replaces the multi-pass torch dequant that dominated the on-the-
    fly dequant cost for the large (n%64==0) layers.

    Args:
        weight: packed NVFP4 weight (uint8), shape (n, k//2).
        weight_scale: per-16-group FP8-E4M3 scale, shape (n, k//16).
        global_scale: per-tensor FP32 multiplier (layer.weight_global_scale).

    Returns:
        Dense FP16 weight of shape (n, k).
    """
    if torch.is_tensor(global_scale):
        global_scale = float(global_scale.item())
    return torch.ops._C.sm70_nvfp4_dequant(
        weight.contiguous(), weight_scale.contiguous(), float(global_scale)
    )
