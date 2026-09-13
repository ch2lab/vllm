# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
from torch import Tensor

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from ..op import register_op


def _use_sm70_fused_rms() -> bool:
    # SM70-specific fused triton rms_norm: eager nsys shows the native path
    # fans out into pow/reduce/rsqrt/mul ATen kernels (~175 calls/step in
    # GDN decode, ~1.4ms). One triton kernel replaces the whole op.
    return (
        os.environ.get("VLLM_SM70_FUSED_RMSNORM", "1") == "1"
        and current_platform.is_cuda()
        and current_platform.has_device_capability(70)
        and not current_platform.has_device_capability(80)
    )


@triton.jit
def _rms_norm_kernel(
    x,
    w,
    y,
    eps,
    H: tl.constexpr,
    VS: tl.constexpr,
    HAS_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < H
    xv = tl.load(x + row * H + cols, mask=m, other=0.0).to(tl.float32)
    var = tl.sum(tl.where(cols < VS, xv * xv, 0.0), axis=0) / VS
    inv = tl.rsqrt(var + eps)
    if HAS_W:
        wv = tl.load(w + cols, mask=m, other=1.0)
        yv = (xv * inv).to(wv.dtype) * wv
    else:
        yv = xv * inv
    tl.store(y + row * H + cols, yv, mask=m)


def _rms_norm_fused(x: Tensor, weight: Tensor | None, epsilon: float,
                    variance_size: int | None = None) -> Tensor:
    H = x.shape[-1]
    block = triton.next_power_of_2(H)
    y = torch.empty_like(x)
    vs = H if variance_size is None else variance_size
    w = weight if weight is not None else x
    _rms_norm_kernel[(x.numel() // H,)](
        x, w, y, epsilon, H, vs, weight is not None, block,
        num_warps=min(block // 256, 8) if block >= 1024 else 1,
    )
    return y.to(x.dtype)


@register_op
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    """Weighted root-mean-square layer normalization"""
    if _use_sm70_fused_rms() and x.is_cuda and x.is_contiguous():
        return _rms_norm_fused(x, weight, epsilon, variance_size)
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


@rms_norm.register_input_generator
def _rms_norm_input_generator(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    epsilon: float = 1e-5,
    device: torch.device | str | None = None,
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    return x, weight, epsilon


# Reductions in rms_norm accumulate rounding error at large shapes
# (e.g. 32768x16384), causing a few elements out of millions to exceed
# the default float16 tolerance.
rms_norm.override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)


@triton.jit
def _fused_add_rms_norm_kernel(
    x,
    xr,
    w,
    y,
    eps,
    H: tl.constexpr,
    VS: tl.constexpr,
    HAS_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < H
    xv = tl.load(x + row * H + cols, mask=m, other=0.0).to(tl.float32)
    rv = tl.load(xr + row * H + cols, mask=m, other=0.0).to(tl.float32)
    xv = xv + rv
    tl.store(xr + row * H + cols, xv.to(xr.dtype.element_ty), mask=m)
    var = tl.sum(tl.where(cols < VS, xv * xv, 0.0), axis=0) / VS
    inv = tl.rsqrt(var + eps)
    if HAS_W:
        wv = tl.load(w + cols, mask=m, other=1.0)
        yv = (xv * inv).to(wv.dtype) * wv
    else:
        yv = xv * inv
    tl.store(y + row * H + cols, yv, mask=m)


def _fused_add_rms_norm_fused(x: Tensor, x_residual: Tensor,
                              weight: Tensor | None, epsilon: float,
                              variance_size: int | None = None) -> tuple[Tensor, Tensor]:
    H = x.shape[-1]
    block = triton.next_power_of_2(H)
    y = torch.empty_like(x)
    vs = H if variance_size is None else variance_size
    w = weight if weight is not None else x
    _fused_add_rms_norm_kernel[(x.numel() // H,)](
        x, x_residual, w, y, epsilon, H, vs, weight is not None, block,
        num_warps=min(block // 256, 8) if block >= 1024 else 1,
    )
    return y.to(x.dtype), x_residual


@register_op(allow_inplace=True)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused add and weighted root-mean-square layer normalization"""
    if (
        _use_sm70_fused_rms()
        and x.is_cuda
        and x.is_contiguous()
        and x_residual.is_contiguous()
    ):
        return _fused_add_rms_norm_fused(x, x_residual, weight, epsilon, variance_size)
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + x_residual.to(torch.float32)
    x_residual = x.to(orig_dtype)

    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype), x_residual


# fused_add_rms_norm has similar rounding error accumulation as rms_norm
fused_add_rms_norm.override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)


@fused_add_rms_norm.register_input_generator
def _fused_add_rms_norm_input_generator(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    epsilon: float = 1e-5,
    device: torch.device | str | None = None,
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    x_residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    return x, x_residual, weight, epsilon
