# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused NVFP4 KV-quantize-and-scatter for the SM70 attention backend.

Replaces the ~55 python tensor ops of the old ``_nvfp4_write`` (which py-spy
showed burning ~70% of the decode worker's CPU) with one Triton kernel that
does the level-search quantization and the paged scatter on-device. The host
only computes the per-(token, head, group) e4m3 scale bytes (amax + cast),
which the kernel decodes back to fp32 for the nibble search.
"""

import torch
import triton
import triton.language as tl


def _e4m3_lut() -> torch.Tensor:
    """fp32 values of all 256 e4m3 bit patterns (decode table)."""
    lut = torch.zeros(256, dtype=torch.float32)
    for b in range(256):
        if b & 0x80:
            continue  # sign bit unused for KV scales
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0 and m == 0:
            v = 0.0
        else:
            v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        lut[b] = v
    return lut


_E4M3_LUT = _e4m3_lut()


@triton.jit
def _nvfp4_quant_and_scatter_kernel(
    key, value, slot_mapping, kv_cache, k_sf, v_sf,
    page: tl.constexpr, bs: tl.constexpr, dd: tl.constexpr, sd: tl.constexpr,
    c: tl.constexpr, num_heads: tl.constexpr, gs_k, gs_v,
):
    """Quantize one (token, kv-head) K/V slice to nvfp4 and scatter it.

    Layout per page block: [K nibbles (hs/2 B) | K scales (hs/16 B) |
    V nibbles | V scales], NHD per region (matches the python path).
    """
    pid = tl.program_id(0)
    t = pid // num_heads
    h = pid % num_heads

    slot = tl.load(slot_mapping + t)
    if slot < 0:
        return  # padded/invalid slot: skip without a host-side mask
    blk = slot // bs
    off = slot % bs
    base = blk * page + off * c + h * (bs * c)

    d = tl.arange(0, dd * 2)
    kk = tl.load(key + t * num_heads * (dd * 2) + h * (dd * 2) + d)
    vv = tl.load(value + t * num_heads * (dd * 2) + h * (dd * 2) + d)

    # scale byte index per 16-element group
    g = d // 16

    # K quantization (inlined; Triton kernels cannot contain closures)
    s = tl.arange(0, sd)
    k_sf_bytes = tl.load(k_sf + t * num_heads * sd + h * sd + s)
    g = d // 16
    k_sf_g = tl.load(k_sf + t * num_heads * sd + h * sd + g)
    k_sf_val = tl.exp2((((k_sf_g >> 3) & 15).to(tl.float32)) - 7.0) * (
        1.0 + (k_sf_g & 7).to(tl.float32) / 8.0)
    # nearest e2m1 level of the 8-value LUT [0,.5,1,1.5,2,3,4,6],
    # exact midpoint rules (ties resolve to the lower level, like argmin)
    # fp64 division: Volta fp32 div is not correctly rounded (1-ulp errors
    # even with ieee_rounding=True), which flips level decisions at exact
    # midpoint values. KV write is not throughput-critical.
    yk = (tl.abs(kk).to(tl.float64)
          / tl.maximum((k_sf_val * gs_k).to(tl.float64), 1e-30)).to(
        tl.float32)
    ik = tl.where(yk <= 0.25, 0, tl.where(yk <= 0.75, 1,
                  tl.where(yk <= 1.25, 2, tl.where(yk <= 1.75, 3,
                  tl.where(yk <= 2.5, 4, tl.where(yk <= 3.5, 5,
                  tl.where(yk <= 5.0, 6, 7)))))))
    ik = ik.to(tl.uint8)
    nk = tl.where(kk < 0, ik | 8, ik)
    nk2 = tl.reshape(nk, (dd, 2))
    nk_lo, nk_hi = tl.split(nk2)
    kb_out = nk_lo | (nk_hi << 4)

    # V quantization
    v_sf_bytes = tl.load(v_sf + t * num_heads * sd + h * sd + s)
    v_sf_g = tl.load(v_sf + t * num_heads * sd + h * sd + g)
    v_sf_val = tl.exp2((((v_sf_g >> 3) & 15).to(tl.float32)) - 7.0) * (
        1.0 + (v_sf_g & 7).to(tl.float32) / 8.0)
    yv = (tl.abs(vv).to(tl.float64)
          / tl.maximum((v_sf_val * gs_v).to(tl.float64), 1e-30)).to(
        tl.float32)
    iv = tl.where(yv <= 0.25, 0, tl.where(yv <= 0.75, 1,
                  tl.where(yv <= 1.25, 2, tl.where(yv <= 1.75, 3,
                  tl.where(yv <= 2.5, 4, tl.where(yv <= 3.5, 5,
                  tl.where(yv <= 5.0, 6, 7)))))))
    iv = iv.to(tl.uint8)
    nv = tl.where(vv < 0, iv | 8, iv)
    nv2 = tl.reshape(nv, (dd, 2))
    nv_lo, nv_hi = tl.split(nv2)
    vb_out = nv_lo | (nv_hi << 4)

    w = tl.arange(0, dd)
    tl.store(kv_cache + base + w, kb_out)
    vbase = base + num_heads * bs * c
    tl.store(kv_cache + vbase + w, vb_out)
    tl.store(kv_cache + base + dd + s, k_sf_bytes)
    tl.store(kv_cache + vbase + dd + s, v_sf_bytes)


def nvfp4_write(layer, key, value, kv_cache, slot_mapping):
    """Fused drop-in for the SM70 backend's `_nvfp4_write` body."""
    num_tokens = key.shape[0]
    if num_tokens == 0:
        return
    if torch.cuda.is_current_stream_capturing():
        # Capture placeholders may carry out-of-range slot ids; clamp so the
        # scattered values stay in-bounds (results are discarded). Negative
        # (padding) slots are skipped by the kernel itself — no host mask,
        # which would force a device-host sync every call.
        slot_mapping = slot_mapping.clamp(
            max=kv_cache.shape[0] * kv_cache.shape[2] - 1)

    hs = key.shape[-1]
    dd = hs // 2
    sd = hs // 16
    num_kv_heads = key.shape[1]
    bs = kv_cache.shape[2]
    page = kv_cache.stride(0)
    c = dd + sd
    dev = key.device

    # host-side scale bytes (e4m3 of amax/(6*gs)) per (token, head, group)
    def scale_bytes(x, gs):
        xf = x.float().view(num_tokens, num_kv_heads, sd, 16)
        amax = xf.abs().amax(-1, keepdim=True)
        return (amax / (6.0 * gs)).to(torch.float8_e4m3fn).view(torch.uint8)

    # small tensors: keep them dense so the kernel reads fixed shapes
    k_sf = scale_bytes(key, layer._k_scale)
    v_sf = scale_bytes(value, layer._v_scale)

    flat = kv_cache.reshape(-1)
    # Host scale copies avoid a device sync during CUDA graph capture.
    # _*_scale_float is a Python float on every init path; _*_scale_cpu may
    # live on device when created under the CUDA default-device context.
    gs_k = float(getattr(layer, "_k_scale_float",
                         getattr(layer, "_k_scale_cpu", layer._k_scale)))
    gs_v = float(getattr(layer, "_v_scale_float",
                         getattr(layer, "_v_scale_cpu", layer._v_scale)))
    _nvfp4_quant_and_scatter_kernel[(num_tokens * num_kv_heads,)](
        key, value, slot_mapping, flat, k_sf, v_sf,
        page=page, bs=bs, dd=dd, sd=sd, c=c,
        num_heads=num_kv_heads,
        gs_k=gs_k, gs_v=gs_v,
    )