// SM70 NVFP4 fused dequant -> dense FP16 (single pass).
//
// The SM70 NVFP4 linear kernel dequants the checkpoint's NVFP4 weight to a dense
// FP16 weight every forward (for the n%64==0 layers that cannot be cached in
// FP16 on a 32GB V100). The torch fallback does this in ~6 separate passes
// (nibble unpack x2, LUT gather x2, strided scatter, repeat_interleave, and a
// global-scale multiply) that move ~1.4 GB for a 52MB -> 105MB conversion,
// making the on-the-fly dequant the dominant decode cost.
//
// This kernel fuses the whole thing into one memory-bound pass: read the
// row-major packed NVFP4 codes + the per-16-group FP8-E4M3 scales, decode, and
// write the dense FP16 weight with the per-tensor global scale baked in (the
// layer stores weight_global_scale as a multiplier, the reciprocal of the
// checkpoint's dequant divisor). The layout matches the checkpoint's natural
// packing (low nibble = even k, high nibble = odd k; one scale per 16-wide K
// group), so no repack is needed.
//
// Registers op `sm70_nvfp4_dequant(weight, weight_scale, global_scale) -> fp16 (n, k)`.

#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace {

// E2M1 (FP4): 1 sign + 2 exp (bias 1) + 1 mant. Magnitudes [0,.5,1,1.5,2,3,4,6].
__device__ __forceinline__ float nvfp4_code_to_float(uint8_t code) {
  static const float MAG[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = MAG[code & 0x7];
  if (code & 0x8) v = -v;
  return v;
}

// FP8-E4M3 -> float (1 sign + 4 exp bias 7 + 3 mant; finite-only, no inf/nan).
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t b) {
  int sgn = (b >> 7) & 1;
  int e = (b >> 3) & 0xF;
  int m = b & 0x7;
  float v = (e == 0) ? (float(m) * (1.0f / 512.0f))
                     : ((1.0f + float(m) / 8.0f) * exp2f(float(e) - 7.0f));
  return sgn ? -v : v;
}

__global__ void nvfp4_dequant_fused_kernel(
    const uint8_t* __restrict__ weight,  // (n, k/2) row-major
    const uint8_t* __restrict__ scale,  // (n, k/16) row-major fp8
    float global_scale, int n, int k,
    __half* __restrict__ out) {  // (n, k)
  // One thread per 16-wide K group: 1 scale byte, 8 weight bytes (16 codes),
  // 32-byte-aligned output. Memory-bound -> vectorize the load + store.
  const int ngroups = k >> 4;
  const int total = n * ngroups;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int i = idx / ngroups;
  const int g = idx - i * ngroups;
  const int kstart = g << 4;
  // k is a multiple of 16, so i*(k/2) + kstart/2 is 8-byte aligned: one 64-bit
  // load. (An 8-byte memcpy does not vectorize when alignment is not provable.)
  const uint64_t w =
      *reinterpret_cast<const uint64_t*>(weight + (size_t)i * (k >> 1) +
                                         (kstart >> 1));
  const float s =
      fp8_e4m3_to_float(scale[(size_t)i * ngroups + g]) * global_scale;
  __half2 h[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const uint8_t b = (uint8_t)(w >> (c * 8));
    h[c] = __half2(__float2half(nvfp4_code_to_float(b & 0xF) * s),
                   __float2half(nvfp4_code_to_float((b >> 4) & 0xF) * s));
  }
  // Store the 16 halves as two 16-byte (128-bit) stores.
  float4* op16 =
      reinterpret_cast<float4*>(out + (size_t)i * k + kstart);
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    float4 fv;
    *reinterpret_cast<__half2*>(&fv.x) = h[q * 4];
    *reinterpret_cast<__half2*>(&fv.y) = h[q * 4 + 1];
    *reinterpret_cast<__half2*>(&fv.z) = h[q * 4 + 2];
    *reinterpret_cast<__half2*>(&fv.w) = h[q * 4 + 3];
    op16[q] = fv;
  }
}

inline void nvfp4_dequant_check_launch() {
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "NVFP4 dequant launch failed: ",
                  cudaGetErrorString(err));
}

}  // namespace

torch::stable::Tensor sm70_nvfp4_dequant(
    torch::stable::Tensor& weight, torch::stable::Tensor& weight_scale,
    double global_scale) {
  torch::stable::accelerator::DeviceGuard device_guard(
      weight.get_device_index());
  int64_t n = weight.size(0);
  int64_t k = weight.size(1) * 2;
  std::vector<int64_t> out_shape = {n, k};
  torch::headeronly::IntHeaderOnlyArrayRef out_sizes(out_shape.data(),
                                                     out_shape.size());
  auto out =
      torch::stable::new_zeros(weight, out_sizes, torch::headeronly::ScalarType::Half);

  int64_t const total = n * (k / 16);
  const int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = get_current_cuda_stream(weight.get_device_index());
  nvfp4_dequant_fused_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(weight.data_ptr()),
      reinterpret_cast<const uint8_t*>(weight_scale.data_ptr()),
      static_cast<float>(global_scale), static_cast<int>(n),
      static_cast<int>(k),
      reinterpret_cast<__half*>(out.mutable_data_ptr()));
  nvfp4_dequant_check_launch();
  return out;
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_nvfp4_dequant_ops) {
  sm70_nvfp4_dequant_ops.def(
      "sm70_nvfp4_dequant(Tensor weight, Tensor weight_scale, "
      "float global_scale) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_nvfp4_dequant_ops) {
  sm70_nvfp4_dequant_ops.impl("sm70_nvfp4_dequant",
                              TORCH_BOX(&sm70_nvfp4_dequant));
}
