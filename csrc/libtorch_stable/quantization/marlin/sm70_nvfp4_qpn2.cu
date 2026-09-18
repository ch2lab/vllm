// SM70 NVFP4 QPN2 decode GEMM — high-occupancy decode kernel + stable-ABI ops.
//
// Ported from 1Cat-vLLM csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu
// (Apache-2.0; the execution layout derives from dnv2003/v100-skinny, MIT —
// see LICENSE.v100-skinny in that directory) for the `_C_stable_libtorch`
// (stable-ABI) build on SM70 (V100).
//
// Why: the SM70 Marlin NVFP4 GEMM is latency-bound at 25% occupancy (114
// registers and 20.48 KB of double-buffered A+B smem each cap it at 4
// blocks/SM), reaching only ~224 GB/s (36% of V100 DRAM peak) for decode
// (m<=8). This kernel feeds the WMMA m8n8k4 B-operand directly from in-register
// dequant (no smem B staging), reads A straight from gmem, and keeps shared
// memory minimal, reaching ~588 GB/s (65% peak) — ~2.6x faster. Best config:
// SplitK=8, NAcc=1.
//
// A one-time prepack (row-major [n,k/2] NVFP4 + [n,k/16] E4M3 scales -> QPN2
// column-major codes/scales) runs at weight-load time.
//
// Registers ops:
//   sm70_nvfp4_qpn2_prepare(weight, weight_scale) -> (codes, scales)
//   sm70_nvfp4_qpn2_gemm(out, input, codes, scales, global_scale, split_k, nacc)

#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace qpn2 {

constexpr int kPrepareThreads = 256;
constexpr int kQpn2RowsPerCta = 8;

__device__ __forceinline__ int qpn2_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int qpn2_logical_k(int physical_k) {
  const int local = physical_k & 7;
  return (physical_k & 8) + ((local & 3) << 1) + (local >> 2);
}

__global__ void nvfp4_qpn2_prepack_codes_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ weight, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 2;
  if (index >= numel) {
    return;
  }

  const int physical_byte = static_cast<int>(index & 7);
  size_t outer = index >> 3;
  const int lane = static_cast<int>(outer & 31);
  outer >>= 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  const int logical_k0 = group * 16 + qpn2_logical_k(physical_byte * 2);
  const int logical_k1 = group * 16 + qpn2_logical_k(physical_byte * 2 + 1);
  const int k_bytes = k >> 1;
  const uint8_t packed0 =
      weight[static_cast<size_t>(column) * k_bytes + (logical_k0 >> 1)];
  const uint8_t packed1 =
      weight[static_cast<size_t>(column) * k_bytes + (logical_k1 >> 1)];
  const uint8_t code0 =
      static_cast<uint8_t>((packed0 >> ((logical_k0 & 1) * 4)) & 0x0f);
  const uint8_t code1 =
      static_cast<uint8_t>((packed1 >> ((logical_k1 & 1) * 4)) & 0x0f);
  output[index] = static_cast<uint8_t>(code0 | (code1 << 4));
}

__global__ void nvfp4_qpn2_prepack_scales_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ scales, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 16;
  if (index >= numel) {
    return;
  }

  const int lane = static_cast<int>(index & 31);
  size_t outer = index >> 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  output[index] = scales[static_cast<size_t>(column) * groups_k16 + group];
}

__device__ __forceinline__ half2 fp8e4m3_to_half2(uint8_t value) {
  const unsigned short bits =
      ((static_cast<unsigned short>(value) & 0x80u) << 8) |
      ((static_cast<unsigned short>(value) & 0x7fu) << 7);
  const half converted =
      __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                               half2 output[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  values[1] = ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  values[2] = ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  values[3] = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    output[index] = __hmul2(*reinterpret_cast<half2*>(&values[index]), scale);
  }
}

#define VLLM_SM70_QPN2_MMA(C, A0, A1, B0, B1)                       \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int SplitK, int NAcc, int RowTiles = 1>
__global__ void nvfp4_qpn2_sm70_kernel(const uint8_t* __restrict__ codes,
                                       const uint8_t* __restrict__ group_scales,
                                       const half* __restrict__ input,
                                       half* __restrict__ output,
                                       float* __restrict__ partials_out,
                                       int n, int k, int m, float global_scale,
                                       int ksplit) {
  static_assert(RowTiles == 1 || RowTiles == 2,
                "NVFP4 QPN2 supports one or two 8-row tiles");
  __shared__ float partials[SplitK][RowTiles * 256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row_base = blockIdx.y * kQpn2RowsPerCta * RowTiles;
  const int groups_k16 = k >> 4;
  const int groups_per_ksplit = groups_k16 / ksplit;
  const int groups_per_warp = groups_per_ksplit / SplitK;
  const int group_begin =
      blockIdx.z * groups_per_ksplit + warp * groups_per_warp;
  const uint2* code_ptr = reinterpret_cast<const uint2*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const uint8_t* scale_ptr =
      group_scales + static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half2 global_scale2 = __float2half2_rn(global_scale * 16384.0f);

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][chain][index] = 0.0f;
      }
    }
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const uint2 packed = __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    const half2 scale = __hmul2(
        fp8e4m3_to_half2(__ldg(scale_ptr + static_cast<size_t>(group) * 32)),
        global_scale2);
    half2 weights[8];
    dequant_e2m1x8(packed.x, scale, weights);
    dequant_e2m1x8(packed.y, scale, weights + 4);

    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      const int row = row_base + row_tile * kQpn2RowsPerCta + local_row;
      if (row < m) {
        const half* input_row = input + static_cast<size_t>(row) * k;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
      VLLM_SM70_QPN2_MMA(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][0][index] += accum[row_tile][chain][index];
      }
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int output_row = row_tile * kQpn2RowsPerCta + (index & 2) +
                             ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col =
          (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
      partials[warp][output_row * 32 + quadpair * 8 + output_col] =
          accum[row_tile][0][index];
    }
  }
  __syncthreads();

  for (int element = threadIdx.x; element < RowTiles * 256;
       element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    const int output_row = row_base + (element >> 5);
    const int output_col = element & 31;
    if (output_row < m) {
      const size_t out_idx =
          static_cast<size_t>(output_row) * n +
          (static_cast<size_t>(tile) * 32 + output_col);
      if (ksplit == 1) {
        output[out_idx] = __float2half(value);
      } else {
        partials_out[static_cast<size_t>(blockIdx.z) * m * n + out_idx] =
            value;
      }
    }
  }
}
// Reduce the per-ksplit float partials (layout [ksplit, m, n]) into the
// half output. Launched on the same stream right after the split kernel.
__global__ void nvfp4_qpn2_reduce_kernel(const float* __restrict__ partials,
                                         half* __restrict__ output, int m,
                                         int n, int ksplit) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<size_t>(m) * n) {
    return;
  }
  const size_t row = idx / n;
  const size_t col = idx - row * n;
  float value = 0.0f;
  for (int z = 0; z < ksplit; ++z) {
    value += partials[(static_cast<size_t>(z) * m + row) * n + col];
  }
  output[idx] = __float2half(value);
}

template <int SplitK, int NAcc, int RowTiles = 1>
void launch_qpn2(const uint8_t* codes, const uint8_t* scales, const half* input,
                 half* output, float* partials_out, int n, int k, int m,
                 float global_scale, int ksplit, cudaStream_t stream) {
  constexpr int kRowsPerCta = kQpn2RowsPerCta * RowTiles;
  const dim3 grid(n / 32, (m + kRowsPerCta - 1) / kRowsPerCta, ksplit);
  nvfp4_qpn2_sm70_kernel<SplitK, NAcc, RowTiles>
      <<<grid, (32 * SplitK), 0, stream>>>(codes, scales, input, output,
                                            partials_out, n, k, m, global_scale,
                                            ksplit);
  if (ksplit > 1) {
    const int total = m * n;
    nvfp4_qpn2_reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        partials_out, output, m, n, ksplit);
  }
}
}  // namespace qpn2

namespace {
inline void qpn2_check_launch() {
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "SM70 NVFP4 QPN2 launch failed: ",
                  cudaGetErrorString(err));
}
}  // namespace
// Persistent per-device float scratch for the K-split partials
// ([ksplit, m, n] row-major). Grown lazily to the largest shape seen so far;
// warmup covers the max before CUDA-graph capture, so the captured address is
// stable. If a grow is requested mid-capture, the caller falls back to
// ksplit=1 (checked before growth via cudaStreamIsCapturing).
struct Qpn2KsplitState {
  torch::stable::Tensor buf;
};

Qpn2KsplitState& qpn2_ksplit_state(int device) {
  static std::vector<Qpn2KsplitState*> states;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (static_cast<int>(device) >= static_cast<int>(states.size())) {
    states.resize(device + 1, nullptr);
  }
  if (!states[device]) {
    states[device] = new Qpn2KsplitState();
  }
  return *states[device];
}

// A/B knob: VLLM_SM70_QPN2_KSPLIT=<int> caps ksplit (0 = K-split off).
// Unset = 0 (off): measured flat vs ksplit=1 on all model shapes, so the
// legacy path stays the default. Set e.g. VLLM_SM70_QPN2_KSPLIT=8 to enable.
int qpn2_ksplit_cap() {
  static int cap = -1;
  if (cap < 0) {
    const char* env = std::getenv("VLLM_SM70_QPN2_KSPLIT");
    cap = (env != nullptr) ? std::atoi(env) : 0;
    if (cap < 0) cap = 0;
  }
  return cap;
}


std::vector<torch::stable::Tensor> sm70_nvfp4_qpn2_prepare(
    torch::stable::Tensor& weight, torch::stable::Tensor& weight_scale) {
  torch::stable::accelerator::DeviceGuard device_guard(
      weight.get_device_index());
  int64_t n = weight.size(0);
  int64_t k = weight.size(1) * 2;
  STD_TORCH_CHECK(n % 32 == 0 && k % 16 == 0,
                  "SM70 NVFP4 QPN2 prepare needs n%32==0 and k%16==0, got n=",
                  n, " k=", k);
  int threads = static_cast<int>(qpn2::kPrepareThreads);
  cudaStream_t stream = get_current_cuda_stream(weight.get_device_index());
  const uint8_t* w = reinterpret_cast<const uint8_t*>(weight.data_ptr());
  const uint8_t* s = reinterpret_cast<const uint8_t*>(weight_scale.data_ptr());

  std::vector<int64_t> codes_shape = {n, k / 2};
  torch::headeronly::IntHeaderOnlyArrayRef codes_sizes(codes_shape.data(),
                                                       codes_shape.size());
  auto codes = torch::stable::new_empty(
      weight, codes_sizes, torch::headeronly::ScalarType::Byte);
  int code_blocks = static_cast<int>((n * k / 2 + threads - 1) / threads);
  qpn2::nvfp4_qpn2_prepack_codes_kernel<<<code_blocks, threads, 0, stream>>>(
      reinterpret_cast<uint8_t*>(codes.mutable_data_ptr()), w,
      static_cast<int>(n), static_cast<int>(k));

  std::vector<int64_t> scales_shape = {n, k / 16};
  torch::headeronly::IntHeaderOnlyArrayRef scales_sizes(scales_shape.data(),
                                                        scales_shape.size());
  auto scales = torch::stable::new_empty(
      weight, scales_sizes, torch::headeronly::ScalarType::Byte);
  int scale_blocks = static_cast<int>((n * k / 16 + threads - 1) / threads);
  qpn2::nvfp4_qpn2_prepack_scales_kernel<<<scale_blocks, threads, 0, stream>>>(
      reinterpret_cast<uint8_t*>(scales.mutable_data_ptr()), s,
      static_cast<int>(n), static_cast<int>(k));
  qpn2_check_launch();
  return {codes, scales};
}

void sm70_nvfp4_qpn2_gemm(torch::stable::Tensor& out,
                          torch::stable::Tensor& input,
                          torch::stable::Tensor& codes,
                          torch::stable::Tensor& scales, double global_scale,
                          int64_t split_k, int64_t accumulator_chains) {
  torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  STD_TORCH_CHECK(split_k == 8 || split_k == 16 || split_k == 32,
                  "SM70 NVFP4 QPN2 split_k must be 8, 16, or 32");
  STD_TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
                  "SM70 NVFP4 QPN2 accumulator_chains must be 1 or 2");
  int n = static_cast<int>(out.size(1));
  int k = static_cast<int>(input.size(1));
  int m = static_cast<int>(input.size(0));
  STD_TORCH_CHECK(k / 16 % split_k == 0,
                  "SM70 NVFP4 QPN2 requires k/16 divisible by split_k");
  const uint8_t* code_ptr = reinterpret_cast<const uint8_t*>(codes.data_ptr());
  const uint8_t* scale_ptr =
      reinterpret_cast<const uint8_t*>(scales.data_ptr());
  const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr());
  half* output_ptr = reinterpret_cast<half*>(out.mutable_data_ptr());
  cudaStream_t stream = get_current_cuda_stream(input.get_device_index());
  float gs = static_cast<float>(global_scale);

  // SM-saturating K-split: small-N GEMMs launch too few CTAs to keep the 80
  // SMs busy (e.g. n=2560 -> 80 CTAs -> ~10% occupancy -> ~1/3 DRAM peak).
  // Split K across grid-z so CTA count ~= 5120/split_k (8 CTAs of 256 threads
  // per SM); the per-slice float partials are reduced by a follow-up kernel.
  // ksplit must divide k/16/split_k.
  const int gk = k / 16;
  const int maxks = gk / static_cast<int>(split_k);
  const int nctas = n / 32;
  int ksplit = 1;
  {
    // target CTA count = 5120/split_k (8 CTAs/SM of (32*split_k) threads)
    const int target = 5120 / static_cast<int>(split_k);
    int want = (target + nctas - 1) / nctas;
    if (want > 8) want = 8;
    if (want > maxks) want = maxks;
    for (int d = want; d > 1; --d) {
      if (gk % (static_cast<int>(split_k) * d) == 0) {
        ksplit = d;
        break;
      }
    }
    const int cap = qpn2_ksplit_cap();
    if (cap == 0) {
      ksplit = 1;
    } else if (ksplit > cap) {
      // clamp to the largest divisor <= cap
      ksplit = 1;
      for (int d = std::min(cap, maxks); d > 1; --d) {
        if (gk % (static_cast<int>(split_k) * d) == 0) {
          ksplit = d;
          break;
        }
      }
    }
  }

  float* partials_ptr = nullptr;
  if (ksplit > 1) {
    auto& st = qpn2_ksplit_state(input.get_device_index());
    const int64_t need = static_cast<int64_t>(ksplit) * m * n;
    if (st.buf.defined() && st.buf.numel() >= need) {
      partials_ptr = reinterpret_cast<float*>(st.buf.mutable_data_ptr());
    } else {
      // Grow only when not mid-capture; otherwise fall back to no K-split
      // (a reallocation would invalidate an in-flight captured graph).
      cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(stream, &cap);
      if (cap == cudaStreamCaptureStatusNone) {
        std::vector<int64_t> shape = {need};
        torch::headeronly::IntHeaderOnlyArrayRef sizes(shape.data(),
                                                       shape.size());
        st.buf = torch::stable::new_empty(input, sizes,
                                          torch::headeronly::ScalarType::Float);
      } else {
        ksplit = 1;
      }
      partials_ptr =
          (ksplit > 1) ? reinterpret_cast<float*>(st.buf.mutable_data_ptr())
                       : nullptr;
    }
  }

#define LAUNCH_QPN2(ROWS, SPLIT, NACC) \
      qpn2::launch_qpn2<SPLIT, NACC, ROWS>(code_ptr, scale_ptr, input_ptr, \
                                          output_ptr, partials_ptr, n, k, m, \
                                          gs, ksplit, stream)
  bool two_tile = (m > qpn2::kQpn2RowsPerCta) && (m <= 16);
  if (two_tile && split_k == 8 && accumulator_chains == 1) {
    LAUNCH_QPN2(2, 8, 1);
  } else if (two_tile && split_k == 8) {
    LAUNCH_QPN2(2, 8, 2);
  } else if (two_tile && split_k == 16 && accumulator_chains == 1) {
    LAUNCH_QPN2(2, 16, 1);
  } else if (two_tile && split_k == 16) {
    LAUNCH_QPN2(2, 16, 2);
  } else if (split_k == 8 && accumulator_chains == 1) {
    LAUNCH_QPN2(1, 8, 1);
  } else if (split_k == 8) {
    LAUNCH_QPN2(1, 8, 2);
  } else if (split_k == 16 && accumulator_chains == 1) {
    LAUNCH_QPN2(1, 16, 1);
  } else if (split_k == 16) {
    LAUNCH_QPN2(1, 16, 2);
  } else if (accumulator_chains == 1) {
    LAUNCH_QPN2(1, 32, 1);
  } else {
    LAUNCH_QPN2(1, 32, 2);
  }
#undef LAUNCH_QPN2
  qpn2_check_launch();
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_nvfp4_qpn2_ops) {
  sm70_nvfp4_qpn2_ops.def(
      "sm70_nvfp4_qpn2_prepare(Tensor weight, Tensor weight_scale) -> Tensor[]");
  sm70_nvfp4_qpn2_ops.def(
      "sm70_nvfp4_qpn2_gemm(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, float global_scale, int split_k, int accumulator_chains) "
      "-> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_nvfp4_qpn2_ops) {
  sm70_nvfp4_qpn2_ops.impl("sm70_nvfp4_qpn2_prepare",
                          TORCH_BOX(&sm70_nvfp4_qpn2_prepare));
  sm70_nvfp4_qpn2_ops.impl("sm70_nvfp4_qpn2_gemm",
                          TORCH_BOX(&sm70_nvfp4_qpn2_gemm));
}
