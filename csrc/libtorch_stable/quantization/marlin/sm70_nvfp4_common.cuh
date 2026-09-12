// SM70 Marlin NVFP4 GEMM — shared helpers (merged self-contained subsystem).
//
// Ported from 1Cat-vLLM `csrc/quantization/marlin/sm70_marlin_{common,
// mma,iterator_utils,splitk,gemm}.cuh` for use on SM70 (V100) inside the
// `_C_stable_libtorch` (stable-ABI) build. Only the symbols used by the
// NVFP4 GEMM (`sm70_nvfp4_gemm.cu`) are kept. The dequant helpers the kernel
// needs are defined here directly (the shared `dequant.h` is guarded at
// __CUDA_ARCH__ >= 750, so it is unavailable for an sm_70 build).
//
// Translated from 1Cat's full-libtorch API to the stable ABI:
//   - `torch::Tensor`      -> `torch::stable::Tensor`
//   - `TORCH_CHECK(...)`   -> `STD_TORCH_CHECK(...)`
//   - `C10_CUDA_CHECK(x)`  -> raw cuda error check + `STD_TORCH_CHECK`
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "core/scalar_type.hpp"
#include "libtorch_stable/torch_utils.h"

#include <torch/headeronly/util/Exception.h>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

namespace marlin {

// The two dequant specializations the NVFP4 kernel instantiates (FP4 -> FP16
// and FP8-E4M3 scale -> FP16). Both are pure bit-manipulation, so they are
// valid on SM70. Mirrors 1Cat's `dequant.h`.
template <typename scalar_t2, vllm::ScalarTypeId w_type_id,
          bool skip_flop = false>
__device__ inline void dequant(int q, scalar_t2* frag_b);

template <>
__device__ inline void dequant<half2, vllm::kFE2M1f.id(), true>(int q,
                                                                 half2* frag_b) {
  // Constants for FP4 (E2M1) and FP16 formats
  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;
  constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

  // Extract and shift FP4 values to FP16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
}

template <typename scalar_t2, vllm::ScalarTypeId s_type_id>
__device__ inline void dequant_fp8_scales(int q, scalar_t2* frag_b);

template <>
__device__ inline void dequant_fp8_scales<half2, vllm::kFE4M3fn.id()>(
    int q, half2* frag_b) {
  int Out1 = (q & 0xFF00FF00) >> 1;
  q <<= 8;
  int Out2 = (q & 0xFF00FF00) >> 1;

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
}

}  // namespace marlin

namespace marlin::sm70 {

constexpr int kCtaK = 32;
constexpr int kQuantTileK = 16;
constexpr int kQuantTileN = 64;

template <typename SharedStorage, typename Kernel>
inline size_t configure_sm70_dynamic_smem(Kernel kernel) {
  size_t smem_bytes = sizeof(SharedStorage);
  if (smem_bytes >= (48u << 10)) {
    cudaError_t err =
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
    STD_TORCH_CHECK(err == cudaSuccess, "cudaFuncSetAttribute failed: ",
                    cudaGetErrorString(err));
  }
  return smem_bytes;
}

constexpr int sm70_const_min(int lhs, int rhs) { return lhs < rhs ? lhs : rhs; }

constexpr bool sm70_marlin_basic_geometry_values_supported(int cta_m, int cta_n,
                                                            int cta_k, int warps,
                                                            int warp_m,
                                                            int warp_n,
                                                            int warp_k) {
  return (cta_m == 32 || cta_m == 64 || cta_m == 128 || cta_m == 256) &&
         (cta_n == 64 || cta_n == 128 || cta_n == 256) &&
         (cta_k == 16 || cta_k == 32 || cta_k == 64 || cta_k == 128) &&
         (warps == 4 || warps == 8) && (warp_m == 32 || warp_m == 64) &&
         (warp_n == 32 || warp_n == 64) && (warp_k == 16 || warp_k == 32);
}

constexpr bool sm70_marlin_warp_decomposition_supported(int cta_m, int cta_n,
                                                         int cta_k, int warps,
                                                         int warp_m, int warp_n,
                                                         int warp_k) {
  if (!sm70_marlin_basic_geometry_values_supported(cta_m, cta_n, cta_k, warps,
                                                   warp_m, warp_n, warp_k)) {
    return false;
  }
  if (cta_m % warp_m != 0 || cta_n % warp_n != 0 || cta_k % warp_k != 0) {
    return false;
  }
  return (cta_m / warp_m) * (cta_n / warp_n) * (cta_k / warp_k) == warps;
}

constexpr bool sm70_marlin_pitchlinear_threadmap_supported(
    int shape_contiguous, int shape_strided, int threads,
    int warp_thread_contiguous, int warp_thread_strided,
    int elements_per_access) {
  if (shape_contiguous % elements_per_access != 0) {
    return false;
  }
  int const shape_access_contiguous = shape_contiguous / elements_per_access;
  int const shape_access_strided = shape_strided;
  if (shape_access_contiguous % warp_thread_contiguous != 0 ||
      shape_access_strided % warp_thread_strided != 0) {
    return false;
  }
  int const warp_access_contiguous =
      shape_access_contiguous / warp_thread_contiguous;
  int const warp_access_strided = shape_access_strided / warp_thread_strided;
  int const warp_count = threads / 32;
  if (warp_access_strided <= 0 || warp_count <= 0) {
    return false;
  }
  int const warps_strided =
      warp_access_strided >= warp_count ? warp_count : warp_access_strided;
  if (warps_strided <= 0) {
    return false;
  }
  int const warps_contiguous =
      warp_count > warp_access_strided ? warp_count / warps_strided : 1;
  if (warps_contiguous <= 0) {
    return false;
  }
  int const iterations_contiguous = warp_access_contiguous / warps_contiguous;
  int const iterations_strided = warp_access_strided / warps_strided;
  return iterations_contiguous > 0 && iterations_strided > 0;
}

constexpr bool sm70_marlin_b_threadmap_matches_quant_iterator(int cta_n,
                                                               int cta_k,
                                                               int threads) {
  constexpr int kElementsPerAccess = 8;
  constexpr int kWarpThreadContiguous = 8;
  constexpr int kWarpThreadStrided = 4;
  if (!sm70_marlin_pitchlinear_threadmap_supported(
          cta_n, cta_k, threads, kWarpThreadContiguous, kWarpThreadStrided,
          kElementsPerAccess)) {
    return false;
  }
  int const shape_access_contiguous = cta_n / kElementsPerAccess;
  int const shape_access_strided = cta_k;
  int const warp_access_contiguous =
      shape_access_contiguous / kWarpThreadContiguous;
  int const warp_access_strided = shape_access_strided / kWarpThreadStrided;
  int const warp_count = threads / 32;
  int const warps_strided =
      warp_access_strided >= warp_count ? warp_count : warp_access_strided;
  int const warps_contiguous =
      warp_count > warp_access_strided ? warp_count / warps_strided : 1;
  int const iterations_contiguous = warp_access_contiguous / warps_contiguous;
  int const delta_contiguous = kWarpThreadContiguous * kElementsPerAccess;
  return iterations_contiguous == cta_n / kQuantTileN &&
         delta_contiguous == kQuantTileN;
}

constexpr bool sm70_marlin_volta_epilogue_supported(int cta_m, int cta_n,
                                                    int warps, int warp_m) {
  constexpr int kShapeRow = 4;
  constexpr int kShapeGroup = 4;
  constexpr int kElementsPerAccess = 8;
  constexpr int kElementBits = 16;
  constexpr int kWarpSize = 32;
  int const shape_cluster = cta_m / warp_m;
  if (shape_cluster <= 0) {
    return false;
  }
  int const warps_remaining_for_groups =
      shape_cluster > warps ? 1 : warps / shape_cluster;
  if (warps_remaining_for_groups <= 0) {
    return false;
  }
  int const warps_remaining_for_rows =
      kShapeGroup > warps_remaining_for_groups
          ? 1
          : warps_remaining_for_groups / kShapeGroup;
  if (warps_remaining_for_rows <= 0) {
    return false;
  }

  if (kShapeRow <= warps_remaining_for_rows) {
    return cta_n / kElementsPerAccess / kWarpSize > 0;
  }

  int const row_shape = kShapeRow / warps_remaining_for_rows;
  if (row_shape <= 0) {
    return false;
  }
  int const shape_width = cta_n / kElementsPerAccess;
  int const target_memory_access_width =
      256 / (kElementsPerAccess * kElementBits / 8);
  int const target_access_rows = kWarpSize / target_memory_access_width;
  int const access_width =
      target_access_rows > row_shape
          ? kWarpSize / row_shape
          : sm70_const_min(
                shape_width,
                sm70_const_min(kWarpSize, target_memory_access_width));
  if (access_width <= 0) {
    return false;
  }
  int const access_rows =
      target_access_rows > row_shape
          ? row_shape
          : sm70_const_min(kShapeRow, kWarpSize / access_width);
  if (access_rows <= 0) {
    return false;
  }
  return row_shape / access_rows > 0 && shape_width / access_width > 0;
}

constexpr bool sm70_marlin_cta_geometry_is_supported_constexpr(
    int cta_m, int cta_n, int cta_k, int warps, int warp_m, int warp_n,
    int warp_k) {
  if (!sm70_marlin_warp_decomposition_supported(cta_m, cta_n, cta_k, warps,
                                                warp_m, warp_n, warp_k)) {
    return false;
  }
  int const threads = warps * 32;
  // Current SM70 Marlin uses CUTLASS' Volta row-major/row-major
  // DefaultMmaCore. Its A/B thread maps are fixed to 128-bit half loads.
  // CTA_K=16 is a mathematical GEMM shape but does not provide enough
  // contiguous A accesses for that thread map.
  if (!sm70_marlin_pitchlinear_threadmap_supported(cta_k, cta_m, threads, 4, 8,
                                                   8)) {
    return false;
  }
  if (!sm70_marlin_b_threadmap_matches_quant_iterator(cta_n, cta_k, threads)) {
    return false;
  }
  return sm70_marlin_volta_epilogue_supported(cta_m, cta_n, warps, warp_m);
}

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK>
struct Sm70WarpShape {
  static_assert(CtaM == 32 || CtaM == 64 || CtaM == 128 || CtaM == 256,
                "SM70 Marlin supports CTA_M in {32, 64, 128, 256}.");
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 Marlin supports CTA_N in {64, 128, 256}.");
  static_assert(CtaK == 16 || CtaK == 32 || CtaK == 64 || CtaK == 128,
                "SM70 Marlin supports CTA_K in {16, 32, 64, 128}.");
  static_assert(Warps == 4 || Warps == 8, "SM70 Marlin supports 4 or 8 warps.");
  static_assert(WarpM == 32 || WarpM == 64,
                "SM70 Marlin supports Warp_M in {32, 64}.");
  static_assert(WarpN == 32 || WarpN == 64,
                "SM70 Marlin supports Warp_N in {32, 64}.");
  static_assert(WarpK == 16 || WarpK == 32,
                "SM70 Marlin supports Warp_K in {16, 32}.");
  static_assert(CtaM % WarpM == 0,
                "SM70 Marlin requires CTA_M divisible by Warp_M.");
  static_assert(CtaN % WarpN == 0,
                "SM70 Marlin requires CTA_N divisible by Warp_N.");
  static_assert(CtaK % WarpK == 0,
                "SM70 Marlin requires CTA_K divisible by Warp_K.");
  static_assert((CtaM / WarpM) * (CtaN / WarpN) * (CtaK / WarpK) == Warps,
                "SM70 Marlin explicit warp shape must decompose the CTA into "
                "the requested warp count.");
  static_assert(sm70_marlin_cta_geometry_is_supported_constexpr(
                    CtaM, CtaN, CtaK, Warps, WarpM, WarpN, WarpK),
                "SM70 Marlin CTA geometry is not supported by the current "
                "CUTLASS Volta thread maps, epilogue, and quantized B "
                "iterator contract.");

  using Type = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
};

struct Sm70CtaGeometry {
  int cta_m;
  int cta_n;
  int cta_k;
  int warps;
  int warp_m;
  int warp_n;
  int warp_k;
};

struct Sm70MarlinAutoParams {
  Sm70CtaGeometry geometry;
  int requested_split_k;
  bool use_metadata_vector_words;
  int packed_macro_n;
};

struct Sm70MarlinDenseAutoParamsContext {
  char const* quant_format;
  int64_t group_size;
  int64_t size_m;
  int64_t size_n;
  int64_t size_k;
  int packed_macro_n;
};

inline constexpr char const* kSupportedSm70MarlinCtaGeometries =
    "{CTA_M}x{CTA_N}x{CTA_K}x{Warps}x{WarpM}x{WarpN}x{WarpK} "
    "with CTA_M in {32,64,128,256}, CTA_N in {64,128,256}, "
    "CTA_K in {16,32,64,128}, Warps in {4,8}, WarpM in {32,64}, "
    "WarpN in {32,64}, WarpK in {16,32}, a valid CTA/warp "
    "decomposition, current CUTLASS Volta thread-map support, and "
    "phase-aware SM70 Marlin warp-K offset support.";

inline bool sm70_marlin_cta_geometry_is_supported(Sm70CtaGeometry geometry) {
  return sm70_marlin_cta_geometry_is_supported_constexpr(
      geometry.cta_m, geometry.cta_n, geometry.cta_k, geometry.warps,
      geometry.warp_m, geometry.warp_n, geometry.warp_k);
}

inline int sm70_marlin_auto_packed_macro_n(int64_t size_n) {
  if (size_n % 256 == 0) {
    return 256;
  }
  if (size_n % 128 == 0) {
    return 128;
  }
  if (size_n % 64 == 0) {
    return 64;
  }
  STD_TORCH_CHECK(false,
                  "SM70 Marlin requires size_n divisible "
                  "by 64. Got size_n = ",
                  size_n, ".");
  return 0;
}

inline bool sm70_marlin_packed_macro_n_is_supported(int packed_macro_n) {
  return packed_macro_n == 64 || packed_macro_n == 128 || packed_macro_n == 256;
}

inline bool sm70_marlin_requested_split_k_is_supported(int requested_split_k) {
  return requested_split_k == 1 || requested_split_k == 2 ||
         requested_split_k == 4 || requested_split_k == 8;
}

inline Sm70MarlinAutoParams sm70_marlin_default_auto_params(int packed_macro_n);

inline Sm70CtaGeometry sm70_marlin_default_geometry_from_packed_macro_n(
    int packed_macro_n) {
  switch (packed_macro_n) {
    case 64:
      return {64, 64, 32, 4, 32, 32, 32};
    case 128:
      return {32, 128, 32, 4, 32, 32, 32};
    case 256:
      return {32, 256, 32, 4, 32, 64, 32};
    default:
      STD_TORCH_CHECK(false,
                      "Unsupported SM70 Marlin packed macro-N=", packed_macro_n,
                      ".");
  }
  return {0, 0, 0, 0, 0, 0, 0};
}

inline Sm70MarlinAutoParams sm70_marlin_default_auto_params(
    int packed_macro_n) {
  return {sm70_marlin_default_geometry_from_packed_macro_n(packed_macro_n), 1,
          true, packed_macro_n};
}

inline bool sm70_marlin_parse_int_component(char const*& cursor, int& value) {
  char* end = nullptr;
  long parsed = std::strtol(cursor, &end, 10);
  if (end == cursor) {
    return false;
  }
  value = static_cast<int>(parsed);
  cursor = end;
  return true;
}

inline bool sm70_marlin_parse_geometry_env_value(char const* value,
                                                 Sm70CtaGeometry& geometry) {
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  char const* cursor = value;
  int fields[7] = {0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 7; ++i) {
    if (!sm70_marlin_parse_int_component(cursor, fields[i])) {
      return false;
    }
    if (i < 6) {
      if (*cursor != 'x') {
        return false;
      }
      ++cursor;
    }
  }
  if (*cursor != '\0') {
    return false;
  }
  geometry = {fields[0], fields[1], fields[2], fields[3],
              fields[4], fields[5], fields[6]};
  return true;
}

inline bool sm70_marlin_try_get_geometry_env(char const* env_name,
                                             Sm70CtaGeometry& geometry) {
  char const* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  STD_TORCH_CHECK(
      sm70_marlin_parse_geometry_env_value(value, geometry), "Invalid ",
      env_name, " value '", value,
      "'. Expected {CTA_M}x{CTA_N}x{CTA_K}x{Warps}x{WarpM}x{WarpN}x{WarpK}.");
  return true;
}

inline bool sm70_marlin_try_get_split_k_env(char const* env_name,
                                           int& requested_split_k) {
  char const* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  char const* cursor = value;
  int parsed = 0;
  STD_TORCH_CHECK(sm70_marlin_parse_int_component(cursor, parsed) &&
                      *cursor == '\0' &&
                      sm70_marlin_requested_split_k_is_supported(parsed),
                  "Invalid ", env_name, " value '", value,
                  "'. Expected one of 1, 2, 4, or 8.");
  requested_split_k = parsed;
  return true;
}

inline bool sm70_marlin_try_get_metadata_env(char const* env_name,
                                             bool& use_metadata_vector_words) {
  char const* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  if (std::strcmp(value, "vector_words") == 0) {
    use_metadata_vector_words = true;
    return true;
  }
  if (std::strcmp(value, "lane_vectors") == 0) {
    use_metadata_vector_words = false;
    return true;
  }
  STD_TORCH_CHECK(false, "Invalid ", env_name, " value '", value,
                  "'. Expected vector_words or lane_vectors.");
  return false;
}

inline bool sm70_marlin_env_is_set(char const* env_name) {
  char const* value = std::getenv(env_name);
  return value != nullptr && value[0] != '\0';
}

inline bool sm70_marlin_any_auto_env_is_set(char const* geometry_env_name,
                                            char const* split_k_env_name,
                                            char const* metadata_env_name) {
  return sm70_marlin_env_is_set(geometry_env_name) ||
         sm70_marlin_env_is_set(split_k_env_name) ||
         sm70_marlin_env_is_set(metadata_env_name);
}

inline void validate_sm70_marlin_packed_macro_n(char const* op_name,
                                               Sm70CtaGeometry geometry,
                                               int packed_macro_n) {
  STD_TORCH_CHECK(sm70_marlin_packed_macro_n_is_supported(packed_macro_n),
                  "Unsupported SM70 Marlin packed macro-N for ", op_name, ": ",
                  packed_macro_n, ".");
  STD_TORCH_CHECK(packed_macro_n % geometry.cta_n == 0, "SM70 Marlin ", op_name,
                  " requires PackedMacroN divisible by CTA_N. Got PackedMacroN=",
                  packed_macro_n, ", CTA_N=", geometry.cta_n, ".");
}

inline void validate_sm70_marlin_auto_params(char const* op_name,
                                             Sm70MarlinAutoParams params) {
  STD_TORCH_CHECK(sm70_marlin_cta_geometry_is_supported(params.geometry),
                  "Unsupported SM70 Marlin CTA geometry for ", op_name, ": ",
                  params.geometry.cta_m, "x", params.geometry.cta_n, "x",
                  params.geometry.cta_k, "x", params.geometry.warps, "x",
                  params.geometry.warp_m, "x", params.geometry.warp_n, "x",
                  params.geometry.warp_k, ". Supported geometries are ",
                  kSupportedSm70MarlinCtaGeometries, ".");
  STD_TORCH_CHECK(
      sm70_marlin_requested_split_k_is_supported(params.requested_split_k),
      "Unsupported SM70 Marlin requested split-K for ", op_name, ": ",
      params.requested_split_k, ". Expected one of 1, 2, 4, or 8.");
  validate_sm70_marlin_packed_macro_n(op_name, params.geometry,
                                      params.packed_macro_n);
}

inline Sm70MarlinAutoParams sm70_marlin_auto_params_from_env(
    char const* op_name, char const* geometry_env_name,
    char const* split_k_env_name, char const* metadata_env_name,
    int packed_macro_n) {
  Sm70MarlinAutoParams params = sm70_marlin_default_auto_params(packed_macro_n);
  sm70_marlin_try_get_geometry_env(geometry_env_name, params.geometry);
  sm70_marlin_try_get_split_k_env(split_k_env_name, params.requested_split_k);
  sm70_marlin_try_get_metadata_env(metadata_env_name,
                                   params.use_metadata_vector_words);
  validate_sm70_marlin_auto_params(op_name, params);
  return params;
}

inline bool sm70_marlin_dense_auto_env_is_set() {
  return sm70_marlin_any_auto_env_is_set("SM70_MARLIN_DENSE_CTA_GEOMETRY",
                                         "SM70_MARLIN_DENSE_SPLIT_K",
                                         "SM70_MARLIN_DENSE_METADATA_CACHE");
}

inline Sm70MarlinAutoParams sm70_marlin_dense_auto_params_from_env(
    Sm70MarlinDenseAutoParamsContext const& ctx) {
  return sm70_marlin_auto_params_from_env(
      "Dense", "SM70_MARLIN_DENSE_CTA_GEOMETRY", "SM70_MARLIN_DENSE_SPLIT_K",
      "SM70_MARLIN_DENSE_METADATA_CACHE", ctx.packed_macro_n);
}

inline bool sm70_marlin_dense_try_select_quanttrio_qwen3_6_27b_awq_params(
    Sm70MarlinDenseAutoParamsContext const& ctx, Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "uint4") != 0 || ctx.group_size != 128 ||
      ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry, int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (ctx.size_n == 4352 && ctx.size_k == 5120) {
    if (ctx.size_m <= 16) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 48) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m < 128) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 4, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 768) {
    if (ctx.size_m <= 16) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, true);
    }
    if (ctx.size_m <= 48) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, true);
    }
    if (ctx.size_m < 128) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 1536) {
    if (ctx.size_m <= 16) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, false);
    }
    if (ctx.size_m <= 48) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m < 128) {
      return set_params({64, 64, 64, 4, 64, 64, 16}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 2176) {
    if (ctx.size_m <= 16) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 48) {
      return set_params({32, 64, 128, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.size_m < 128) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 4352) {
    if (ctx.size_m <= 16) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 48) {
      return set_params({32, 256, 64, 8, 32, 64, 32}, 4, true);
    }
    if (ctx.size_m < 128) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 8704 && ctx.size_k == 5120) {
    if (ctx.size_m <= 48) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m < 128) {
      return set_params({64, 128, 128, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  return false;
}

inline bool
sm70_marlin_dense_try_select_cyankiwi_qwen3_6_27b_awq_bf16_nvfp4_params(
    Sm70MarlinDenseAutoParamsContext const& ctx, Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "nvfp4") != 0 || ctx.group_size != 16 ||
      ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry, int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (ctx.size_n == 2048 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 64, 4, 32, 64, 32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  if (ctx.size_n == 3584 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 4352 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 4, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 768) {
    if (ctx.size_m <= 1) {
      return set_params({64, 64, 64, 4, 32, 64, 32}, 1, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 1536) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 1, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 2176) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 128, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 4352) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 128, 64, 8, 32, 32, 32}, 2, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 8704 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 128, 128, 8, 64, 64, 32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  return false;
}

inline bool sm70_marlin_dense_try_select_nvidia_glm_4_7_nvfp4_params(
    Sm70MarlinDenseAutoParamsContext const& ctx, Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "nvfp4") != 0 || ctx.group_size != 16 ||
      ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry, int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (ctx.size_n == 384 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 8, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64, 64, 64, 4, 64, 32, 32}, 1, true);
    }
    return set_params({64, 128, 32, 4, 32, 64, 32}, 1, true);
  }

  if (ctx.size_n == 768 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 8, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 64, 32, 4, 32, 64, 32}, 1, true);
    }
    return set_params({64, 128, 32, 4, 32, 64, 32}, 1, true);
  }

  if (ctx.size_n == 3072 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 128, 64, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 192) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 128, 32, 4, 64, 64, 32}, 1, true);
    }
    return set_params({128, 128, 32, 4, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 384) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 1536) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 3072) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 2, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 6144 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 128, 32, 4, 32, 64, 32}, 4, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  return false;
}

inline bool sm70_marlin_dense_try_select_nvidia_qwen3_6_35b_a3b_nvfp4_params(
    Sm70MarlinDenseAutoParamsContext const& ctx, Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr || ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry, int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (std::strcmp(ctx.quant_format, "fp8_e4m3") == 0 && ctx.group_size == 512 &&
      ctx.size_n == 2048 && ctx.size_k == 512) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (std::strcmp(ctx.quant_format, "fp8_e4m3") == 0 &&
      ctx.group_size == 1024 && ctx.size_n == 2048 && ctx.size_k == 1024) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (std::strcmp(ctx.quant_format, "fp8_e4m3") == 0 &&
      ctx.group_size == 2048 && ctx.size_n == 1536 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64, 128, 32, 4, 32, 64, 32}, 1, false);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (std::strcmp(ctx.quant_format, "fp8_e4m3") == 0 &&
      ctx.group_size == 2048 && ctx.size_n == 2560 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 2, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (std::strcmp(ctx.quant_format, "nvfp4") == 0 && ctx.group_size == 16 &&
      ctx.size_n == 128 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64, 64, 64, 4, 32, 64, 32}, 1, true);
    }
    return set_params({128, 64, 128, 8, 64, 64, 32}, 1, true);
  }

  if (std::strcmp(ctx.quant_format, "nvfp4") == 0 && ctx.group_size == 16 &&
      ctx.size_n == 256 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 64, 128, 8, 64, 64, 32}, 1, false);
    }
    return set_params({128, 64, 32, 4, 32, 64, 32}, 1, false);
  }

  if (std::strcmp(ctx.quant_format, "nvfp4") == 0 && ctx.group_size == 16 &&
      ctx.size_n == 2048 && ctx.size_k == 64) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
    }
    return set_params({128, 64, 32, 4, 32, 64, 32}, 1, false);
  }

  if (std::strcmp(ctx.quant_format, "nvfp4") == 0 && ctx.group_size == 16 &&
      ctx.size_n == 2048 && ctx.size_k == 128) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 64, 32, 4, 32, 64, 32}, 1, true);
    }
    return set_params({128, 64, 32, 4, 32, 64, 32}, 1, true);
  }

  return false;
}

inline bool
sm70_marlin_dense_try_select_nvidia_qwen3_next_80b_a3b_thinking_nvfp4_params(
    Sm70MarlinDenseAutoParamsContext const& ctx, Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "nvfp4") != 0 || ctx.group_size != 16 ||
      ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry, int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (ctx.size_n == 128 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64, 64, 64, 4, 32, 64, 32}, 1, true);
    }
    return set_params({128, 64, 128, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 256 && ctx.size_k == 2048) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 64, 128, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 64, 32, 4, 32, 64, 32}, 1, true);
  }

  if (ctx.size_n == 2048 && ctx.size_k == 64) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 1, true);
    }
    return set_params({128, 64, 32, 4, 32, 64, 32}, 1, true);
  }

  if (ctx.size_n == 2048 && ctx.size_k == 128) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 64, 32, 4, 32, 64, 32}, 1, true);
    }
    return set_params({128, 64, 32, 4, 32, 64, 32}, 1, false);
  }

  if (ctx.size_n == 2048 && ctx.size_k == 512) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 1, false);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 128, 32, 4, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 2048 && ctx.size_k == 1024) {
    if (ctx.size_m <= 1) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  return false;
}

inline bool sm70_marlin_dense_try_select_cyankiwi_minimax_m2_7_awq_4bit_params(
    Sm70MarlinDenseAutoParamsContext const& ctx, Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "uint4b8") != 0 || ctx.group_size != 32 ||
      ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry, int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  // size_n=1024, size_k=3072  (tp8 qkv_proj)
  if (ctx.size_n == 1024 && ctx.size_k == 3072) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=2048, size_k=3072  (tp4 qkv_proj)
  if (ctx.size_n == 2048 && ctx.size_k == 3072) {
    if (ctx.size_m <= 1) {
      return set_params({32, 64, 32, 4, 32, 32, 16}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({64, 64, 32, 4, 64, 32, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=3072, size_k=768  (tp8 o_proj)
  if (ctx.size_n == 3072 && ctx.size_k == 768) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 64, 4, 32, 64, 32}, 1, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 64, 4, 32, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=3072, size_k=1536  (tp4 o_proj)
  if (ctx.size_n == 3072 && ctx.size_k == 1536) {
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 64, 4, 64, 64, 16}, 1, false);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  return false;
}

inline bool sm70_marlin_dense_try_select_cyankiwi_glm_4_7_awq_4bit_params(
    Sm70MarlinDenseAutoParamsContext const& ctx, Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "uint4b8") != 0 || ctx.group_size != 32 ||
      ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry, int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  // size_n=1792, size_k=5120  (tp8 qkv_proj)
  if (ctx.size_n == 1792 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 128, 64, 8, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 2048) {
      return set_params({64, 128, 32, 4, 64, 32, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=3584, size_k=5120  (tp4 qkv_proj)
  if (ctx.size_n == 3584 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 64, 4, 32, 64, 32}, 4, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=5120, size_k=1536  (tp8 o_proj)
  if (ctx.size_n == 5120 && ctx.size_k == 1536) {
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 1, false);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  // size_n=5120, size_k=3072  (tp4 o_proj)
  if (ctx.size_n == 5120 && ctx.size_k == 3072) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 128, 64, 8, 32, 32, 32}, 2, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, false);
  }

  return false;
}

inline bool sm70_marlin_dense_try_select_quanttrio_glm_4_7_awq_params(
    Sm70MarlinDenseAutoParamsContext const& ctx, Sm70MarlinAutoParams& params) {
  if (ctx.quant_format == nullptr ||
      std::strcmp(ctx.quant_format, "uint4") != 0 || ctx.group_size != 128 ||
      ctx.size_m <= 0) {
    return false;
  }

  auto const set_params = [&](Sm70CtaGeometry geometry, int requested_split_k,
                              bool use_metadata_vector_words) {
    params = {geometry, requested_split_k, use_metadata_vector_words,
              ctx.packed_macro_n};
    return true;
  };

  if (ctx.size_n == 1792 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 4, false);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 3072 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 4, false);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 3584 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 64, 16}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 32, 4, 32, 32, 32}, 4, false);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 1536) {
    if (ctx.size_m <= 1) {
      return set_params({64, 128, 32, 4, 32, 64, 32}, 4, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 64, 64, 4, 32, 32, 32}, 1, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 64, 4, 64, 64, 16}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 5120 && ctx.size_k == 3072) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 8, false);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 2, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({32, 128, 128, 8, 32, 64, 32}, 1, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  if (ctx.size_n == 6144 && ctx.size_k == 5120) {
    if (ctx.size_m <= 1) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 32) {
      return set_params({32, 128, 32, 4, 32, 32, 32}, 4, true);
    }
    if (ctx.size_m <= 64) {
      return set_params({64, 64, 64, 4, 32, 32, 32}, 2, true);
    }
    return set_params({128, 256, 32, 8, 64, 64, 32}, 1, true);
  }

  return false;
}

inline Sm70MarlinAutoParams sm70_marlin_dense_auto_params(
    char const* quant_format, int64_t group_size, int64_t size_m,
    int64_t size_n, int64_t size_k) {
  Sm70MarlinDenseAutoParamsContext const ctx{
      quant_format, group_size, size_m,
      size_n,       size_k,     sm70_marlin_auto_packed_macro_n(size_n)};

  if (sm70_marlin_dense_auto_env_is_set()) {
    return sm70_marlin_dense_auto_params_from_env(ctx);
  }

  Sm70MarlinAutoParams params{};
  if (sm70_marlin_dense_try_select_quanttrio_qwen3_6_27b_awq_params(ctx,
                                                                    params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_cyankiwi_qwen3_6_27b_awq_bf16_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_nvidia_glm_4_7_nvfp4_params(ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_nvidia_qwen3_6_35b_a3b_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_nvidia_qwen3_next_80b_a3b_thinking_nvfp4_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_cyankiwi_minimax_m2_7_awq_4bit_params(
          ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_cyankiwi_glm_4_7_awq_4bit_params(ctx,
                                                                    params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }
  if (sm70_marlin_dense_try_select_quanttrio_glm_4_7_awq_params(ctx, params)) {
    validate_sm70_marlin_auto_params("Dense", params);
    return params;
  }

  return sm70_marlin_default_auto_params(ctx.packed_macro_n);
}

inline void validate_sm70_marlin_dense_cta_geometry_supported(
    char const* op_name, Sm70CtaGeometry geometry) {
  STD_TORCH_CHECK(sm70_marlin_cta_geometry_is_supported(geometry),
                  "Unsupported SM70 Marlin CTA geometry for ", op_name, ": ",
                  geometry.cta_m, "x", geometry.cta_n, "x", geometry.cta_k,
                  "x", geometry.warps, "x", geometry.warp_m, "x",
                  geometry.warp_n, "x", geometry.warp_k, ". Supported "
                  "geometries are ",
                  kSupportedSm70MarlinCtaGeometries, ".");
}

inline void validate_sm70_marlin_dense_cta_n_alignment(char const* op_name,
                                                       Sm70CtaGeometry geometry,
                                                       int64_t size_n) {
  STD_TORCH_CHECK(size_n % geometry.cta_n == 0 && size_n % kQuantTileN == 0,
                  "SM70 Marlin requires size_n divisible by both CTA_N and 64 "
                  "for ",
                  op_name, " with CTA geometry ", geometry.cta_m, "x",
                  geometry.cta_n, "x", geometry.cta_k, "x", geometry.warps,
                  "x", geometry.warp_m, "x", geometry.warp_n, "x",
                  geometry.warp_k,
                  ". Got size_n = ", size_n, ".");
}

}  // namespace marlin::sm70

// =============================================================================
// Iterator utilities (from 1Cat sm70_marlin_iterator_utils.cuh)
// =============================================================================
namespace marlin::sm70 {

template <int ContiguousWords>
struct QwordVector;

template <>
struct QwordVector<1> {
  using Type = uint32_t;
};

template <>
struct QwordVector<2> {
  using Type = uint2;
};

template <>
struct QwordVector<4> {
  using Type = uint4;
};

template <int ContiguousWords>
CUTLASS_DEVICE typename QwordVector<ContiguousWords>::Type load_qword_vector(
    uint32_t const* ptr) {
  static_assert(
      ContiguousWords == 1 || ContiguousWords == 2 || ContiguousWords == 4,
      "SM70 qword vector load supports 1, 2, or 4 words.");
  if constexpr (ContiguousWords == 1) {
    return *ptr;
  } else {
    return *reinterpret_cast<
        typename QwordVector<ContiguousWords>::Type const*>(ptr);
  }
}

template <int ContiguousWords>
CUTLASS_DEVICE uint32_t qword_from_vector(
    typename QwordVector<ContiguousWords>::Type const& words, int c) {
  static_assert(
      ContiguousWords == 1 || ContiguousWords == 2 || ContiguousWords == 4,
      "SM70 qword vector access supports 1, 2, or 4 words.");
  if constexpr (ContiguousWords == 1) {
    return words;
  } else {
    uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
    return words_ptr[c];
  }
}

CUTLASS_DEVICE uint32_t qword_from_vector(uint4 const& words, int c) {
  uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
  return words_ptr[c];
}

CUTLASS_DEVICE uint32_t qword_from_vector(uint2 const& words, int c) {
  uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
  return words_ptr[c];
}

template <int PackedMacroN>
CUTLASS_DEVICE int u4_packed_macro_n_qweight_offset_from_logical(
    int size_n, int logical_k, int logical_n) {
  static_assert(
      PackedMacroN == 64 || PackedMacroN == 128 || PackedMacroN == 256,
      "SM70 Marlin packed macro-N must be 64, 128, or 256.");
  constexpr int kGroupTiles = PackedMacroN / kQuantTileN;
  int const k_tile = logical_k / kQuantTileK;
  int const local_k = logical_k - k_tile * kQuantTileK;
  int const n_tile = logical_n / kQuantTileN;
  int const group_n_tile = n_tile / kGroupTiles;
  int const group_first_n_tile = group_n_tile * kGroupTiles;
  int const subtile = n_tile - group_first_n_tile;
  int const local_n_vec = (logical_n - n_tile * kQuantTileN) / 8;
  int const local_word = local_k * (kQuantTileN / 8) + local_n_vec;

  return k_tile * (size_n * 2) +
         group_n_tile * kGroupTiles * (kQuantTileK * kQuantTileN / 8) +
         local_word * kGroupTiles + subtile;
}

template <int PackedMacroN>
CUTLASS_DEVICE int u8_packed_macro_n_qweight_offset_from_logical(
    int size_n, int logical_k, int logical_n) {
  static_assert(
      PackedMacroN == 64 || PackedMacroN == 128 || PackedMacroN == 256,
      "SM70 Marlin packed macro-N must be 64, 128, or 256.");
  constexpr int kGroupTiles = PackedMacroN / kQuantTileN;
  int const k_tile = logical_k / kQuantTileK;
  int const local_k = logical_k - k_tile * kQuantTileK;
  int const n_tile = logical_n / kQuantTileN;
  int const group_n_tile = n_tile / kGroupTiles;
  int const group_first_n_tile = group_n_tile * kGroupTiles;
  int const subtile = n_tile - group_first_n_tile;
  int const local_n_word = (logical_n - n_tile * kQuantTileN) / 4;
  int const local_word = local_k * (kQuantTileN / 4) + local_n_word;

  return k_tile * (size_n * 4) +
         group_n_tile * kGroupTiles * (kQuantTileK * kQuantTileN / 4) +
         local_word * kGroupTiles + subtile;
}

template <int PackedMacroN>
CUTLASS_DEVICE int u8_packed_macro_n_qweight_word_stride() {
  static_assert(
      PackedMacroN == 64 || PackedMacroN == 128 || PackedMacroN == 256,
      "SM70 Marlin packed macro-N must be 64, 128, or 256.");
  return PackedMacroN / kQuantTileN;
}

}  // namespace marlin::sm70

// =============================================================================
// Split-K partitioning + atomic epilogues (from sm70_marlin_splitk.cuh)
// =============================================================================
namespace marlin::sm70 {

struct Sm70SplitKPartition {
  int k_begin;
  int partition_k;
};

CUTLASS_HOST_DEVICE
int sm70_splitk_ceil_div_int(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

CUTLASS_HOST_DEVICE
int sm70_splitk_min_int(int lhs, int rhs) { return lhs < rhs ? lhs : rhs; }

template <int GroupSize, int CtaK = kCtaK>
CUTLASS_HOST_DEVICE int sm70_splitk_group_tiles() {
  if constexpr (GroupSize > 0) {
    if constexpr (GroupSize >= CtaK) {
      static_assert(GroupSize % CtaK == 0,
                    "SM70 Marlin split-K group size must be CTA_K aligned.");
      return GroupSize / CtaK;
    } else {
      return 1;
    }
  } else {
    return 1;
  }
}

CUTLASS_HOST_DEVICE
int sm70_active_split_k(int k, int requested_split_k, int cta_k = kCtaK) {
  int const total_tiles = k / cta_k;
  if (total_tiles <= 0) {
    return 0;
  }
  return sm70_splitk_min_int(requested_split_k, total_tiles);
}

CUTLASS_HOST_DEVICE
int sm70_splitk_partition_tile_count(int remaining_tiles,
                                     int remaining_partitions,
                                     int group_tiles) {
  if (remaining_tiles <= 0 || remaining_partitions <= 0) {
    return 0;
  }
  if (remaining_partitions == 1) {
    return remaining_tiles;
  }

  int const target_tiles =
      sm70_splitk_ceil_div_int(remaining_tiles, remaining_partitions);
  int const max_current_tiles = remaining_tiles - (remaining_partitions - 1);
  int partition_tiles = target_tiles;
  if (group_tiles > 1) {
    int const rounded_tiles =
        sm70_splitk_ceil_div_int(partition_tiles, group_tiles) * group_tiles;
    if (rounded_tiles <= max_current_tiles) {
      partition_tiles = rounded_tiles;
    }
  }
  return sm70_splitk_min_int(partition_tiles, max_current_tiles);
}

template <int GroupSize, int CtaK = kCtaK>
CUTLASS_HOST_DEVICE Sm70SplitKPartition
sm70_splitk_partition(int k, int requested_split_k, int partition_idx) {
  int const active_split_k = sm70_active_split_k(k, requested_split_k, CtaK);
  if (partition_idx >= active_split_k) {
    return {0, 0};
  }

  int const group_tiles = sm70_splitk_group_tiles<GroupSize, CtaK>();
  int remaining_tiles = k / CtaK;
  int start_tiles = 0;
  for (int idx = 0; idx < partition_idx; ++idx) {
    int const partition_tiles = sm70_splitk_partition_tile_count(
        remaining_tiles, active_split_k - idx, group_tiles);
    start_tiles += partition_tiles;
    remaining_tiles -= partition_tiles;
  }

  int const partition_tiles = sm70_splitk_partition_tile_count(
      remaining_tiles, active_split_k - partition_idx, group_tiles);
  return {start_tiles * CtaK, partition_tiles * CtaK};
}

template <typename Traits>
class Sm70AtomicFp16Epilogue {
 public:
  using CutlassEpilogue = typename Traits::Epilogue;
  using SharedStorage = typename CutlassEpilogue::Base::SharedStorage;
  using AccumulatorTile = typename CutlassEpilogue::AccumulatorTile;
  using AccumulatorFragmentIterator =
      typename CutlassEpilogue::AccumulatorFragmentIterator;
  using WarpTileIterator = typename CutlassEpilogue::WarpTileIterator;
  using SharedLoadIterator = typename CutlassEpilogue::SharedLoadIterator;
  using OutputTileIterator = typename CutlassEpilogue::OutputTileIterator;
  using ThreadMap = typename OutputTileIterator::ThreadMap;

 private:
  WarpTileIterator warp_tile_iterator_;
  SharedLoadIterator shared_load_iterator_;

  CUTLASS_DEVICE
  void atomic_store_fragment(OutputTileIterator const& destination_iterator,
                             typename SharedLoadIterator::Fragment const& frag,
                             cutlass::half_t* __restrict__ c, int n) const {
    float const* frag_ptr = reinterpret_cast<float const*>(&frag);
    half* c_half = reinterpret_cast<half*>(c);
    int const thread_start_row = destination_iterator.thread_start_row();
    int const thread_start_column = destination_iterator.thread_start_column();
    int const extent_row = destination_iterator.extent_row();

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
         ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int const frag_row_idx =
              row + ThreadMap::Iterations::kRow *
                        (group + ThreadMap::Iterations::kGroup * cluster);
          int const row_offset = row * ThreadMap::Delta::kRow +
                                 group * ThreadMap::Delta::kGroup +
                                 cluster * ThreadMap::Delta::kCluster;
          int const logical_row = thread_start_row + row_offset;

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            int const logical_column_base =
                thread_start_column + column * ThreadMap::Delta::kColumn;
            int const frag_base =
                (frag_row_idx * ThreadMap::Iterations::kColumn + column) *
                ThreadMap::kElementsPerAccess;

            if (logical_row < extent_row) {
              CUTLASS_PRAGMA_UNROLL
              for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
                int64_t const offset =
                    int64_t(logical_row) * n + logical_column_base + e;
                atomicAdd(c_half + offset,
                          __float2half_rn(frag_ptr[frag_base + e]));
              }
            }
          }
        }
      }
    }
  }

 public:
  CUTLASS_DEVICE
  Sm70AtomicFp16Epilogue(SharedStorage& shared_storage, int thread_idx,
                         int warp_idx, int lane_idx)
      : warp_tile_iterator_(shared_storage.reference(), lane_idx),
        shared_load_iterator_(shared_storage.reference(), thread_idx) {
    using WarpCount = typename CutlassEpilogue::WarpCount;
    int const warp_k = warp_idx / (WarpCount::kM * WarpCount::kN);
    int const warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int const warp_m = warp_mn % WarpCount::kM;
    int const warp_n = warp_mn / WarpCount::kM;

    cutlass::MatrixCoord warp_offset{warp_k * WarpCount::kM + warp_m, warp_n};
    warp_tile_iterator_.add_tile_offset(warp_offset);
  }

  CUTLASS_DEVICE
  void operator()(OutputTileIterator destination_iterator,
                  AccumulatorTile const& accumulators,
                  cutlass::half_t* __restrict__ c, int n) {
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      __syncthreads();

      typename AccumulatorFragmentIterator::Fragment accum_fragment;
      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;
      warp_tile_iterator_.store(accum_fragment);

      __syncthreads();

      typename SharedLoadIterator::Fragment aligned_accum_fragment;
      shared_load_iterator_.load(aligned_accum_fragment);

      if (CutlassEpilogue::kPartitionsK > 1) {
        cutlass::plus<typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < CutlassEpilogue::kPartitionsK; ++i) {
          typename SharedLoadIterator::Fragment aligned_addend_fragment;
          shared_load_iterator_.add_pointer_offset(
              CutlassEpilogue::kSmemPointerOffset);
          shared_load_iterator_.load(aligned_addend_fragment);
          aligned_accum_fragment =
              add_fragments(aligned_accum_fragment, aligned_addend_fragment);
        }

        shared_load_iterator_.add_pointer_offset(
            (1 - CutlassEpilogue::kPartitionsK) *
            CutlassEpilogue::kSmemPointerOffset);
      }

      atomic_store_fragment(destination_iterator, aligned_accum_fragment, c, n);
      ++destination_iterator;
    }
  }
};

template <typename Traits>
class Sm70AtomicFp32Epilogue {
 public:
  using CutlassEpilogue = typename Traits::Epilogue;
  using SharedStorage = typename CutlassEpilogue::Base::SharedStorage;
  using AccumulatorTile = typename CutlassEpilogue::AccumulatorTile;
  using AccumulatorFragmentIterator =
      typename CutlassEpilogue::AccumulatorFragmentIterator;
  using WarpTileIterator = typename CutlassEpilogue::WarpTileIterator;
  using SharedLoadIterator = typename CutlassEpilogue::SharedLoadIterator;
  using OutputTileIterator = typename CutlassEpilogue::OutputTileIterator;
  using ThreadMap = typename OutputTileIterator::ThreadMap;

 private:
  WarpTileIterator warp_tile_iterator_;
  SharedLoadIterator shared_load_iterator_;

  CUTLASS_DEVICE
  void atomic_store_fragment(OutputTileIterator const& destination_iterator,
                             typename SharedLoadIterator::Fragment const& frag,
                             float* __restrict__ c, int n) const {
    float const* frag_ptr = reinterpret_cast<float const*>(&frag);
    int const thread_start_row = destination_iterator.thread_start_row();
    int const thread_start_column = destination_iterator.thread_start_column();
    int const extent_row = destination_iterator.extent_row();

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
         ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int const frag_row_idx =
              row + ThreadMap::Iterations::kRow *
                        (group + ThreadMap::Iterations::kGroup * cluster);
          int const row_offset = row * ThreadMap::Delta::kRow +
                                 group * ThreadMap::Delta::kGroup +
                                 cluster * ThreadMap::Delta::kCluster;
          int const logical_row = thread_start_row + row_offset;

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            int const logical_column_base =
                thread_start_column + column * ThreadMap::Delta::kColumn;
            int const frag_base =
                (frag_row_idx * ThreadMap::Iterations::kColumn + column) *
                ThreadMap::kElementsPerAccess;

            if (logical_row < extent_row) {
              CUTLASS_PRAGMA_UNROLL
              for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
                int64_t const offset =
                    int64_t(logical_row) * n + logical_column_base + e;
                atomicAdd(c + offset, frag_ptr[frag_base + e]);
              }
            }
          }
        }
      }
    }
  }

 public:
  CUTLASS_DEVICE
  Sm70AtomicFp32Epilogue(SharedStorage& shared_storage, int thread_idx,
                         int warp_idx, int lane_idx)
      : warp_tile_iterator_(shared_storage.reference(), lane_idx),
        shared_load_iterator_(shared_storage.reference(), thread_idx) {
    using WarpCount = typename CutlassEpilogue::WarpCount;
    int const warp_k = warp_idx / (WarpCount::kM * WarpCount::kN);
    int const warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int const warp_m = warp_mn % WarpCount::kM;
    int const warp_n = warp_mn / WarpCount::kM;

    cutlass::MatrixCoord warp_offset{warp_k * WarpCount::kM + warp_m, warp_n};
    warp_tile_iterator_.add_tile_offset(warp_offset);
  }

  CUTLASS_DEVICE
  void operator()(OutputTileIterator destination_iterator,
                  AccumulatorTile const& accumulators, float* __restrict__ c,
                  int n) {
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      __syncthreads();

      typename AccumulatorFragmentIterator::Fragment accum_fragment;
      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;
      warp_tile_iterator_.store(accum_fragment);

      __syncthreads();

      typename SharedLoadIterator::Fragment aligned_accum_fragment;
      shared_load_iterator_.load(aligned_accum_fragment);

      if (CutlassEpilogue::kPartitionsK > 1) {
        cutlass::plus<typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < CutlassEpilogue::kPartitionsK; ++i) {
          typename SharedLoadIterator::Fragment aligned_addend_fragment;
          shared_load_iterator_.add_pointer_offset(
              CutlassEpilogue::kSmemPointerOffset);
          shared_load_iterator_.load(aligned_addend_fragment);
          aligned_accum_fragment =
              add_fragments(aligned_accum_fragment, aligned_addend_fragment);
        }

        shared_load_iterator_.add_pointer_offset(
            (1 - CutlassEpilogue::kPartitionsK) *
            CutlassEpilogue::kSmemPointerOffset);
      }

      atomic_store_fragment(destination_iterator, aligned_accum_fragment, c, n);
      ++destination_iterator;
    }
  }
};

// single fp32->fp16 narrow of the split-K scratch. Templated so the symbol
// has vague linkage across translation units that include this header.
template <int BlockSize = 256>
__global__ void sm70_marlin_narrow_f32_to_f16(const float* __restrict__ src,
                                               half* __restrict__ dst,
                                               int64_t numel) {
  int64_t i = int64_t(blockIdx.x) * BlockSize + threadIdx.x;
  if (i < numel) dst[i] = __float2half_rn(src[i]);
}

}  // namespace marlin::sm70

// =============================================================================
// CUTLASS SM70 GEMM pipelined Mma (from sm70_marlin_mma.cuh)
// =============================================================================
namespace marlin::sm70 {

template <typename Shape_, typename IteratorA_, typename SmemIteratorA_,
          typename IteratorB_, typename SmemIteratorB_, typename ElementC_,
          typename LayoutC_, typename Policy_,
          typename TransformA_ = cutlass::NumericArrayConverter<
              typename SmemIteratorA_::Element, typename IteratorA_::Element,
              IteratorA_::Fragment::kElements>,
          typename TransformB_ = cutlass::NumericArrayConverter<
              typename SmemIteratorB_::Element, typename IteratorB_::Element,
              IteratorB_::Fragment::kElements>,
          typename Enable = bool>
class Sm70MarlinMmaPipelined
    : public cutlass::gemm::threadblock::MmaBase<Shape_, Policy_, 2> {
 public:
  using Base = cutlass::gemm::threadblock::MmaBase<Shape_, Policy_, 2>;

  using Shape = Shape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;

  using FragmentA = typename IteratorA::Fragment;
  using FragmentB = typename IteratorB::Fragment;
  using FragmentC = typename Policy::Operator::FragmentC;
  using Operator = typename Policy::Operator;
  using ArchTag = typename Policy::Operator::ArchTag;

  static cutlass::ComplexTransform const kTransformA = Operator::kTransformA;
  static cutlass::ComplexTransform const kTransformB = Operator::kTransformB;

  static_assert(Base::kStages == 2,
                "Sm70MarlinMmaPipelined requires a two-stage pipeline.");

 protected:
  Operator warp_mma;
  SmemIteratorA smem_iterator_A_;
  SmemIteratorB smem_iterator_B_;
  TransformA transform_A_;
  TransformB transform_B_;
  typename Base::TensorRefA warp_operand_A_ref_;
  typename Base::TensorRefB warp_operand_B_ref_;
  int smem_write_stage_idx;
  int lane_idx_;
  int warp_idx_m_;
  int warp_idx_n_;
  int warp_tile_k_group_;

 private:
  static constexpr int kStageKGroups =
      Policy::kPartitionsK * Base::kWarpGemmIterations;
  static constexpr int kCircularKGroups = Base::kStages * kStageKGroups;

  CUTLASS_DEVICE
  static int normalize_warp_tile_k_group(int k_group) {
    int normalized = k_group % kCircularKGroups;
    return normalized < 0 ? normalized + kCircularKGroups : normalized;
  }

  CUTLASS_DEVICE
  void reset_phase_aware_warp_tile_iterators(int k_group) {
    warp_tile_k_group_ = normalize_warp_tile_k_group(k_group);
    this->warp_tile_iterator_A_ =
        typename Operator::IteratorA(warp_operand_A_ref_, lane_idx_);
    this->warp_tile_iterator_B_ =
        typename Operator::IteratorB(warp_operand_B_ref_, lane_idx_);
    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m_, 0});
    this->warp_tile_iterator_B_.add_tile_offset({0, warp_idx_n_});

    for (int i = 0; i < warp_tile_k_group_; ++i) {
      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;
    }
  }

  CUTLASS_DEVICE
  void add_initial_warp_tile_offset(int k_group) {
    reset_phase_aware_warp_tile_iterators(k_group);
  }

  CUTLASS_DEVICE
  void add_warp_tile_k_group_offset(int k_group_delta) {
    reset_phase_aware_warp_tile_iterators(warp_tile_k_group_ + k_group_delta);
  }

  CUTLASS_DEVICE
  void advance_warp_tile_iterators() {
    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;
    warp_tile_k_group_ = normalize_warp_tile_k_group(warp_tile_k_group_ + 1);
  }

 public:
  CUTLASS_DEVICE
  Sm70MarlinMmaPipelined(typename Base::SharedStorage& shared_storage,
                         int thread_idx, int warp_idx, int lane_idx,
                         TransformA transform_A = TransformA(),
                         TransformB transform_B = TransformB())
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        transform_A_(transform_A),
        transform_B_(transform_B),
        warp_operand_A_ref_(shared_storage.operand_A_ref()),
        warp_operand_B_ref_(shared_storage.operand_B_ref()),
        smem_write_stage_idx(0),
        lane_idx_(lane_idx),
        warp_idx_m_(0),
        warp_idx_n_(0),
        warp_tile_k_group_(0) {
    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    warp_idx_m_ = warp_idx_mn % Base::WarpCount::kM;
    warp_idx_n_ = warp_idx_mn / Base::WarpCount::kM;

    add_initial_warp_tile_offset(Base::kWarpGemmIterations * warp_idx_k);
  }

  CUTLASS_DEVICE
  void advance_smem_write_stage() {
    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    if (smem_write_stage_idx == 1) {
      this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
    }

    smem_write_stage_idx ^= 1;
  }

  CUTLASS_DEVICE
  void advance_smem_stages() {
    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    if (smem_write_stage_idx == 1) {
      this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});

      if constexpr (Policy::kPartitionsK > 1) {
        add_warp_tile_k_group_offset((Policy::kPartitionsK - 1) *
                                     Base::kWarpGemmIterations);
      }
    } else {
      add_warp_tile_k_group_offset(-(Policy::kPartitionsK + 1) *
                                   Base::kWarpGemmIterations);
    }

    smem_write_stage_idx ^= 1;
  }

  CUTLASS_DEVICE
  void prologue(IteratorA& iterator_A, IteratorB& iterator_B,
                int& /*gemm_k_iterations*/) {
    FragmentA tb_frag_A;
    tb_frag_A.clear();
    iterator_A.load(tb_frag_A);
    ++iterator_A;

    FragmentB tb_frag_B;
    tb_frag_B.clear();
    iterator_B.load(tb_frag_B);
    ++iterator_B;

    this->smem_iterator_A_.store(transform_A_(tb_frag_A));
    this->smem_iterator_B_.store(transform_B_(tb_frag_B));

    advance_smem_write_stage();
  }

  CUTLASS_DEVICE
  void gmem_wait() { __syncthreads(); }

  CUTLASS_DEVICE
  void gemm_iters(int gemm_k_iterations, FragmentC& accum,
                  IteratorA& iterator_A, IteratorB& iterator_B) {
    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(warp_frag_A[0]);

    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);
    advance_warp_tile_iterators();

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {
        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          this->smem_iterator_A_.store(transform_A_(tb_frag_A));
          this->smem_iterator_B_.store(transform_B_(tb_frag_B));

          gmem_wait();
          advance_smem_stages();
        }

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) %
                                                     Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) %
                                                     Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        advance_warp_tile_iterators();

        if (warp_mma_k == 0) {
          tb_frag_A.clear();
          iterator_A.load(tb_frag_A);
          ++iterator_A;

          tb_frag_B.clear();
          iterator_B.load(tb_frag_B);
          ++iterator_B;

          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B.clear_mask(gemm_k_iterations <= 2);
        }

        warp_mma(accum, warp_frag_A[warp_mma_k % 2],
                 warp_frag_B[warp_mma_k % 2], accum);
      }
    }
  }

  CUTLASS_DEVICE
  void operator()(int gemm_k_iterations, FragmentC& accum, IteratorA iterator_A,
                  IteratorB iterator_B, FragmentC const& src_accum) {
    prologue(iterator_A, iterator_B, gemm_k_iterations);
    gmem_wait();
    accum = src_accum;
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
  }
};

}  // namespace marlin::sm70

// =============================================================================
// GEMM traits + fixed-group-size geometry dispatch (from sm70_marlin_gemm.cuh,
// trimmed to the symbols the NVFP4 GEMM uses; returns torch::stable::Tensor)
// =============================================================================
namespace marlin::sm70 {

template <typename Spec, int CtaM, int CtaN, int CtaK, int Warps, int WarpM,
          int WarpN, int WarpK, int GroupSize, int PackedMacroN>
struct Sm70MarlinGemmTraits {
  static_assert(CtaM == 32 || CtaM == 64 || CtaM == 128 || CtaM == 256,
                "SM70 Marlin supports CTA_M in {32, 64, 128, 256}.");
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 Marlin supports CTA_N in {64, 128, 256}.");
  static_assert(CtaK == 16 || CtaK == 32 || CtaK == 64 || CtaK == 128,
                "SM70 Marlin supports CTA_K in {16, 32, 64, 128}.");
  static_assert(Warps == 4 || Warps == 8, "SM70 Marlin supports 4 or 8 warps.");
  static_assert(PackedMacroN == 64 || PackedMacroN == 128 ||
                    PackedMacroN == 256,
                "SM70 Marlin packed macro-N must be 64, 128, or 256.");
  static_assert(PackedMacroN % CtaN == 0,
                "SM70 Marlin packed macro-N must be divisible by CTA_N.");
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<CtaM, CtaN, CtaK>;
  using WarpShape = typename Sm70WarpShape<CtaM, CtaN, CtaK, Warps, WarpM,
                                           WarpN, WarpK>::Type;
  static_assert(WarpShape::kM <= 64 && WarpShape::kN <= 64,
                "SM70 Marlin keeps per-warp M/N no larger than 64.");
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAdd>;
  static_assert(MmaCore::kThreads == Warps * 32,
                "SM70 Marlin launch threads must match CUTLASS warp count.");
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA,
      128 / cutlass::sizeof_bits<ElementA>::value>;
  using IteratorB =
      typename Spec::template IteratorB<ThreadblockShape,
                                        typename MmaCore::IteratorThreadMapB,
                                        GroupSize, PackedMacroN>;
  using Mma = Sm70MarlinMmaPipelined<
      ThreadblockShape, IteratorA, typename MmaCore::SmemIteratorA, IteratorB,
      typename MmaCore::SmemIteratorB, ElementAccumulator, LayoutC,
      typename MmaCore::MmaPolicy>;
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementAccumulator>;
  using ExpectedSmemLayoutB =
      cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
          cutlass::sizeof_bits<ElementB>::value>;
  using ActualSmemLayoutB = typename Mma::SmemIteratorB::Layout;
  static_assert(std::is_same<ActualSmemLayoutB, ExpectedSmemLayoutB>::value,
                "SM70 Marlin B operand must use CUTLASS' predefined Volta "
                "B-congruous shared-memory layout.");
  static int const kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, OutputOp,
          OutputOp::kCount>::Epilogue;

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };
};

inline dim3 sm70_marlin_cta_grid(int64_t size_m, int64_t size_n, int cta_m,
                                 int cta_n) {
  return dim3(static_cast<unsigned>((size_m + cta_m - 1) / cta_m),
              static_cast<unsigned>(size_n / cta_n));
}

template <typename Launcher>
torch::stable::Tensor dispatch_sm70_marlin_cta_geometry(
    Launcher const& launcher, Sm70CtaGeometry geometry, int packed_macro_n,
    char const* quant_name) {
#define DISPATCH_SM70_PACKED_MACRO_N(CM, CN, CK, W, WM, WN, WK, PMN)       \
  if (packed_macro_n == PMN) {                                             \
    return launcher.template operator()<CM, CN, CK, W, WM, WN, WK, PMN>(); \
  }

#define DISPATCH_SM70_GEOMETRY(CM, CN, CK, W, WM, WN, WK, PMN)                 \
  if (geometry.cta_m == CM && geometry.cta_n == CN && geometry.cta_k == CK &&  \
      geometry.warps == W && geometry.warp_m == WM && geometry.warp_n == WN && \
      geometry.warp_k == WK) {                                                 \
    DISPATCH_SM70_PACKED_MACRO_N(CM, CN, CK, W, WM, WN, WK, PMN)               \
  }

#define FOR_EACH_SM70_GEOMETRY(M)     \
  M(32, 64, 32, 4, 32, 32, 16, 128)   \
  M(32, 64, 32, 4, 32, 32, 16, 256)   \
  M(32, 64, 64, 4, 32, 32, 32, 128)   \
  M(32, 64, 64, 4, 32, 32, 32, 256)   \
  M(32, 64, 64, 4, 32, 64, 16, 128)   \
  M(32, 64, 64, 4, 32, 64, 16, 256)   \
  M(32, 64, 128, 4, 32, 64, 32, 128)  \
  M(32, 64, 128, 4, 32, 64, 32, 256)  \
  M(32, 128, 32, 4, 32, 32, 32, 128)  \
  M(32, 128, 32, 4, 32, 32, 32, 256)  \
  M(32, 128, 64, 4, 32, 64, 32, 128)  \
  M(32, 128, 64, 4, 32, 64, 32, 256)  \
  M(32, 128, 64, 8, 32, 32, 32, 128)  \
  M(32, 128, 64, 8, 32, 32, 32, 256)  \
  M(32, 128, 128, 8, 32, 64, 32, 128) \
  M(32, 128, 128, 8, 32, 64, 32, 256) \
  M(32, 256, 32, 4, 32, 64, 32, 256)  \
  M(32, 256, 64, 8, 32, 64, 32, 256)  \
  M(64, 64, 32, 4, 32, 32, 32, 64)    \
  M(64, 64, 32, 4, 32, 32, 32, 128)   \
  M(64, 64, 32, 4, 32, 32, 32, 256)   \
  M(64, 64, 32, 4, 64, 32, 16, 128)   \
  M(64, 64, 32, 4, 64, 32, 16, 256)   \
  M(64, 64, 64, 4, 32, 64, 32, 128)   \
  M(64, 64, 64, 4, 32, 64, 32, 256)   \
  M(64, 64, 64, 4, 64, 32, 32, 128)   \
  M(64, 64, 64, 4, 64, 32, 32, 256)   \
  M(64, 64, 64, 4, 64, 64, 16, 128)   \
  M(64, 64, 64, 4, 64, 64, 16, 256)   \
  M(64, 128, 32, 4, 32, 64, 32, 128)  \
  M(64, 128, 32, 4, 32, 64, 32, 256)  \
  M(64, 128, 32, 4, 64, 32, 32, 128)  \
  M(64, 128, 32, 4, 64, 32, 32, 256)  \
  M(64, 128, 128, 8, 64, 64, 32, 128) \
  M(64, 128, 128, 8, 64, 64, 32, 256) \
  M(128, 64, 32, 4, 32, 64, 32, 128)  \
  M(128, 64, 32, 4, 32, 64, 32, 256)  \
  M(128, 64, 128, 8, 64, 64, 32, 128) \
  M(128, 64, 128, 8, 64, 64, 32, 256) \
  M(128, 128, 32, 4, 64, 64, 32, 128) \
  M(128, 128, 32, 4, 64, 64, 32, 256) \
  M(128, 128, 64, 8, 64, 64, 32, 128) \
  M(128, 128, 64, 8, 64, 64, 32, 256) \
  M(128, 256, 32, 8, 64, 64, 32, 256)

  FOR_EACH_SM70_GEOMETRY(DISPATCH_SM70_GEOMETRY)

#undef FOR_EACH_SM70_GEOMETRY
#undef DISPATCH_SM70_GEOMETRY
#undef DISPATCH_SM70_PACKED_MACRO_N

  STD_TORCH_CHECK(false, "Unreachable SM70 Marlin ", quant_name,
                  " CTA geometry dispatch.");
}

template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
          int WarpK, int GroupSize, int PackedMacroN, typename Launcher>
torch::stable::Tensor dispatch_sm70_marlin_fixed_group_size(
    Launcher const& launcher, int64_t group_size, char const* quant_name) {
  if (group_size == GroupSize) {
    return launcher.template operator()<CtaM, CtaN, CtaK, Warps, WarpM, WarpN,
                                        WarpK, GroupSize, PackedMacroN>();
  }
  STD_TORCH_CHECK(false, "SM70 Marlin ", quant_name, " supports only group_size ",
                  GroupSize, ". Got ", group_size, ".");
  return torch::stable::Tensor();
}

template <int GroupSize, typename Launcher>
struct Sm70MarlinFixedGroupDispatchLauncher {
  Launcher const& inner;
  int64_t group_size;
  char const* quant_name;

  template <int CtaM, int CtaN, int CtaK, int Warps, int WarpM, int WarpN,
            int WarpK, int PackedMacroN>
  torch::stable::Tensor operator()() const {
    return dispatch_sm70_marlin_fixed_group_size<
        CtaM, CtaN, CtaK, Warps, WarpM, WarpN, WarpK, GroupSize, PackedMacroN>(
        inner, group_size, quant_name);
  }
};

template <int GroupSize, typename Launcher>
torch::stable::Tensor dispatch_sm70_marlin_fixed_group_geometry(
    Launcher const& launcher, Sm70CtaGeometry geometry, int packed_macro_n,
    int64_t group_size, char const* quant_name) {
  return dispatch_sm70_marlin_cta_geometry(
      Sm70MarlinFixedGroupDispatchLauncher<GroupSize, Launcher>{
          launcher, group_size, quant_name},
      geometry, packed_macro_n, quant_name);
}

}  // namespace marlin::sm70
