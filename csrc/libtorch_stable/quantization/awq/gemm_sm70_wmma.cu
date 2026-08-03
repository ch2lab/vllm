#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include "libtorch_stable/torch_utils.h"

using namespace nvcuda;

namespace vllm {
namespace sm70_awq {

// AWQ packing: logical N offset i reads physical nibble AWQ_ORDER[i]
__device__ __constant__ int AWQ_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

// ============================================================
// GEMV kernel for M<=2 (decode): bandwidth-oriented.
// CTA covers 256 columns (4 warps x 64); each lane owns 2 columns
// extracted from one byte of a packed int (AWQ byte pairs
// {0,2},{1,3},{4,6},{5,7} at byte shifts {0,16,8,24}), across the
// whole K slice, so no cross-lane reduction is needed.
// group_size%32==0 => each 32-row chunk lies in one group, so
// scales/zeros are hoisted out of the row loop (fallback: per-row
// global loads). grid: (ceil(N/256), split_k_slices), block 128.
// Writes fp32 partials to output_f32[slice, M, N].
// ============================================================
__global__ __launch_bounds__(128, 6)
void awq_gemv_sm70_kernel(
    const int* __restrict__ qweight,   // [K, N//8]
    const half* __restrict__ scales,   // [num_groups, N]
    const int* __restrict__ qzeros,    // [num_groups, N//8]
    const half* __restrict__ input,    // [M, K]
    float* __restrict__ output_f32,
    int M, int N, int K, int group_size,
    int split_k_slices, int K_per_slice) {

  const int col0 = blockIdx.x * 256;
  const int slice = blockIdx.y;
  const int k_start = slice * K_per_slice;
  const int k_end = min(k_start + K_per_slice, K);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const bool two_rows = M > 1;
  const bool group_aligned = (group_size % 32 == 0);

  // AWQ nibble order {0,4,1,5,2,6,3,7}: each byte holds 2 columns:
  //   byte0={col0,col2} byte1={col4,col6} byte2={col1,col3} byte3={col5,col7}
  static constexpr int PAIR_A[4] = {0, 1, 4, 5};
  static constexpr int PAIR_B[4] = {2, 3, 6, 7};
  static constexpr int BYTE_SHIFT[4] = {0, 16, 8, 24};
  const int wi = warp * 8 + (lane >> 2);  // packed int for this lane's cols
  const int sub = lane & 3;
  const int bs = BYTE_SHIFT[sub];
  const int col_a = col0 + warp * 64 + (lane >> 2) * 8 + PAIR_A[sub];
  const int col_b = col0 + warp * 64 + (lane >> 2) * 8 + PAIR_B[sub];

  __shared__ int sW[32][32];
  __shared__ int sZ[32][32];
  __shared__ half sS[256];       // one group row (aligned path only)
  __shared__ half sX[2][32];

  float acc0_a = 0.0f, acc0_b = 0.0f, acc1_a = 0.0f, acc1_b = 0.0f;

  for (int kb = k_start; kb < k_end; kb += 32) {
    // qweight chunk: 32 rows x 32 packed ints = 1024 ints, 8 per thread
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      int t = threadIdx.x + j * 128;
      int r = t / 32, c = t % 32;
      int gk = kb + r;
      sW[r][c] = (gk < K) ? qweight[gk * (N / 8) + col0 / 8 + c] : 0;
    }
    // qzeros chunk (group-indexed)
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      int t = threadIdx.x + j * 128;
      int r = t / 32, c = t % 32;
      int gk = kb + r;
      sZ[r][c] = (gk < K)
          ? qzeros[(gk / group_size) * (N / 8) + col0 / 8 + c]
          : 0;
    }
    // scales: one group row is enough when group-aligned
    if (group_aligned) {
      if (threadIdx.x < 32) {
        const uint4 v = *reinterpret_cast<const uint4*>(
            &scales[(kb / group_size) * N + col0 + threadIdx.x * 8]);
        const half* hv = reinterpret_cast<const half*>(&v);
        #pragma unroll
        for (int i = 0; i < 8; i++) sS[threadIdx.x * 8 + i] = hv[i];
      }
    }
    // activation rows (broadcast across lanes)
    if (threadIdx.x < 32) {
      int gk = kb + threadIdx.x;
      sX[0][threadIdx.x] = (gk < K) ? input[gk] : __float2half(0.0f);
      if (two_rows)
        sX[1][threadIdx.x] = (gk < K) ? input[K + gk] : __float2half(0.0f);
    }
    __syncthreads();

    if (group_aligned) {
      // scales/zeros are constant across this 32-row chunk
      int zbyte = (sZ[0][wi] >> bs) & 0xFF;
      int z_lo = zbyte & 0xF;
      int z_hi = zbyte >> 4;
      float sf_a = __half2float(sS[col_a - col0]);
      float sf_b = __half2float(sS[col_b - col0]);
      #pragma unroll
      for (int r = 0; r < 32; r++) {
        int gk = kb + r;
        if (gk >= k_end) break;
        int byte = (sW[r][wi] >> bs) & 0xFF;
        int d_a = (byte & 0xF) - z_lo;
        int d_b = (byte >> 4) - z_hi;
        float xf = __half2float(sX[0][r]);
        acc0_a += (float)d_a * (xf * sf_a);
        acc0_b += (float)d_b * (xf * sf_b);
        if (two_rows) {
          float xf1 = __half2float(sX[1][r]);
          acc1_a += (float)d_a * (xf1 * sf_a);
          acc1_b += (float)d_b * (xf1 * sf_b);
        }
      }
    } else {
      // fallback: per-row scales/zeros straight from global memory
      #pragma unroll
      for (int r = 0; r < 32; r++) {
        int gk = kb + r;
        if (gk >= k_end) break;
        int p = sW[r][wi];
        int zint = qzeros[(gk / group_size) * (N / 8) + col0 / 8 + wi];
        int z_lo = (zint >> bs) & 0xF;
        int z_hi = (zint >> (bs + 4)) & 0xF;
        float sf_a = __half2float(scales[(gk / group_size) * N + col_a]);
        float sf_b = __half2float(scales[(gk / group_size) * N + col_b]);
        int byte = (p >> bs) & 0xFF;
        int d_a = (byte & 0xF) - z_lo;
        int d_b = (byte >> 4) - z_hi;
        float xf = __half2float(sX[0][r]);
        acc0_a += (float)d_a * (xf * sf_a);
        acc0_b += (float)d_b * (xf * sf_b);
        if (two_rows) {
          float xf1 = __half2float(sX[1][r]);
          acc1_a += (float)d_a * (xf1 * sf_a);
          acc1_b += (float)d_b * (xf1 * sf_b);
        }
      }
    }
    __syncthreads();
  }

  float* base = output_f32 + (long long)slice * M * N;
  if (col_a < N) base[col_a] = acc0_a;
  if (col_b < N) base[col_b] = acc0_b;
  if (two_rows) {
    if (col_a < N) base[N + col_a] = acc1_a;
    if (col_b < N) base[N + col_b] = acc1_b;
  }
}

// ============================================================
// Main kernel: CTA_M=16, CTA_N=128, CTA_K=32, 4 warps
// Warp layout: 1 row x 4 cols -> each warp: M=16, N=32
// Each warp computes one 16x16 WMMA tile per N step (2 tiles for N=32)
// Split-K via blockIdx.z
// ============================================================
__global__ __launch_bounds__(128, 6)
void awq_gemm_sm70_kernel(
    const int* __restrict__ qweight,   // [K, N//8]
    const half* __restrict__ scales,   // [num_groups, N]
    const int* __restrict__ qzeros,    // [num_groups, N//8]
    const half* __restrict__ input,    // [M, K]
    float* __restrict__ output_f32,    // [M, N] (split-K workspace or final)
    int M, int N, int K, int group_size,
    int split_k_slices, int K_per_slice) {

  constexpr int CTA_M = 16;
  constexpr int CTA_N = 128;
  constexpr int CTA_K = 32;

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int block_m = blockIdx.x * CTA_M;
  const int block_n = blockIdx.y * CTA_N;
  const int slice_id = blockIdx.z;
  const int k_start = slice_id * K_per_slice;
  const int k_end = min(k_start + K_per_slice, K);
  float* my_output = output_f32 + (long long)slice_id * M * N;

  // Each warp handles M=16, N=32 (warp_id 0..3 -> N offset 0,32,64,96)
  const int warp_n_offset = warp_id * 32;

  // WMMA fragments: 16x16x16
  // Each warp does 2 N-tiles (32/16=2) per K step
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
  #pragma unroll
  for (int i = 0; i < 2; i++) wmma::fill_fragment(acc[i], 0.0f);

  // Shared memory: A[16][32+pad], B[32][128+pad]
  __shared__ half sA[16][CTA_K + 8];
  __shared__ half sB[CTA_K][CTA_N + 8];

  for (int k_base = k_start; k_base < k_end; k_base += CTA_K) {
    // --- Load A tile [16, 32] cooperatively ---
    // 128 threads, 16*32=512 elements -> 4 per thread
    #pragma unroll
    for (int idx = threadIdx.x; idx < CTA_M * CTA_K; idx += 128) {
      int m_i = idx / CTA_K;
      int k_i = idx % CTA_K;
      int gm = block_m + m_i;
      int gk = k_base + k_i;
      sA[m_i][k_i] = (gm < M && gk < K) ? input[gm * K + gk]
                                          : __float2half(0.0f);
    }

    // --- Load B tile [32, 128]: dequant INT4 weight ---
    // Each thread owns 8 consecutive N columns: one packed int (8 nibbles),
    // one vectorized scale load (8 halfs), one packed zero.
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < CTA_K * (CTA_N / 8); idx += 128) {
      int k_i = idx / (CTA_N / 8);
      int n8 = (idx % (CTA_N / 8)) * 8;
      int gk = k_base + k_i;
      int gn = block_n + n8;

      if (gk < K && gn < N) {
        // AWQ packing: logical N offset i reads physical nibble {0,4,1,5,2,6,3,7}[i]
        static constexpr int AWQ_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};
        int packed = qweight[gk * (N / 8) + gn / 8];
        int group_idx = gk / group_size;
        // 8 halfs in one 16B transaction
        const uint4 scv = *reinterpret_cast<const uint4*>(
            &scales[group_idx * N + gn]);
        const half* sc = reinterpret_cast<const half*>(&scv);
        int zp = qzeros[group_idx * (N / 8) + gn / 8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          int bit_pos = AWQ_ORDER[j] * 4;
          int w_val = (packed >> bit_pos) & 0xF;
          int z_val = (zp >> bit_pos) & 0xF;
          sB[k_i][n8 + j] = __hmul(
              __hsub(__int2half_rn(w_val), __int2half_rn(z_val)), sc[j]);
        }
      } else {
        #pragma unroll
        for (int j = 0; j < 8; j++) sB[k_i][n8 + j] = __float2half(0.0f);
      }
    }
    __syncthreads();

    // --- WMMA compute: K steps of 16 ---
    #pragma unroll
    for (int k_step = 0; k_step < CTA_K; k_step += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];

      // Load A: [16, 16] from sA[0:16, k_step:k_step+16]
      wmma::load_matrix_sync(a_frag, &sA[0][k_step], CTA_K + 8);

      // Load B: [16, 16] from sB[k_step:k_step+16, warp_n_offset+tile*16]
      // sB is [K][N] row-major, so B fragment uses row_major with ldm=N stride
      #pragma unroll
      for (int tile = 0; tile < 2; tile++) {
        int n_off = warp_n_offset + tile * 16;
        wmma::load_matrix_sync(b_frag[tile], &sB[k_step][n_off], CTA_N + 8);
        wmma::mma_sync(acc[tile], a_frag, b_frag[tile], acc[tile]);
      }
    }
    __syncthreads();
  }

  // --- Write output ---
  // Each warp gets its own sC buffer to avoid inter-warp races
  __shared__ float sC[4][16][16];
  #pragma unroll
  for (int tile = 0; tile < 2; tile++) {
    int n_off = block_n + warp_n_offset + tile * 16;
    if (block_m < M && n_off < N) {
      wmma::store_matrix_sync(&sC[warp_id][0][0], acc[tile], 16,
                              wmma::mem_row_major);
      __syncwarp();

      for (int idx = lane_id; idx < 16 * 16; idx += 32) {
        int m_i = idx / 16;
        int n_i = idx % 16;
        int gm = block_m + m_i;
        int gn = n_off + n_i;
        if (gm < M && gn < N) {
          my_output[gm * N + gn] = sC[warp_id][m_i][n_i];
        }
      }
    }
  }
}

// Split-K reduction kernel
__global__ void split_k_reduce_kernel(const float* __restrict__ workspace,
                                      half* __restrict__ output,
                                      int M, int N, int slices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  float sum = 0.0f;
  for (int s = 0; s < slices; s++) {
    sum += workspace[s * M * N + idx];
  }
  output[idx] = __float2half(sum);
}

}  // namespace sm70_awq
}  // namespace vllm

// ============================================================
// Torch stable ABI entry point
// ============================================================
torch::stable::Tensor awq_gemm_sm70(torch::stable::Tensor input,
                                    torch::stable::Tensor qweight,
                                    torch::stable::Tensor scales,
                                    torch::stable::Tensor qzeros,
                                    int64_t group_size,
                                    int64_t split_k_target) {
  int M = input.size(0);
  int K = input.size(1);
  int N = scales.size(1);

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());

  constexpr int CTA_M = 16;
  constexpr int CTA_N = 128;
  int grid_m = (M + CTA_M - 1) / CTA_M;
  int grid_n = (N + CTA_N - 1) / CTA_N;

  int split_k = 1;
  // Small grid_m*grid_n (decode shapes) underfills the SMs at split_k=1
  // (e.g. M=5 gate_up -> 136 blocks). Raise the threshold so the split-K
  // scaling kicks in for these shapes too.
  if (split_k_target > 0 && grid_m * grid_n < 320) {
    split_k = (split_k_target + grid_m * grid_n - 1) / (grid_m * grid_n);
    int max_slices = K / group_size;
    split_k = std::max(1, std::min(split_k, max_slices));
    split_k = std::min(split_k, 32);
  }
  int K_per_slice =
      ((K + split_k - 1) / split_k + group_size - 1) / group_size * group_size;
  split_k = std::min(split_k, (K + K_per_slice - 1) / K_per_slice);

  auto output = torch::stable::empty({M, N}, input.scalar_type(), std::nullopt,
                                     input.device());

  auto in_ptr = reinterpret_cast<const half*>(
      input.mutable_data_ptr<torch::headeronly::Half>());
  auto qw_ptr = reinterpret_cast<const int*>(
      qweight.mutable_data_ptr<int>());
  auto sc_ptr = reinterpret_cast<const half*>(
      scales.mutable_data_ptr<torch::headeronly::Half>());
  auto qz_ptr = reinterpret_cast<const int*>(
      qzeros.mutable_data_ptr<int>());
  auto out_ptr = reinterpret_cast<half*>(
      output.mutable_data_ptr<torch::headeronly::Half>());

  const cudaStream_t stream = get_current_cuda_stream();

  if (M <= 2) {
    // Decode GEMV path: bandwidth-oriented kernel, split-K over blockIdx.y
    int slices = std::max(2, std::min(32, K / 256));
    int K_per_slice =
        ((K + slices - 1) / slices + group_size - 1) / group_size * group_size;
    slices = std::min(slices, (K + K_per_slice - 1) / K_per_slice);

    auto workspace = torch::stable::empty(
        {slices, M, N}, torch::stable::ScalarType::Float, std::nullopt,
        input.device());
    auto ws_ptr = workspace.mutable_data_ptr<float>();

    dim3 grid((N + 255) / 256, slices);
    vllm::sm70_awq::awq_gemv_sm70_kernel<<<grid, 128, 0, stream>>>(
        qw_ptr, sc_ptr, qz_ptr, in_ptr, ws_ptr, M, N, K, group_size, slices,
        K_per_slice);

    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    vllm::sm70_awq::split_k_reduce_kernel<<<blocks, threads, 0, stream>>>(
        ws_ptr, out_ptr, M, N, slices);
    return output;
  }

  if (split_k == 1) {
    auto workspace = torch::stable::empty(
        {M, N}, torch::stable::ScalarType::Float, std::nullopt, input.device());
    auto ws_ptr = workspace.mutable_data_ptr<float>();

    dim3 grid(grid_m, grid_n, 1);
    vllm::sm70_awq::awq_gemm_sm70_kernel<<<grid, 128, 0, stream>>>(
        qw_ptr, sc_ptr, qz_ptr, in_ptr, ws_ptr, M, N, K, group_size, 1, K);

    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    vllm::sm70_awq::split_k_reduce_kernel<<<blocks, threads, 0, stream>>>(
        ws_ptr, out_ptr, M, N, 1);
  } else {
    auto workspace = torch::stable::empty(
        {split_k, M, N}, torch::stable::ScalarType::Float, std::nullopt,
        input.device());
    auto ws_ptr = workspace.mutable_data_ptr<float>();

    dim3 grid(grid_m, grid_n, split_k);
    vllm::sm70_awq::awq_gemm_sm70_kernel<<<grid, 128, 0, stream>>>(
        qw_ptr, sc_ptr, qz_ptr, in_ptr, ws_ptr, M, N, K, group_size, split_k,
        K_per_slice);

    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    vllm::sm70_awq::split_k_reduce_kernel<<<blocks, threads, 0, stream>>>(
        ws_ptr, out_ptr, M, N, split_k);
  }

  return output;
}
