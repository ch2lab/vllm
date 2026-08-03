#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include "libtorch_stable/torch_utils.h"

using namespace nvcuda;

namespace vllm {
namespace sm70_attn {

__device__ __forceinline__ half fp8e4m3_to_half_scaled(uint8_t x,
                                                       float scale) {
  __half_raw hr = __nv_cvt_fp8_to_halfraw(x, __NV_E4M3);
  float val = __half2float(*reinterpret_cast<half*>(&hr)) * scale;
  return __float2half(val);
}

// Flash-attention prefill kernel for SM70
// BM=16, BN=32, HD=128
// Shared memory: ~27KB (under 48KB SM70 limit)
// Grid: (ceil(Sq/BM), num_heads, batch), Block: 128 threads

constexpr int BM = 16;
constexpr int BN = 32;
constexpr int MAX_PARTITIONS = 16;
constexpr int HD = 128;

template <bool FP8_KV>
__global__ __launch_bounds__(128)
void flash_attn_sm70_prefill_kernel(
    const half* __restrict__ Q,   // [B, H, Sq, D]
    const half* __restrict__ K,   // [B, H, Skv, D] (fp16) or uint8 (fp8)
    const half* __restrict__ V,   // [B, H, Skv, D] (fp16) or uint8 (fp8)
    half* __restrict__ O,         // [B, H, Sq, D]
    int Sq, int Skv, int num_heads, float scale, bool causal,
    float k_scale, float v_scale) {

  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.x;
  const int q_start = q_tile_idx * BM;
  if (q_start >= Sq) return;

  const half* q_ptr = Q + ((batch_idx * num_heads + head_idx) * Sq) * HD;
  const half* k_ptr = K + ((batch_idx * num_heads + head_idx) * Skv) * HD;
  const half* v_ptr = V + ((batch_idx * num_heads + head_idx) * Skv) * HD;
  half* o_ptr = O + ((batch_idx * num_heads + head_idx) * Sq) * HD;

  // FP8 pointers: 1 byte per element vs 2 for half
  const uint8_t* k_fp8 = reinterpret_cast<const uint8_t*>(K) +
      ((batch_idx * num_heads + head_idx) * Skv) * HD;
  const uint8_t* v_fp8 = reinterpret_cast<const uint8_t*>(V) +
      ((batch_idx * num_heads + head_idx) * Skv) * HD;

  // Shared memory layout (total ~27KB):
  __shared__ half sQ[BM][HD + 8];      // 16*136*2 = 4352
  __shared__ half sKV[BN][HD + 8];     // 32*136*2 = 8704 (reused for K then V)
  __shared__ float sS[BM][BN];         // 16*32*4 = 2048
  __shared__ float sO[BM][HD];         // 16*128*4 = 8192
  __shared__ half sP[BM][BN + 8];      // 16*40*2 = 1280
  __shared__ float s_max[BM];          // 64
  __shared__ float s_sum[BM];          // 64
  __shared__ float s_rescale[BM];      // 64
  // Total: ~25KB

  // Load Q tile
  for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
    int m = idx / HD, d = idx % HD;
    int gq = q_start + m;
    sQ[m][d] = (gq < Sq) ? q_ptr[gq * HD + d] : __float2half(0.0f);
  }
  if (threadIdx.x < BM) {
    s_max[threadIdx.x] = -1e30f;
    s_sum[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  // O accumulator in registers (survives across KV tiles)
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[8];
  #pragma unroll
  for (int dt = 0; dt < 8; dt++) {
    wmma::fill_fragment(o_frag[dt], 0.0f);
  }

  int q_offset = Skv - Sq;  // query tokens start at this KV position
  int kv_end = causal ? min(Skv, q_start + BM + q_offset) : Skv;

  for (int kv_start = 0; kv_start < kv_end; kv_start += BN) {
    int actual_bn = min(BN, kv_end - kv_start);

    // Load K tile [BN, HD]
    for (int idx = threadIdx.x; idx < BN * HD; idx += 128) {
      int n = idx / HD, d = idx % HD;
      int gkv = kv_start + n;
      if (gkv < Skv && n < actual_bn) {
        if constexpr (FP8_KV) {
          sKV[n][d] = fp8e4m3_to_half_scaled(k_fp8[gkv * HD + d], k_scale);
        } else {
          sKV[n][d] = k_ptr[gkv * HD + d];
        }
      } else {
        sKV[n][d] = __float2half(0.0f);
      }
    }
    __syncthreads();

    // S = Q * K^T * scale using WMMA
    // Q[16,128] * K^T[128,32] -> S[16,32]
    // M-tiles: 1, N-tiles: 2 (BN/16), K-tiles: 8 (HD/16)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[2];
    wmma::fill_fragment(s_frag[0], 0.0f);
    wmma::fill_fragment(s_frag[1], 0.0f);

    #pragma unroll
    for (int kt = 0; kt < HD / 16; kt++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf;
      wmma::load_matrix_sync(qf, &sQ[0][kt * 16], HD + 8);
      #pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
        wmma::load_matrix_sync(kf, &sKV[nt * 16][kt * 16], HD + 8);
        wmma::mma_sync(s_frag[nt], qf, kf, s_frag[nt]);
      }
    }

    // Store S fragments to sS
    __shared__ float sS_tmp[2][16][16];
    wmma::store_matrix_sync(&sS_tmp[0][0][0], s_frag[0], 16, wmma::mem_row_major);
    wmma::store_matrix_sync(&sS_tmp[1][0][0], s_frag[1], 16, wmma::mem_row_major);
    __syncthreads();

    // Apply scale + causal mask, compute online softmax
    for (int idx = threadIdx.x; idx < BM * BN; idx += 128) {
      int m = idx / BN, n = idx % BN;
      float val = sS_tmp[n / 16][m][n % 16] * scale;
      if (causal && (kv_start + n > q_offset + q_start + m)) val = -1e30f;
      if (n >= actual_bn) val = -1e30f;
      sS[m][n] = val;
    }
    __syncthreads();

    // Online softmax: update max, rescale, compute weights
    for (int m = threadIdx.x; m < BM; m += 128) {
      float old_max = s_max[m];
      float new_max = old_max;
      for (int n = 0; n < BN; n++) new_max = fmaxf(new_max, sS[m][n]);

      float rescale = __expf(old_max - new_max);
      float new_sum = 0.0f;
      for (int n = 0; n < BN; n++) {
        float p = __expf(sS[m][n] - new_max);
        sS[m][n] = p;
        new_sum += p;
      }
      s_sum[m] = s_sum[m] * rescale + new_sum;
      s_max[m] = new_max;
      s_rescale[m] = rescale;
    }
    __syncthreads();

    // Rescale O accumulator via shared memory
    #pragma unroll
    for (int dt = 0; dt < 8; dt++) {
      wmma::store_matrix_sync(&sO[0][dt * 16], o_frag[dt], HD, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
      sO[idx / HD][idx % HD] *= s_rescale[idx / HD];
    }
    __syncthreads();
    #pragma unroll
    for (int dt = 0; dt < 8; dt++) {
      wmma::load_matrix_sync(o_frag[dt], &sO[0][dt * 16], HD, wmma::mem_row_major);
    }

    // Load V tile [BN, HD] (reuse sKV)
    for (int idx = threadIdx.x; idx < BN * HD; idx += 128) {
      int n = idx / HD, d = idx % HD;
      int gkv = kv_start + n;
      if (gkv < Skv && n < actual_bn) {
        if constexpr (FP8_KV) {
          sKV[n][d] = fp8e4m3_to_half_scaled(v_fp8[gkv * HD + d], v_scale);
        } else {
          sKV[n][d] = v_ptr[gkv * HD + d];
        }
      } else {
        sKV[n][d] = __float2half(0.0f);
      }
    }

    // Convert P to half
    for (int idx = threadIdx.x; idx < BM * BN; idx += 128) {
      sP[idx / BN][idx % BN] = __float2half(sS[idx / BN][idx % BN]);
    }
    __syncthreads();

    // O += P * V using WMMA: [16,32] * [32,128] -> [16,128]
    #pragma unroll
    for (int dt = 0; dt < 8; dt++) {
      #pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
        wmma::load_matrix_sync(pf, &sP[0][nt * 16], BN + 8);
        wmma::load_matrix_sync(vf, &sKV[nt * 16][dt * 16], HD + 8);
        wmma::mma_sync(o_frag[dt], pf, vf, o_frag[dt]);
      }
    }
    __syncthreads();
  }

  // Store O fragments to shared for final division
  #pragma unroll
  for (int dt = 0; dt < 8; dt++) {
    wmma::store_matrix_sync(&sO[0][dt * 16], o_frag[dt], HD, wmma::mem_row_major);
  }
  __syncthreads();

  // Final: O /= sum, write to global
  for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
    int m = idx / HD, d = idx % HD;
    int gq = q_start + m;
    if (gq < Sq) {
      float sum = s_sum[m];
      o_ptr[gq * HD + d] = __float2half(sum > 0.0f ? sO[m][d] / sum : 0.0f);
    }
  }
}

}  // namespace sm70_attn
}  // namespace vllm

// Paged WMMA attention kernel: reads K/V directly from paged cache
namespace vllm {
namespace sm70_attn {

// NVFP4 (e2m1) magnitude/sign lookup: nibble -> half value
__device__ __constant__ float NVFP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// KV_MODE: 0 = FP16, 1 = FP8 (e4m3, dequant inline), 2 = INT4, 3 = NVFP4
// INT4: 2 nibbles/byte, slot [K(HD/2 B) | V(HD/2 B)], symmetric (nib-8)*scale
// NVFP4: upstream-compatible layout: K/V are separate head rows
//        (V head = kv_head + num_kv_heads), each slot is
//        [e2m1 data (HD/2 B) | fp8-e4m3 block scales (HD/16 B, per 16 elems)];
//        value = lut[nib] * fp8(block_scale) * global_scale
template <int KV_MODE, int HEAD_DIM, bool BATCHED>
__global__ __launch_bounds__(128)
void flash_attn_sm70_paged_kernel(
    const half* __restrict__ Q,       // [B, H, Sq, D] or [total_q, H, D]
    const half* __restrict__ kv_cache,// [num_blocks, num_kv_heads, block_size, 2*D]
    const int* __restrict__ block_table, // [B, max_blocks]
    const int* __restrict__ query_start_loc,  // BATCHED: [B+1]
    const int* __restrict__ seq_lens_arr,     // BATCHED: [B]
    half* __restrict__ O,             // [B, H, Sq, D] or [total_q, H, D]
    int Sq_arg, int seq_len_arg, int num_heads, int num_kv_heads,
    float scale, bool causal,
    int block_size, int max_blocks, float k_scale, float v_scale,
    int64_t stride_block, int64_t stride_head, int64_t stride_slot,
    int num_partitions, half* __restrict__ o_partial,
    float* __restrict__ max_partial, float* __restrict__ sum_partial) {

  constexpr int HD = HEAD_DIM;
  constexpr int K_TILES = HD / 16;       // QK reduction tiles
  constexpr int D_TILES = HD / 16;       // PV output tiles
  constexpr int TILES_PER_WARP = D_TILES / 4;  // 2 for HD=128, 4 for HD=256

  const int head_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.x;
  const int q_start = q_tile_idx * BM;

  // Per-sequence parameters and Q/O row layout
  int Sq, kv_len;
  const half* q_ptr;
  half* o_ptr;
  const int* bt;
  int q_row_stride;
  int seq = 0;
  int partition_idx = 0;
  if constexpr (BATCHED) {
    seq = blockIdx.z / num_partitions;
    partition_idx = blockIdx.z % num_partitions;
    const int q_base = query_start_loc[seq];
    Sq = query_start_loc[seq + 1] - q_base;
    kv_len = seq_lens_arr[seq];
    bt = block_table + seq * max_blocks;
    q_ptr = Q + (q_base * num_heads + head_idx) * HD;
    o_ptr = O + (q_base * num_heads + head_idx) * HD;
    q_row_stride = num_heads * HD;
  } else {
    Sq = Sq_arg;
    kv_len = seq_len_arg;
    bt = block_table + blockIdx.z * max_blocks;
    q_ptr = Q + ((blockIdx.z * num_heads + head_idx) * Sq) * HD;
    o_ptr = O + ((blockIdx.z * num_heads + head_idx) * Sq) * HD;
    q_row_stride = HD;
  }
  if (q_start >= Sq) return;
  const int warp_id = threadIdx.x / 32;
  const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
  const half* q_tile = q_ptr + q_start * q_row_stride;
  half* o_tile = o_ptr + q_start * q_row_stride;

  // kv_cache element strides (passed in): physical layout may be NHD
  const half* cache_base = kv_cache + kv_head_idx * stride_head;
  // FP8: 1 byte per element, compute byte-level base
  const uint8_t* cache_base_fp8 = reinterpret_cast<const uint8_t*>(kv_cache) +
      kv_head_idx * stride_head;

  __shared__ half sQ[BM][HD + 8];
  __shared__ half sKV[BN][HD + 8];
  __shared__ float sS[BM][BN];
  __shared__ float sO[BM][HD];
  __shared__ half sP[BM][BN + 8];
  __shared__ float s_max[BM];
  __shared__ float s_sum[BM];
  __shared__ float s_rescale[BM];

  // Load Q tile
  for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
    int m = idx / HD, d = idx % HD;
    int gq = q_start + m;
    sQ[m][d] = (gq < Sq) ? q_tile[m * q_row_stride + d] : __float2half(0.0f);
  }
  if (threadIdx.x < BM) {
    s_max[threadIdx.x] = -1e30f;
    s_sum[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  // Each warp owns TILES_PER_WARP D-tiles of O
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[TILES_PER_WARP];
  #pragma unroll
  for (int dt = 0; dt < TILES_PER_WARP; dt++) wmma::fill_fragment(o_frag[dt], 0.0f);

  int q_offset = kv_len - Sq;
  int kv_end = causal ? min(kv_len, q_start + BM + q_offset) : kv_len;
  // Parallel KV partition: this block handles [seg_start, seg_end) of the
  // visible KV. num_partitions==1 keeps the original single-block scan.
  const int seg_len = (kv_end + num_partitions - 1) / num_partitions;
  const int seg_start = min(partition_idx * seg_len, kv_end);
  const int seg_end = min(kv_end, seg_start + seg_len);

  for (int kv_start = seg_start; kv_start < seg_end; kv_start += BN) {
    int actual_bn = min(BN, kv_end - kv_start);

    // Load K tile from paged cache (vectorized: 8 elements per load)
    if constexpr (KV_MODE == 1) {
      for (int chunk = threadIdx.x; chunk < BN * (HD / 8); chunk += 128) {
        int n = chunk / (HD / 8), d8 = (chunk % (HD / 8)) * 8;
        int gkv = kv_start + n;
        if (gkv < kv_len && n < actual_bn) {
          int blk = bt[gkv / block_size];
          int off = gkv % block_size;
          int64_t base = (int64_t)blk * stride_block +
              (int64_t)off * stride_slot + d8;
          uint2 raw = *reinterpret_cast<const uint2*>(cache_base_fp8 + base);
          const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);
          #pragma unroll
          for (int i = 0; i < 8; i++)
            sKV[n][d8 + i] = fp8e4m3_to_half_scaled(bytes[i], k_scale);
        } else {
          #pragma unroll
          for (int i = 0; i < 8; i++) sKV[n][d8 + i] = __float2half(0.0f);
        }
      }
    } else if constexpr (KV_MODE == 2) {
      // INT4: 8 nibbles per uint load, byte offset d8/2
      for (int chunk = threadIdx.x; chunk < BN * (HD / 8); chunk += 128) {
        int n = chunk / (HD / 8), d8 = (chunk % (HD / 8)) * 8;
        int gkv = kv_start + n;
        if (gkv < kv_len && n < actual_bn) {
          int blk = bt[gkv / block_size];
          int off = gkv % block_size;
          int64_t base = (int64_t)blk * stride_block +
              (int64_t)off * stride_slot + d8 / 2;
          uint raw = *reinterpret_cast<const uint*>(cache_base_fp8 + base);
          #pragma unroll
          for (int i = 0; i < 8; i++) {
            int nib = (raw >> (i * 4)) & 0xF;
            sKV[n][d8 + i] = __float2half(((float)nib - 8.0f) * k_scale);
          }
        } else {
          #pragma unroll
          for (int i = 0; i < 8; i++) sKV[n][d8 + i] = __float2half(0.0f);
        }
      }
    } else if constexpr (KV_MODE == 3) {
      // NVFP4 engine layout, per side: [data (KVH*BS*HD/2 B) |
      // scales (KVH*BS*HD/16 B)], NHD within each region.
      // cache_base_fp8 carries a kv_head pre-offset; remove it — this mode
      // indexes heads explicitly.
      const uint8_t* side =
          cache_base_fp8 - (int64_t)kv_head_idx * stride_head;
      constexpr int DD = HD / 2;   // data bytes per slot
      constexpr int SD = HD / 16;  // scale bytes per slot
      for (int chunk = threadIdx.x; chunk < BN * (HD / 8); chunk += 128) {
        int n = chunk / (HD / 8), d8 = (chunk % (HD / 8)) * 8;
        int gkv = kv_start + n;
        if (gkv < kv_len && n < actual_bn) {
          int blk = bt[gkv / block_size];
          int off = gkv % block_size;
          int64_t page = (int64_t)blk * stride_block;
          uint raw = *reinterpret_cast<const uint*>(
              side + page +
              (int64_t)kv_head_idx * block_size * DD + off * DD + d8 / 2);
          float bs = __half2float(fp8e4m3_to_half_scaled(
              side[page +
                  (int64_t)num_kv_heads * block_size * DD +
                  (int64_t)kv_head_idx * block_size * SD + off * SD +
                  d8 / 16],
              k_scale));
          #pragma unroll
          for (int i = 0; i < 8; i++) {
            int nib = (raw >> (i * 4)) & 0xF;
            sKV[n][d8 + i] = __float2half(NVFP4_LUT[nib] * bs);
          }
        } else {
          #pragma unroll
          for (int i = 0; i < 8; i++) sKV[n][d8 + i] = __float2half(0.0f);
        }
      }
    } else {
      for (int chunk = threadIdx.x; chunk < BN * (HD / 8); chunk += 128) {
        int n = chunk / (HD / 8), d8 = (chunk % (HD / 8)) * 8;
        int gkv = kv_start + n;
        if (gkv < kv_len && n < actual_bn) {
          int blk = bt[gkv / block_size];
          int off = gkv % block_size;
          int64_t base = (int64_t)blk * stride_block +
              (int64_t)off * stride_slot + d8;
          *reinterpret_cast<uint4*>(&sKV[n][d8]) =
              *reinterpret_cast<const uint4*>(cache_base + base);
        } else {
          #pragma unroll
          for (int i = 0; i < 8; i++) sKV[n][d8 + i] = __float2half(0.0f);
        }
      }
    }
    __syncthreads();

    // QK matmul: warps 0-1 each compute one 16x16 N-tile
    __shared__ float sS_tmp[2][16][16];
    if (warp_id < 2) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
      wmma::fill_fragment(s_frag, 0.0f);
      #pragma unroll
      for (int kt = 0; kt < K_TILES; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf;
        wmma::load_matrix_sync(qf, &sQ[0][kt * 16], HD + 8);
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
        wmma::load_matrix_sync(kf, &sKV[warp_id * 16][kt * 16], HD + 8);
        wmma::mma_sync(s_frag, qf, kf, s_frag);
      }
      wmma::store_matrix_sync(&sS_tmp[warp_id][0][0], s_frag, 16,
                              wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < BM * BN; idx += 128) {
      int m = idx / BN, n = idx % BN;
      float val = sS_tmp[n / 16][m][n % 16] * scale;
      if (causal && (kv_start + n > q_offset + q_start + m)) val = -1e30f;
      if (n >= actual_bn) val = -1e30f;
      sS[m][n] = val;
    }
    __syncthreads();

    // Softmax + rescale
    for (int m = threadIdx.x; m < BM; m += 128) {
      float old_max = s_max[m];
      float new_max = old_max;
      for (int n = 0; n < BN; n++) new_max = fmaxf(new_max, sS[m][n]);
      float rescale = __expf(old_max - new_max);
      float new_sum = 0.0f;
      for (int n = 0; n < BN; n++) {
        float p = __expf(sS[m][n] - new_max);
        sS[m][n] = p;
        new_sum += p;
      }
      s_sum[m] = s_sum[m] * rescale + new_sum;
      s_max[m] = new_max;
      s_rescale[m] = rescale;
    }
    __syncthreads();

    #pragma unroll
    for (int dt = 0; dt < TILES_PER_WARP; dt++)
      wmma::store_matrix_sync(&sO[0][warp_id * TILES_PER_WARP * 16 + dt * 16],
                              o_frag[dt], HD, wmma::mem_row_major);
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM * HD; idx += 128)
      sO[idx / HD][idx % HD] *= s_rescale[idx / HD];
    __syncthreads();
    #pragma unroll
    for (int dt = 0; dt < TILES_PER_WARP; dt++)
      wmma::load_matrix_sync(o_frag[dt],
                             &sO[0][warp_id * TILES_PER_WARP * 16 + dt * 16],
                             HD, wmma::mem_row_major);

    // Load V tile from paged cache (vectorized, V at offset HD)
    if constexpr (KV_MODE == 1) {
      for (int chunk = threadIdx.x; chunk < BN * (HD / 8); chunk += 128) {
        int n = chunk / (HD / 8), d8 = (chunk % (HD / 8)) * 8;
        int gkv = kv_start + n;
        if (gkv < kv_len && n < actual_bn) {
          int blk = bt[gkv / block_size];
          int off = gkv % block_size;
          int64_t base = (int64_t)blk * stride_block +
              (int64_t)off * stride_slot + HD + d8;
          uint2 raw = *reinterpret_cast<const uint2*>(cache_base_fp8 + base);
          const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);
          #pragma unroll
          for (int i = 0; i < 8; i++)
            sKV[n][d8 + i] = fp8e4m3_to_half_scaled(bytes[i], v_scale);
        } else {
          #pragma unroll
          for (int i = 0; i < 8; i++) sKV[n][d8 + i] = __float2half(0.0f);
        }
      }
    } else if constexpr (KV_MODE == 2) {
      // INT4: V nibbles start at byte offset HD/2 within the slot
      for (int chunk = threadIdx.x; chunk < BN * (HD / 8); chunk += 128) {
        int n = chunk / (HD / 8), d8 = (chunk % (HD / 8)) * 8;
        int gkv = kv_start + n;
        if (gkv < kv_len && n < actual_bn) {
          int blk = bt[gkv / block_size];
          int off = gkv % block_size;
          int64_t base = (int64_t)blk * stride_block +
              (int64_t)off * stride_slot + HD / 2 + d8 / 2;
          uint raw = *reinterpret_cast<const uint*>(cache_base_fp8 + base);
          #pragma unroll
          for (int i = 0; i < 8; i++) {
            int nib = (raw >> (i * 4)) & 0xF;
            sKV[n][d8 + i] = __float2half(((float)nib - 8.0f) * v_scale);
          }
        } else {
          #pragma unroll
          for (int i = 0; i < 8; i++) sKV[n][d8 + i] = __float2half(0.0f);
        }
      }
    } else if constexpr (KV_MODE == 3) {
      // NVFP4 engine layout: V side starts after the K side within the page.
      const uint8_t* side =
          cache_base_fp8 - (int64_t)kv_head_idx * stride_head +
          (int64_t)num_kv_heads * stride_head;
      constexpr int DD = HD / 2;
      constexpr int SD = HD / 16;
      for (int chunk = threadIdx.x; chunk < BN * (HD / 8); chunk += 128) {
        int n = chunk / (HD / 8), d8 = (chunk % (HD / 8)) * 8;
        int gkv = kv_start + n;
        if (gkv < kv_len && n < actual_bn) {
          int blk = bt[gkv / block_size];
          int off = gkv % block_size;
          int64_t page = (int64_t)blk * stride_block;
          uint raw = *reinterpret_cast<const uint*>(
              side + page +
              (int64_t)kv_head_idx * block_size * DD + off * DD + d8 / 2);
          float bs = __half2float(fp8e4m3_to_half_scaled(
              side[page +
                  (int64_t)num_kv_heads * block_size * DD +
                  (int64_t)kv_head_idx * block_size * SD + off * SD +
                  d8 / 16],
              v_scale));
          #pragma unroll
          for (int i = 0; i < 8; i++) {
            int nib = (raw >> (i * 4)) & 0xF;
            sKV[n][d8 + i] = __float2half(NVFP4_LUT[nib] * bs);
          }
        } else {
          #pragma unroll
          for (int i = 0; i < 8; i++) sKV[n][d8 + i] = __float2half(0.0f);
        }
      }
    } else {
      for (int chunk = threadIdx.x; chunk < BN * (HD / 8); chunk += 128) {
        int n = chunk / (HD / 8), d8 = (chunk % (HD / 8)) * 8;
        int gkv = kv_start + n;
        if (gkv < kv_len && n < actual_bn) {
          int blk = bt[gkv / block_size];
          int off = gkv % block_size;
          int64_t base = (int64_t)blk * stride_block +
              (int64_t)off * stride_slot + HD + d8;
          *reinterpret_cast<uint4*>(&sKV[n][d8]) =
              *reinterpret_cast<const uint4*>(cache_base + base);
        } else {
          #pragma unroll
          for (int i = 0; i < 8; i++) sKV[n][d8 + i] = __float2half(0.0f);
        }
      }
    }

    for (int idx = threadIdx.x; idx < BM * BN; idx += 128)
      sP[idx / BN][idx % BN] = __float2half(sS[idx / BN][idx % BN]);
    __syncthreads();

    // PV matmul: warp w computes its TILES_PER_WARP D-tiles
    #pragma unroll
    for (int dt = 0; dt < TILES_PER_WARP; dt++) {
      const int d_tile = warp_id * TILES_PER_WARP + dt;
      #pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
        wmma::load_matrix_sync(pf, &sP[0][nt * 16], BN + 8);
        wmma::load_matrix_sync(vf, &sKV[nt * 16][d_tile * 16], HD + 8);
        wmma::mma_sync(o_frag[dt], pf, vf, o_frag[dt]);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int dt = 0; dt < TILES_PER_WARP; dt++)
    wmma::store_matrix_sync(&sO[0][warp_id * TILES_PER_WARP * 16 + dt * 16],
                            o_frag[dt], HD, wmma::mem_row_major);
  __syncthreads();

  for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
    int m = idx / HD, d = idx % HD;
    int gq = q_start + m;
    if (gq < Sq) {
      float sum = s_sum[m];
      o_tile[m * q_row_stride + d] =
          __float2half(sum > 0.0f ? sO[m][d] / sum : 0.0f);
    }
  }

  if (num_partitions > 1) {
    // Write unscaled partial (out, max, sum) for the log-sum-exp merge kernel.
    const int pidx = (seq * num_partitions + partition_idx) * num_heads + head_idx;
    half* op = o_partial + (int64_t)pidx * BM * HD;
    float* mp = max_partial + (int64_t)pidx * BM;
    float* sp = sum_partial + (int64_t)pidx * BM;
    for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
      int m = idx / HD, d = idx % HD;
      if (q_start + m < Sq) op[m * HD + d] = __float2half(sO[m][d]);
    }
    for (int m = threadIdx.x; m < BM; m += 128)
      if (q_start + m < Sq) {
        mp[m] = s_max[m];
        sp[m] = s_sum[m];
      }
  }
}

__global__ __launch_bounds__(128)
void flash_attn_sm70_decode_merge_kernel(
    const half* __restrict__ o_partial,   // [num_seqs*P, H, BM, HD]
    const float* __restrict__ max_partial,// [num_seqs*P, H, BM]
    const float* __restrict__ sum_partial,// [num_seqs*P, H, BM]
    const int* __restrict__ query_start_loc,  // [B+1]
    half* __restrict__ O,                 // [total_q, H, HD]
    int num_heads, int num_partitions, int BM, int HD) {
  const int seq = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.x;
  const int q_start = q_tile_idx * BM;
  const int q_base = query_start_loc[seq];
  const int Sq = query_start_loc[seq + 1] - q_base;
  if (q_start >= Sq) return;

  for (int gm = 0; gm < BM; gm++) {
    const int gq = q_start + gm;
    if (gq >= Sq) break;
    float global_max = -1e30f;
    for (int p = 0; p < num_partitions; p++) {
      const int pidx = (seq * num_partitions + p) * num_heads + head_idx;
      global_max = fmaxf(global_max, max_partial[pidx * BM + gm]);
    }
    float den = 0.0f;
    float w[MAX_PARTITIONS];
    for (int p = 0; p < num_partitions; p++) {
      const int pidx = (seq * num_partitions + p) * num_heads + head_idx;
      w[p] = __expf(max_partial[pidx * BM + gm] - global_max);
      den += w[p] * sum_partial[pidx * BM + gm];
    }
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;
    half* o = O + ((int64_t)(q_base + gq) * num_heads + head_idx) * HD;
    for (int d = 0; d < HD; d++) {
      float num_d = 0.0f;
      for (int p = 0; p < num_partitions; p++) {
        const int pidx = (seq * num_partitions + p) * num_heads + head_idx;
        num_d += w[p] * __half2float(
            o_partial[((int64_t)pidx * BM + gm) * HD + d]);
      }
      o[d] = __float2half(num_d * inv);
    }
  }
}

}  // namespace sm70_attn
}  // namespace vllm

torch::stable::Tensor flash_attn_sm70_decode_partitioned(
    torch::stable::Tensor Q, torch::stable::Tensor kv_cache,
    torch::stable::Tensor block_table, torch::stable::Tensor query_start_loc,
    torch::stable::Tensor seq_lens, int64_t max_query_len, int64_t num_seqs,
    int64_t num_partitions, double scale, double k_scale, double v_scale,
    int64_t kv_mode) {
  int total_q = Q.size(0);
  int H = Q.size(1);
  int head_dim = Q.size(2);
  int num_kv_heads = kv_cache.size(1);
  int block_size = kv_cache.size(2);
  int max_blocks = block_table.size(1);
  int mode = (int)kv_mode;
  if (mode == 0) mode = (kv_cache.scalar_type() != Q.scalar_type()) ? 1 : 0;
  const int BM = vllm::sm70_attn::BM;
  const int P = (int)num_partitions;

  const torch::stable::accelerator::DeviceGuard device_guard(
      Q.get_device_index());

  const int64_t total_partial = (int64_t)num_seqs * P * H * BM;
  auto o_partial = torch::stable::empty(
      {total_partial, head_dim}, Q.scalar_type(), std::nullopt, Q.device());
  auto max_partial = torch::stable::empty(
      {total_partial}, torch::headeronly::ScalarType::Float, std::nullopt,
      Q.device());
  auto sum_partial = torch::stable::empty(
      {total_partial}, torch::headeronly::ScalarType::Float, std::nullopt,
      Q.device());
  auto O = torch::stable::empty({total_q, H, head_dim}, Q.scalar_type(),
                                std::nullopt, Q.device());

  dim3 grid((max_query_len + BM - 1) / BM, H, num_seqs * P);
  dim3 mgrid((max_query_len + BM - 1) / BM, H, num_seqs);
  const cudaStream_t stream = get_current_cuda_stream();

  #define LAUNCH_PARTED(MODE, HD_VAL) \
    vllm::sm70_attn::flash_attn_sm70_paged_kernel<MODE, HD_VAL, true> \
        <<<grid, 128, 0, stream>>>( \
        reinterpret_cast<const half*>(Q.mutable_data_ptr<torch::headeronly::Half>()), \
        reinterpret_cast<const half*>(kv_cache.mutable_data_ptr()), \
        block_table.mutable_data_ptr<int>(), \
        query_start_loc.mutable_data_ptr<int>(), \
        seq_lens.mutable_data_ptr<int>(), \
        reinterpret_cast<half*>(O.mutable_data_ptr<torch::headeronly::Half>()), \
        0, 0, H, num_kv_heads, (float)scale, true, \
        block_size, max_blocks, (float)k_scale, (float)v_scale, \
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), \
        P, reinterpret_cast<half*>(o_partial.mutable_data_ptr<torch::headeronly::Half>()), \
        max_partial.mutable_data_ptr<float>(), sum_partial.mutable_data_ptr<float>()); \
    vllm::sm70_attn::flash_attn_sm70_decode_merge_kernel \
        <<<mgrid, 128, 0, stream>>>( \
        reinterpret_cast<const half*>(o_partial.mutable_data_ptr<torch::headeronly::Half>()), \
        max_partial.mutable_data_ptr<float>(), sum_partial.mutable_data_ptr<float>(), \
        query_start_loc.mutable_data_ptr<int>(), \
        reinterpret_cast<half*>(O.mutable_data_ptr<torch::headeronly::Half>()), \
        H, P, BM, HD_VAL)

  if (head_dim == 128) {
    if (mode == 2) { LAUNCH_PARTED(2, 128); }
    else if (mode == 3) { LAUNCH_PARTED(3, 128); }
    else if (mode == 1) { LAUNCH_PARTED(1, 128); }
    else { LAUNCH_PARTED(0, 128); }
  } else {
    if (mode == 2) { LAUNCH_PARTED(2, 256); }
    else if (mode == 3) { LAUNCH_PARTED(3, 256); }
    else if (mode == 1) { LAUNCH_PARTED(1, 256); }
    else { LAUNCH_PARTED(0, 256); }
  }
  #undef LAUNCH_PARTED

  return O;
}

torch::stable::Tensor flash_attn_sm70_prefill_paged(
    torch::stable::Tensor Q, torch::stable::Tensor kv_cache,
    torch::stable::Tensor block_table, double scale, bool causal,
    int64_t seq_len, double k_scale, double v_scale) {
  int B = Q.size(0);
  int H = Q.size(1);
  int Sq = Q.size(2);
  int head_dim = Q.size(3);
  int num_kv_heads = kv_cache.size(1);
  int block_size = kv_cache.size(2);
  int max_blocks = block_table.size(1);
  bool fp8 = (kv_cache.scalar_type() != Q.scalar_type());

  const torch::stable::accelerator::DeviceGuard device_guard(
      Q.get_device_index());

  auto O = torch::stable::empty({B, H, Sq, head_dim},
                                Q.scalar_type(), std::nullopt, Q.device());

  dim3 grid((Sq + vllm::sm70_attn::BM - 1) / vllm::sm70_attn::BM, H, B);
  const cudaStream_t stream = get_current_cuda_stream();

  #define LAUNCH_PAGED_KERNEL(MODE, HD_VAL) \
    vllm::sm70_attn::flash_attn_sm70_paged_kernel<MODE, HD_VAL, false> \
        <<<grid, 128, 0, stream>>>( \
        reinterpret_cast<const half*>(Q.mutable_data_ptr<torch::headeronly::Half>()), \
        reinterpret_cast<const half*>(kv_cache.mutable_data_ptr()), \
        block_table.mutable_data_ptr<int>(), \
        nullptr, nullptr, \
        reinterpret_cast<half*>(O.mutable_data_ptr<torch::headeronly::Half>()), \
        Sq, (int)seq_len, H, num_kv_heads, (float)scale, causal, \
        block_size, max_blocks, (float)k_scale, (float)v_scale, \
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), \
        1, nullptr, nullptr, nullptr);

  if (head_dim == 128) {
    if (fp8) { LAUNCH_PAGED_KERNEL(1, 128) }
    else { LAUNCH_PAGED_KERNEL(0, 128) }
  } else if (head_dim == 256) {
    if (fp8) { LAUNCH_PAGED_KERNEL(1, 256) }
    else { LAUNCH_PAGED_KERNEL(0, 256) }
  }
  #undef LAUNCH_PAGED_KERNEL

  return O;
}

torch::stable::Tensor flash_attn_sm70_prefill_paged_batched(
    torch::stable::Tensor Q, torch::stable::Tensor kv_cache,
    torch::stable::Tensor block_table,
    torch::stable::Tensor query_start_loc, torch::stable::Tensor seq_lens,
    int64_t num_seqs, int64_t max_query_len,
    double scale, bool causal, double k_scale, double v_scale,
    int64_t kv_mode) {
  int total_q = Q.size(0);
  int H = Q.size(1);
  int head_dim = Q.size(2);
  int num_kv_heads = kv_cache.size(1);
  int block_size = kv_cache.size(2);
  int max_blocks = block_table.size(1);
  int mode = (int)kv_mode;
  if (mode == 0) mode = (kv_cache.scalar_type() != Q.scalar_type()) ? 1 : 0;

  const torch::stable::accelerator::DeviceGuard device_guard(
      Q.get_device_index());

  auto O = torch::stable::empty({total_q, H, head_dim},
                                Q.scalar_type(), std::nullopt, Q.device());

  dim3 grid((max_query_len + vllm::sm70_attn::BM - 1) / vllm::sm70_attn::BM,
            H, num_seqs);
  const cudaStream_t stream = get_current_cuda_stream();

  #define LAUNCH_BATCHED_KERNEL(MODE, HD_VAL) \
    vllm::sm70_attn::flash_attn_sm70_paged_kernel<MODE, HD_VAL, true> \
        <<<grid, 128, 0, stream>>>( \
        reinterpret_cast<const half*>(Q.mutable_data_ptr<torch::headeronly::Half>()), \
        reinterpret_cast<const half*>(kv_cache.mutable_data_ptr()), \
        block_table.mutable_data_ptr<int>(), \
        query_start_loc.mutable_data_ptr<int>(), \
        seq_lens.mutable_data_ptr<int>(), \
        reinterpret_cast<half*>(O.mutable_data_ptr<torch::headeronly::Half>()), \
        0, 0, H, num_kv_heads, (float)scale, causal, \
        block_size, max_blocks, (float)k_scale, (float)v_scale, \
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), \
        1, nullptr, nullptr, nullptr);

  #define DISPATCH_BATCHED(HD_VAL) \
    if (mode == 2) { LAUNCH_BATCHED_KERNEL(2, HD_VAL) } \
    else if (mode == 3) { LAUNCH_BATCHED_KERNEL(3, HD_VAL) } \
    else if (mode == 1) { LAUNCH_BATCHED_KERNEL(1, HD_VAL) } \
    else { LAUNCH_BATCHED_KERNEL(0, HD_VAL) }

  if (head_dim == 128) { DISPATCH_BATCHED(128) }
  else if (head_dim == 256) { DISPATCH_BATCHED(256) }
  #undef DISPATCH_BATCHED
  #undef LAUNCH_BATCHED_KERNEL

  return O;
}

torch::stable::Tensor flash_attn_sm70_prefill(
    torch::stable::Tensor Q, torch::stable::Tensor K,
    torch::stable::Tensor V, double scale, bool causal) {
  int B = Q.size(0);
  int H = Q.size(1);
  int Sq = Q.size(2);
  int Skv = K.size(2);

  const torch::stable::accelerator::DeviceGuard device_guard(
      Q.get_device_index());

  auto O = torch::stable::empty({B, H, Sq, vllm::sm70_attn::HD},
                                Q.scalar_type(), std::nullopt, Q.device());

  dim3 grid((Sq + vllm::sm70_attn::BM - 1) / vllm::sm70_attn::BM, H, B);
  const cudaStream_t stream = get_current_cuda_stream();

  vllm::sm70_attn::flash_attn_sm70_prefill_kernel<false><<<grid, 128, 0, stream>>>(
      reinterpret_cast<const half*>(Q.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(K.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(V.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<half*>(O.mutable_data_ptr<torch::headeronly::Half>()),
      Sq, Skv, H, (float)scale, causal, 1.0f, 1.0f);

  return O;
}

torch::stable::Tensor flash_attn_sm70_prefill_fp8(
    torch::stable::Tensor Q, torch::stable::Tensor K_fp8,
    torch::stable::Tensor V_fp8, double scale, bool causal,
    double k_scale, double v_scale) {
  int B = Q.size(0);
  int H = Q.size(1);
  int Sq = Q.size(2);
  int Skv = K_fp8.size(2);

  const torch::stable::accelerator::DeviceGuard device_guard(
      Q.get_device_index());

  auto O = torch::stable::empty({B, H, Sq, vllm::sm70_attn::HD},
                                Q.scalar_type(), std::nullopt, Q.device());

  dim3 grid((Sq + vllm::sm70_attn::BM - 1) / vllm::sm70_attn::BM, H, B);
  const cudaStream_t stream = get_current_cuda_stream();

  vllm::sm70_attn::flash_attn_sm70_prefill_kernel<true><<<grid, 128, 0, stream>>>(
      reinterpret_cast<const half*>(Q.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(K_fp8.mutable_data_ptr<uint8_t>()),
      reinterpret_cast<const half*>(V_fp8.mutable_data_ptr<uint8_t>()),
      reinterpret_cast<half*>(O.mutable_data_ptr<torch::headeronly::Half>()),
      Sq, Skv, H, (float)scale, causal, (float)k_scale, (float)v_scale);

  return O;
}
