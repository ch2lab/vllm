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
    float* __restrict__ max_partial, float* __restrict__ sum_partial,
    int xqa) {

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
    if (xqa) {
      // head_idx is the KV head; process q_per_kv query heads sharing it.
      const int q_per_kv = num_heads / num_kv_heads;
      q_ptr = Q + (q_base * num_heads + head_idx * q_per_kv) * HD;
      o_ptr = O + (q_base * num_heads + head_idx * q_per_kv) * HD;
      q_row_stride = HD;
    } else {
      q_ptr = Q + (q_base * num_heads + head_idx) * HD;
      o_ptr = O + (q_base * num_heads + head_idx) * HD;
      q_row_stride = num_heads * HD;
    }
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
  const int kv_head_idx =
      xqa ? head_idx : (head_idx / (num_heads / num_kv_heads));
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
    if (xqa) {
      // m is the query-head offset within the shared KV-head group; all
      // q_per_kv heads query the same (single) token position.
      const int q_per_kv = num_heads / num_kv_heads;
      sQ[m][d] = (m < q_per_kv) ? q_tile[m * HD + d] : __float2half(0.0f);
    } else {
      int gq = q_start + m;
      sQ[m][d] = (gq < Sq) ? q_tile[m * q_row_stride + d] : __float2half(0.0f);
    }
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
      // NVFP4: per-head slot = [fp4 data (HD/2 B) | fp8 scales (HD/16 B)];
      // K slots are the first num_kv_heads rows, V rows follow. Host-passed
      // strides make the addressing layout-agnostic (LBHNC).
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
          int64_t koff = (int64_t)blk * stride_block +
              (int64_t)kv_head_idx * stride_head +
              (int64_t)off * stride_slot;
          uint raw = *reinterpret_cast<const uint*>(
              side + koff + d8 / 2);
          float bs = __half2float(fp8e4m3_to_half_scaled(
              side[koff + DD + d8 / 16], k_scale));
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
      if (causal && (kv_start + n > q_offset + q_start + (xqa ? 0 : m)))
        val = -1e30f;
      sS[m][n] = val;
    }
    __syncthreads();

    // Softmax + rescale
    for (int m = threadIdx.x; m < BM; m += 128) {
      float old_max = s_max[m];
      float new_max = old_max;
      #pragma unroll
      for (int n = 0; n < BN; n++) new_max = fmaxf(new_max, sS[m][n]);
      float rescale = __expf(old_max - new_max);
      float new_sum = 0.0f;
      #pragma unroll
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
      // NVFP4: V slots are the num_kv_heads rows after the K rows; each
      // slot = [fp4 data (HD/2 B) | fp8 scales (HD/16 B)] (LBHNC).
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
          int64_t koff = (int64_t)blk * stride_block +
              (int64_t)kv_head_idx * stride_head +
              (int64_t)off * stride_slot;
          uint raw = *reinterpret_cast<const uint*>(
              side + koff + d8 / 2);
          float bs = __half2float(fp8e4m3_to_half_scaled(
              side[koff + DD + d8 / 16], v_scale));
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
    if (xqa) {
      const int q_per_kv = num_heads / num_kv_heads;
      if (m < q_per_kv) {
        float sum = s_sum[m];
        o_tile[m * HD + d] =
            __float2half(sum > 0.0f ? sO[m][d] / sum : 0.0f);
      }
    } else {
      int gq = q_start + m;
      if (gq < Sq) {
        float sum = s_sum[m];
        o_tile[m * q_row_stride + d] =
            __float2half(sum > 0.0f ? sO[m][d] / sum : 0.0f);
      }
    }
  }

  if (num_partitions > 1) {
    // Write unscaled partial (out, max, sum) for the log-sum-exp merge kernel.
    const int part_heads = xqa ? num_kv_heads : num_heads;
    const int pidx =
        (seq * num_partitions + partition_idx) * part_heads + head_idx;
    half* op = o_partial + (int64_t)pidx * BM * HD;
    float* mp = max_partial + (int64_t)pidx * BM;
    float* sp = sum_partial + (int64_t)pidx * BM;
    const int q_per_kv = num_heads / num_kv_heads;
    for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
      int m = idx / HD, d = idx % HD;
      if (xqa ? (m < q_per_kv) : (q_start + m < Sq))
        op[m * HD + d] = __float2half(sO[m][d]);
    }
    for (int m = threadIdx.x; m < BM; m += 128)
      if (xqa ? (m < q_per_kv) : (q_start + m < Sq)) {
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
    int num_heads, int num_kv_heads, int num_partitions, int BM, int HD,
    int xqa) {
  const int seq = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.x;
  const int q_start = q_tile_idx * BM;
  const int q_base = query_start_loc[seq];
  const int Sq = query_start_loc[seq + 1] - q_base;
  if (q_start >= Sq) return;

  if (xqa) {
    // XQA: this block produces one query head. Its partial lives under the
    // shared KV head, at row m = head_idx % q_per_kv.
    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = head_idx / q_per_kv;
    const int m = head_idx % q_per_kv;
    const int gq = q_start;
    float global_max = -1e30f;
    int pidx[MAX_PARTITIONS];
    for (int p = 0; p < num_partitions; p++) {
      pidx[p] = (seq * num_partitions + p) * num_kv_heads + kv_head;
      global_max = fmaxf(global_max, max_partial[pidx[p] * BM + m]);
    }
    float den = 0.0f;
    float w[MAX_PARTITIONS];
    for (int p = 0; p < num_partitions; p++) {
      w[p] = __expf(max_partial[pidx[p] * BM + m] - global_max);
      den += w[p] * sum_partial[pidx[p] * BM + m];
    }
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;
    half* o = O + ((int64_t)(q_base + gq) * num_heads + head_idx) * HD;
    for (int d = 0; d < HD; d++) {
      float num_d = 0.0f;
      for (int p = 0; p < num_partitions; p++)
        num_d += w[p] * __half2float(
            o_partial[((int64_t)pidx[p] * BM + m) * HD + d]);
      o[d] = __float2half(num_d * inv);
    }
    return;
  }

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
    int64_t kv_mode, int64_t xqa) {
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
  const int part_heads = (int)xqa ? num_kv_heads : H;

  const torch::stable::accelerator::DeviceGuard device_guard(
      Q.get_device_index());

  const int64_t total_partial = (int64_t)num_seqs * P * part_heads * BM;
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

  dim3 grid((max_query_len + BM - 1) / BM, part_heads, num_seqs * P);
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
        max_partial.mutable_data_ptr<float>(), sum_partial.mutable_data_ptr<float>(), \
        (int)xqa); \
    vllm::sm70_attn::flash_attn_sm70_decode_merge_kernel \
        <<<mgrid, 128, 0, stream>>>( \
        reinterpret_cast<const half*>(o_partial.mutable_data_ptr<torch::headeronly::Half>()), \
        max_partial.mutable_data_ptr<float>(), sum_partial.mutable_data_ptr<float>(), \
        query_start_loc.mutable_data_ptr<int>(), \
        reinterpret_cast<half*>(O.mutable_data_ptr<torch::headeronly::Half>()), \
        H, num_kv_heads, P, BM, HD_VAL, (int)xqa)

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
        1, nullptr, nullptr, nullptr, 0);

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
        1, nullptr, nullptr, nullptr, 0);

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

// ============================================================
// FLA (Flash Linear Attention) SM70 WMMA kernels
// Triton tl.dot does NOT emit WMMA on SM70 (verified: 0 wmma
// instructions in generated PTX). These CUDA kernels use WMMA
// 16x16x16 fp16 tensor cores directly, enabling ~3-5x speedup
// for GDN layer prefill (48/64 = 75% of model layers).
// ============================================================

namespace vllm {
namespace sm70_fla {

constexpr int FLA_BT = 64;
constexpr int FLA_BK = 128;
constexpr int FLA_SK = FLA_BK + 8;

// K @ K^T kernel: A = beta * K @ K^T * gate, with causal mask.
// Grid: (NT, H), Block: 128 (4 warps), Shared: ~34KB.
// Each warp handles 1 M-tile (16 rows) x 4 N-tiles (64 cols).
// K is loaded once into shared and used for both A (row_major)
// and B=K^T (col_major) WMMA operands.
__global__ __launch_bounds__(128, 1)
void fla_kkt_kernel(
    const half* __restrict__ k,
    const half* __restrict__ beta,
    const half* __restrict__ g,
    float* __restrict__ A,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ chunk_indices,
    int H, int Hg, int K) {
  const int i_t = blockIdx.x;
  const int i_h = blockIdx.y;

  int i_n = chunk_indices[i_t * 2];
  int i_tc = chunk_indices[i_t * 2 + 1];
  int bos = cu_seqlens[i_n];
  int eos = cu_seqlens[i_n + 1];
  int T = eos - bos;

  const int hg = i_h / (H / Hg);
  const int HgK = Hg * K;

  const half* k_head = k + (bos * Hg + hg) * K;
  const half* beta_head = beta + bos * H + i_h;
  const half* g_head = g + bos * H + i_h;
  float* A_head = A + (bos * H + i_h) * FLA_BT;

  const int warp_id = threadIdx.x / 32;
  const int m_tile = warp_id;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) wmma::fill_fragment(acc[i], 0.0f);

  __shared__ half sK[FLA_BT][FLA_SK];
  __shared__ float sA[FLA_BT][FLA_BT];
  __shared__ float sG[FLA_BT];
  __shared__ float sBeta[FLA_BT];

  for (int idx = threadIdx.x; idx < FLA_BT; idx += 128) {
    int gt = i_tc * FLA_BT + idx;
    sG[idx] = (gt < T) ? __half2float(g_head[gt * H]) : -1e30f;
    sBeta[idx] = (gt < T) ? __half2float(beta_head[gt * H]) : 0.0f;
  }
  __syncthreads();

  for (int k_base = 0; k_base < K; k_base += FLA_BK) {
    int actual_BK = min(FLA_BK, K - k_base);
    for (int idx = threadIdx.x; idx < FLA_BT * FLA_BK; idx += 128) {
      int t = idx / FLA_BK;
      int d = idx % FLA_BK;
      int gt = i_tc * FLA_BT + t;
      sK[t][d] = (gt < T && d < actual_BK)
                     ? k_head[gt * HgK + k_base + d]
                     : __float2half(0.0f);
    }
    __syncthreads();

    int n_k_tiles = (actual_BK + 15) / 16;
    for (int k_tile = 0; k_tile < n_k_tiles; k_tile++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                     wmma::row_major> a_frag;
      wmma::load_matrix_sync(a_frag, &sK[m_tile * 16][k_tile * 16],
                             FLA_SK);
      #pragma unroll
      for (int n_tile = 0; n_tile < 4; n_tile++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                       wmma::col_major> b_frag;
        wmma::load_matrix_sync(b_frag, &sK[n_tile * 16][k_tile * 16],
                               FLA_SK);
        wmma::mma_sync(acc[n_tile], a_frag, b_frag, acc[n_tile]);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int n_tile = 0; n_tile < 4; n_tile++)
    wmma::store_matrix_sync(&sA[m_tile * 16][n_tile * 16], acc[n_tile],
                            FLA_BT, wmma::mem_row_major);
  __syncthreads();

  for (int idx = threadIdx.x; idx < FLA_BT * FLA_BT; idx += 128) {
    int m = idx / FLA_BT;
    int n = idx % FLA_BT;
    int gm = i_tc * FLA_BT + m;
    int gn = i_tc * FLA_BT + n;
    float val = 0.0f;
    if (gm < T && gn < T && gm > gn)
      val = sA[m][n] * sBeta[m] * __expf(sG[m] - sG[n]);
    A_head[gm * (H * FLA_BT) + n] = val;
  }
}

}  // namespace sm70_fla
}  // namespace vllm

torch::stable::Tensor fla_kkt_sm70(
    torch::stable::Tensor k,
    torch::stable::Tensor beta,
    torch::stable::Tensor g,
    torch::stable::Tensor cu_seqlens,
    torch::stable::Tensor chunk_indices,
    int64_t NT_total) {
  int B = k.size(0);
  int T_total = k.size(1);
  int Hg = k.size(2);
  int K = k.size(3);
  int H = beta.size(2);
  int NT = (int)NT_total;

  const torch::stable::accelerator::DeviceGuard device_guard(
      k.get_device_index());

  auto A = torch::stable::empty({B, T_total, H, vllm::sm70_fla::FLA_BT},
      torch::stable::ScalarType::Float, std::nullopt, k.device());

  dim3 grid(NT, H);
  const cudaStream_t stream = get_current_cuda_stream();

  vllm::sm70_fla::fla_kkt_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<const half*>(k.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(beta.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(g.mutable_data_ptr<torch::headeronly::Half>()),
      A.mutable_data_ptr<float>(),
      cu_seqlens.mutable_data_ptr<int>(),
      chunk_indices.mutable_data_ptr<int>(),
      H, Hg, K);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("fla_kkt_sm70 launch failed: %s grid=(%d,%d) block=128 smem=0\n",
           cudaGetErrorString(err), (int)grid.x, (int)grid.y);
    assert(false);
  }

  return A;
}

// WY representation kernel: u = A @ (v*beta), w = A @ (k*beta*exp(g))
// Grid: (NT, H), Block: 128 (4 warps), Shared: ~35KB
namespace vllm {
namespace sm70_fla {

constexpr int WY_BT = 64;
constexpr int WY_BV = 64;
constexpr int WY_BK = 64;
constexpr int WY_SV = WY_BV + 8;
constexpr int WY_SK = WY_BK + 8;

__global__ __launch_bounds__(128, 1)
void fla_wy_kernel(
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ beta,
    const half* __restrict__ g,
    const float* __restrict__ A,
    half* __restrict__ w_out,
    half* __restrict__ u_out,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ chunk_indices,
    int H, int Hg, int K, int V) {
  const int i_t = blockIdx.x;
  const int i_h = blockIdx.y;

  int i_n = chunk_indices[i_t * 2];
  int i_tc = chunk_indices[i_t * 2 + 1];
  int bos = cu_seqlens[i_n];
  int eos = cu_seqlens[i_n + 1];
  int T = eos - bos;

  const int hg = i_h / (H / Hg);
  const int warp_id = threadIdx.x / 32;
  const int m_tile = warp_id;

  const half* k_head = k + (bos * Hg + hg) * K;
  const half* v_head = v + (bos * H + i_h) * V;
  const half* beta_head = beta + bos * H + i_h;
  const half* g_head = g + bos * H + i_h;
  const float* A_head = A + (bos * H + i_h) * WY_BT;
  half* w_head = w_out + (bos * H + i_h) * K;
  half* u_head = u_out + (bos * H + i_h) * V;

  __shared__ half sA[WY_BT][WY_BT + 8];
  __shared__ half sB[WY_BT][WY_SV];
  __shared__ float sOut[WY_BT][WY_BT];
  __shared__ float sBeta[WY_BT];
  __shared__ float sGexp[WY_BT];

  // Load A (fp32 → fp16) and beta, g
  for (int idx = threadIdx.x; idx < WY_BT * WY_BT; idx += 128) {
    int m = idx / WY_BT, n = idx % WY_BT;
    int gm = i_tc * WY_BT + m;
    sA[m][n] = (gm < T) ? __float2half(A_head[gm * (H * WY_BT) + n])
                        : __float2half(0.0f);
  }
  for (int idx = threadIdx.x; idx < WY_BT; idx += 128) {
    int gt = i_tc * WY_BT + idx;
    sBeta[idx] = (gt < T) ? __half2float(beta_head[gt * H]) : 0.0f;
    float gv = (gt < T) ? __half2float(g_head[gt * H]) : 0.0f;
    sGexp[idx] = expf(gv);
  }
  __syncthreads();

  // Compute u = A @ (v * beta) for each V-tile
  for (int i_v = 0; i_v < V; i_v += WY_BV) {
    int actual_BV = min(WY_BV, V - i_v);
    // Load v and scale by beta
    for (int idx = threadIdx.x; idx < WY_BT * WY_BV; idx += 128) {
      int t = idx / WY_BV, d = idx % WY_BV;
      int gt = i_tc * WY_BT + t;
      half vh = (gt < T && d < actual_BV)
                    ? v_head[gt * (H * V) + i_v + d]
                    : __float2half(0.0f);
      sB[t][d] = __float2half(__half2float(vh) * sBeta[t]);
    }
    __syncthreads();

    // WMMA: u = sA @ sB
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(acc[i], 0.0f);

    int n_k_tiles = (WY_BT + 15) / 16;
    for (int kt = 0; kt < n_k_tiles; kt++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                     wmma::row_major> a_frag;
      wmma::load_matrix_sync(a_frag, &sA[m_tile * 16][kt * 16], WY_BT + 8);
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        if (nt * 16 >= actual_BV) break;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                       wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, &sB[kt * 16][nt * 16], WY_SV);
        wmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
      }
    }
    __syncthreads();

    // Store and convert to fp16
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      if (nt * 16 >= actual_BV) break;
      wmma::store_matrix_sync(&sOut[m_tile * 16][nt * 16], acc[nt],
                              WY_BT, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < WY_BT * WY_BV; idx += 128) {
      int t = idx / WY_BV, d = idx % WY_BV;
      int gt = i_tc * WY_BT + t;
      if (gt < T && d < actual_BV)
        u_head[gt * (H * V) + i_v + d] = __float2half(sOut[t][d]);
    }
    __syncthreads();
  }

  // Compute w = A @ (k * beta * exp(g)) for each K-tile
  for (int i_k = 0; i_k < K; i_k += WY_BK) {
    int actual_BK = min(WY_BK, K - i_k);
    // Load k and scale by beta * exp(g)
    for (int idx = threadIdx.x; idx < WY_BT * WY_BK; idx += 128) {
      int t = idx / WY_BK, d = idx % WY_BK;
      int gt = i_tc * WY_BT + t;
      half kh = (gt < T && d < actual_BK)
                    ? k_head[gt * (Hg * K) + i_k + d]
                    : __float2half(0.0f);
      sB[t][d] = __float2half(__half2float(kh) * sBeta[t] * sGexp[t]);
    }
    __syncthreads();

    // WMMA: w = sA @ sB
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(acc[i], 0.0f);

    int n_k_tiles = (WY_BT + 15) / 16;
    for (int kt = 0; kt < n_k_tiles; kt++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                     wmma::row_major> a_frag;
      wmma::load_matrix_sync(a_frag, &sA[m_tile * 16][kt * 16], WY_BT + 8);
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        if (nt * 16 >= actual_BK) break;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                       wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, &sB[kt * 16][nt * 16], WY_SV);
        wmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
      }
    }
    __syncthreads();

    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      if (nt * 16 >= actual_BK) break;
      wmma::store_matrix_sync(&sOut[m_tile * 16][nt * 16], acc[nt],
                              WY_BT, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < WY_BT * WY_BK; idx += 128) {
      int t = idx / WY_BK, d = idx % WY_BK;
      int gt = i_tc * WY_BT + t;
      if (gt < T && d < actual_BK)
        w_head[gt * (H * K) + i_k + d] = __float2half(sOut[t][d]);
    }
    __syncthreads();
  }
}

}  // namespace sm70_fla
}  // namespace vllm

std::tuple<torch::stable::Tensor, torch::stable::Tensor> fla_wy_sm70(
    torch::stable::Tensor k, torch::stable::Tensor v,
    torch::stable::Tensor beta, torch::stable::Tensor g,
    torch::stable::Tensor A, torch::stable::Tensor cu_seqlens,
    torch::stable::Tensor chunk_indices, int64_t NT_total) {
  int B = k.size(0);
  int T_total = k.size(1);
  int Hg = k.size(2);
  int K = k.size(3);
  int H = v.size(2);
  int V = v.size(3);
  int NT = (int)NT_total;

  const torch::stable::accelerator::DeviceGuard device_guard(
      k.get_device_index());

  auto w = torch::stable::empty({B, T_total, H, K}, k.scalar_type(),
                                 std::nullopt, k.device());
  auto u = torch::stable::empty({B, T_total, H, V}, v.scalar_type(),
                                 std::nullopt, v.device());

  dim3 grid(NT, H);
  const cudaStream_t stream = get_current_cuda_stream();

  vllm::sm70_fla::fla_wy_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<const half*>(k.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(v.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(beta.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(g.mutable_data_ptr<torch::headeronly::Half>()),
      A.mutable_data_ptr<float>(),
      reinterpret_cast<half*>(w.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<half*>(u.mutable_data_ptr<torch::headeronly::Half>()),
      cu_seqlens.mutable_data_ptr<int>(),
      chunk_indices.mutable_data_ptr<int>(),
      H, Hg, K, V);

  return {w, u};
}

// delta_h kernel: recurrent state update with WMMA
// v = sum_i(w_i @ h_i^T), v_new = u - v (gated), h_i += (k_i @ v_new)^T
// Grid: (cdiv(V,32), N*H), Block: 128 (4 warps), Shared: ~41KB
namespace vllm {
namespace sm70_fla {

constexpr int DH_BV = 32;
constexpr int DH_BT = 64;
constexpr int DH_KS = 64;  // K sub-block size
constexpr int DH_SData = DH_KS + 8;

__global__ __launch_bounds__(128, 1)
void fla_delta_h_kernel(
    const half* __restrict__ k,      // [T_total, Hg, K]
    const half* __restrict__ w,      // [T_total, H, K]
    const half* __restrict__ u,      // [T_total, H, V] (original v from WY)
    const half* __restrict__ g,      // [T_total, H] (cumsum)
    half* __restrict__ v_new,        // [T_total, H, V]
    half* __restrict__ h_out,        // [NT_total, H, V, K]
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ chunk_offsets,
    int H, int Hg, int K, int V) {
  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_h = i_nh % H;
  const int i_n = i_nh / H;

  int bos = cu_seqlens[i_n];
  int eos = cu_seqlens[i_n + 1];
  int T = eos - bos;
  int NT = (T + DH_BT - 1) / DH_BT;
  int boh = chunk_offsets[i_n];

  const int hg = i_h / (H / Hg);
  const int HgK = Hg * K;
  const int HK = H * K;
  const int HV = H * V;
  const int HVK = H * V * K;

  const half* k_head = k + (bos * Hg + hg) * K;
  const half* w_head = w + (bos * H + i_h) * K;
  const half* u_head = u + (bos * H + i_h) * V;
  const half* g_head = g + bos * H + i_h;
  half* vn_head = v_new + (bos * H + i_h) * V;
  half* h_head = h_out + ((boh * H + i_h) * V) * K;

  const int warp_id = threadIdx.x / 32;

  __shared__ half sH1[DH_BV][DH_KS];
  __shared__ half sH2[DH_BV][DH_KS];
  __shared__ half sH3[DH_BV][DH_KS];
  __shared__ half sH4[DH_BV][DH_KS];
  __shared__ half sData[DH_BT][DH_SData];
  __shared__ float sV[DH_BT][DH_BV];
  __shared__ half sVh[DH_BT][DH_BV];
  __shared__ float sG[DH_BT];

  half* sH_ptrs[4] = {
    &sH1[0][0], &sH2[0][0], &sH3[0][0], &sH4[0][0]
  };

  // Zero state
  for (int idx = threadIdx.x; idx < DH_BV * DH_KS; idx += 128) {
    sH1[idx / DH_KS][idx % DH_KS] = __float2half(0.0f);
    sH2[idx / DH_KS][idx % DH_KS] = __float2half(0.0f);
    sH3[idx / DH_KS][idx % DH_KS] = __float2half(0.0f);
    sH4[idx / DH_KS][idx % DH_KS] = __float2half(0.0f);
  }
  __syncthreads();

  // V computation: each warp handles 1 M-tile (16 rows of BT=64), 2 N-tiles (BV=32)
  const int m_tile_v = warp_id;

  for (int i_t = 0; i_t < NT; i_t++) {
    int cs = i_t * DH_BT;
    int ce = min(cs + DH_BT, T);

    // Store current state to h_out
    for (int sub = 0; sub < 4; sub++) {
      for (int idx = threadIdx.x; idx < DH_BV * DH_KS; idx += 128) {
        int r = idx / DH_KS, c = idx % DH_KS;
        h_head[i_t * HVK + r * K + sub * DH_KS + c] = sH_ptrs[sub][r * DH_KS + c];
      }
    }
    __syncthreads();

    // Compute v = sum_i(w_i @ h_i^T)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> v_acc[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) wmma::fill_fragment(v_acc[i], 0.0f);

    for (int sub = 0; sub < 4; sub++) {
      // Load w[64, 64] into sData
      for (int idx = threadIdx.x; idx < DH_BT * DH_KS; idx += 128) {
        int t = idx / DH_KS, d = idx % DH_KS;
        int gt = cs + t;
        sData[t][d] = (gt < T) ? w_head[gt * HK + sub * DH_KS + d]
                              : __float2half(0.0f);
      }
      __syncthreads();

      // v += w @ h^T  (w[64,64] @ h^T[64,32] → v[64,32])
      // A=w row_major, B=h^T col_major from sH[sub]
      for (int kt = 0; kt < 4; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                       wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, &sData[m_tile_v * 16][kt * 16],
                               DH_SData);
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                         wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag,
                                 &sH_ptrs[sub][nt * 16 * DH_KS + kt * 16],
                                 DH_KS);
          wmma::mma_sync(v_acc[nt], a_frag, b_frag, v_acc[nt]);
        }
      }
      __syncthreads();
    }

    // Store v to sV
    #pragma unroll
    for (int nt = 0; nt < 2; nt++)
      wmma::store_matrix_sync(&sV[m_tile_v * 16][nt * 16], v_acc[nt],
                              DH_BV, wmma::mem_row_major);
    __syncthreads();

    // v_new = u - v
    for (int idx = threadIdx.x; idx < DH_BT * DH_BV; idx += 128) {
      int t = idx / DH_BV, d = idx % DH_BV;
      int gt = cs + t;
      float u_val = (gt < T) ? __half2float(u_head[gt * HV + i_v * DH_BV + d]) : 0.0f;
      sV[t][d] = u_val - sV[t][d];
    }
    __syncthreads();

    // Save v_new
    for (int idx = threadIdx.x; idx < DH_BT * DH_BV; idx += 128) {
      int t = idx / DH_BV, d = idx % DH_BV;
      int gt = cs + t;
      if (gt < T)
        vn_head[gt * HV + i_v * DH_BV + d] = __float2half(sV[t][d]);
    }

    // Gate: v *= exp(g_last - g), h *= exp(g_last)
    int last_idx = min(ce, T) - 1;
    float g_last = __half2float(g_head[last_idx * H]);
    float g_last_exp = expf(g_last);

    for (int idx = threadIdx.x; idx < DH_BT; idx += 128) {
      int gt = cs + idx;
      sG[idx] = (gt < T) ? __half2float(g_head[gt * H]) : 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < DH_BT * DH_BV; idx += 128) {
      int t = idx / DH_BV, d = idx % DH_BV;
      int gt = cs + t;
      if (gt < T)
        sV[t][d] *= expf(g_last - sG[t]);
    }

    for (int sub = 0; sub < 4; sub++) {
      for (int idx = threadIdx.x; idx < DH_BV * DH_KS; idx += 128) {
        int r = idx / DH_KS, c = idx % DH_KS;
        float v = __half2float(sH_ptrs[sub][r * DH_KS + c]) * g_last_exp;
        sH_ptrs[sub][r * DH_KS + c] = __float2half(v);
      }
    }
    __syncthreads();

    // Convert sV to fp16 for WMMA
    for (int idx = threadIdx.x; idx < DH_BT * DH_BV; idx += 128) {
      int t = idx / DH_BV, d = idx % DH_BV;
      sVh[t][d] = __float2half(sV[t][d]);
    }
    __syncthreads();

    // h update: h_i += (k_i @ v_new)^T = v_new^T @ k_i^T
    // v_new^T[32, 64] from sV[64][32] col_major
    // k^T[64, 64] from sData[64][72] col_major
    // h_update[32, 64]: M=2, N=4, K=4
    // 4 warps: 2 M-tiles × 2 N-tile-groups
    int m_tile_h = warp_id / 2;  // 0 or 1
    int n_start = (warp_id % 2) * 2;  // 0 or 2

    for (int sub = 0; sub < 4; sub++) {
      // Load k[64, 64] into sData
      for (int idx = threadIdx.x; idx < DH_BT * DH_KS; idx += 128) {
        int t = idx / DH_KS, d = idx % DH_KS;
        int gt = cs + t;
        sData[t][d] = (gt < T) ? k_head[gt * HgK + sub * DH_KS + d]
                              : __float2half(0.0f);
      }
      __syncthreads();

      wmma::fragment<wmma::accumulator, 16, 16, 16, float> h_acc[2];
      #pragma unroll
      for (int i = 0; i < 2; i++) wmma::fill_fragment(h_acc[i], 0.0f);

      for (int kt = 0; kt < 4; kt++) {
        // A = v^T[16, 16] from sVh, col_major
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                       wmma::col_major> a_frag;
        wmma::load_matrix_sync(a_frag,
                               &sVh[kt * 16][m_tile_h * 16],
                               DH_BV);
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
          int n_tile = n_start + nt;
          // B = k^T[16, 16] from sData, col_major
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                         wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag,
                                 &sData[n_tile * 16][kt * 16],
                                 DH_SData);
          wmma::mma_sync(h_acc[nt], a_frag, b_frag, h_acc[nt]);
        }
      }
      __syncthreads();

      // Store h_update to sV (reused as flat buffer), add to state
      float* sV_flat = &sV[0][0];
      #pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        int n_tile = n_start + nt;
        wmma::store_matrix_sync(&sV_flat[(m_tile_h * 16) * DH_KS + n_tile * 16],
                                h_acc[nt], DH_KS, wmma::mem_row_major);
      }
      __syncthreads();

      // Add h_update to state
      for (int idx = threadIdx.x; idx < DH_BV * DH_KS; idx += 128) {
        int r = idx / DH_KS, c = idx % DH_KS;
        float old = __half2float(sH_ptrs[sub][r * DH_KS + c]);
        sH_ptrs[sub][r * DH_KS + c] = __float2half(old + sV_flat[r * DH_KS + c]);
      }
      __syncthreads();
    }
  }
}

}  // namespace sm70_fla
}  // namespace vllm

std::tuple<torch::stable::Tensor, torch::stable::Tensor,
           torch::stable::Tensor>
fla_delta_h_sm70(
    torch::stable::Tensor k, torch::stable::Tensor w,
    torch::stable::Tensor u, torch::stable::Tensor g,
    torch::stable::Tensor cu_seqlens,
    torch::stable::Tensor chunk_offsets,
    torch::stable::Tensor initial_state,
    bool output_final_state,
    int64_t NT_total) {
  int B = k.size(0);
  int T_total = k.size(1);
  int Hg = k.size(2);
  int K = k.size(3);
  int H = u.size(2);
  int V = u.size(3);
  int N = cu_seqlens.size(0) - 1;
  int NT = (int)NT_total;

  const torch::stable::accelerator::DeviceGuard device_guard(
      k.get_device_index());

  auto h = torch::stable::empty({B, NT, H, V, K}, k.scalar_type(),
                                std::nullopt, k.device());
  auto v_new = torch::stable::empty({B, T_total, H, V}, u.scalar_type(),
                                    std::nullopt, u.device());
  auto final_state = output_final_state
      ? torch::stable::empty({N, H, V, K},
              torch::stable::ScalarType::Float, std::nullopt, k.device())
      : torch::stable::empty({0}, torch::stable::ScalarType::Float,
                             std::nullopt, k.device());

  dim3 grid((V + 31) / 32, N * H);
  const cudaStream_t stream = get_current_cuda_stream();

  vllm::sm70_fla::fla_delta_h_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<const half*>(k.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(w.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(u.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(g.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<half*>(v_new.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<half*>(h.mutable_data_ptr<torch::headeronly::Half>()),
      cu_seqlens.mutable_data_ptr<int>(),
      chunk_offsets.mutable_data_ptr<int>(),
      H, Hg, K, V);

  return {h, v_new, final_state};
}
