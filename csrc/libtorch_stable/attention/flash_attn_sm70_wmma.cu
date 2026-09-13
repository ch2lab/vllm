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

__device__ __forceinline__ half fp8e5m2_to_half_scaled(uint8_t x,
                                                       float scale) {
  __half_raw hr = __nv_cvt_fp8_to_halfraw(x, __NV_E5M2);
  float val = __half2float(*reinterpret_cast<half*>(&hr)) * scale;
  return __float2half(val);
}

// Flash-attention prefill kernel for SM70
// BM=16, BN=32, HD=128
// Shared memory: ~27KB (under 48KB SM70 limit)
// Grid: (ceil(Sq/BM), num_heads, batch), Block: 128 threads

 constexpr int BM = 16;
 constexpr int BN = 32;
// Cap on the KV partition count for the decode path. Each partition is a
// CTA wave slot: with 2 KV heads x P partitions the grid must fill the 70
// V100 SMs. P=128 (up to 256 CTAs) with small segments maximizes latency
// hiding for the serial per-tile dequant/QK/PV chain at long contexts.
constexpr int MAX_PARTITIONS = 128;
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

// KV_MODE: 0 = FP16, 1 = FP8 e4m3, 2 = INT4, 3 = NVFP4, 4 = FP8 e5m2
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
  // sO is aliased onto sKV's backing memory (not a separate allocation).
  // The K tile in sKV is fully consumed by S=Q·K before the rescale writes
  // sO, and V is reloaded into sKV only after the rescale reloads o_frag —
  // so sO and the live K/V never coexist. This removes the separate
  // BM*HD*4 (=16KB at HD=256) allocation so the CTA's smem fits 3/SM.
  static_assert(sizeof(sKV) >= (size_t)BM * HD * sizeof(float),
                "sKV must be large enough to alias the sO accumulator");
  float* const sO = reinterpret_cast<float*>(sKV);
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
  // Graph-safe dynamic partitioning: the effective partition count is
  // derived from the live KV length (device data), so a graph captured
  // with the max grid still scans short contexts with ~1 partition while
  // long contexts keep the full split. Skipped partitions write empty
  // partials (max=-inf) which the log-sum-exp merge zero-weights.
  int P_eff = num_partitions;
  if (num_partitions > 1) {
    const int SEG_TOKENS = 1024;
    P_eff = min(num_partitions,
                max(1, (kv_end + SEG_TOKENS - 1) / SEG_TOKENS));
  }
  if (num_partitions > 1 && partition_idx >= P_eff) {
    const int part_heads_e = xqa ? num_kv_heads : num_heads;
    const int pidx_e =
        (seq * num_partitions + partition_idx) * part_heads_e + head_idx;
    half* op_e = o_partial + (int64_t)pidx_e * BM * HD;
    float* mp_e = max_partial + (int64_t)pidx_e * BM;
    float* sp_e = sum_partial + (int64_t)pidx_e * BM;
    const int q_per_kv_e = num_heads / num_kv_heads;
    for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
      const int m = idx / HD;
      if (xqa ? (m < q_per_kv_e) : (q_start + m < Sq))
        op_e[idx] = __float2half(0.0f);
    }
    for (int m = threadIdx.x; m < BM; m += 128)
      if (xqa ? (m < q_per_kv_e) : (q_start + m < Sq)) {
        mp_e[m] = -1e30f;
        sp_e[m] = 0.0f;
      }
    return;
  }
  const int seg_len = (kv_end + P_eff - 1) / P_eff;
  const int seg_start = min(partition_idx * seg_len, kv_end);
  const int seg_end = min(kv_end, seg_start + seg_len);

  for (int kv_start = seg_start; kv_start < seg_end; kv_start += BN) {
    int actual_bn = min(BN, kv_end - kv_start);

    // Load K tile from paged cache (vectorized: 8 elements per load)
    if constexpr (KV_MODE == 1 || KV_MODE == 4) {
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
            sKV[n][d8 + i] = (KV_MODE == 4)
                ? fp8e5m2_to_half_scaled(bytes[i], k_scale)
                : fp8e4m3_to_half_scaled(bytes[i], k_scale);
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
      wmma::store_matrix_sync(
          sO + (warp_id * TILES_PER_WARP * 16 + dt * 16),
          o_frag[dt], HD, wmma::mem_row_major);
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM * HD; idx += 128)
      sO[idx] *= s_rescale[idx / HD];
    __syncthreads();
    #pragma unroll
    for (int dt = 0; dt < TILES_PER_WARP; dt++)
      wmma::load_matrix_sync(
          o_frag[dt], sO + (warp_id * TILES_PER_WARP * 16 + dt * 16), HD,
          wmma::mem_row_major);

    // Load V tile from paged cache (vectorized, V at offset HD)
    if constexpr (KV_MODE == 1 || KV_MODE == 4) {
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
            sKV[n][d8 + i] = (KV_MODE == 4)
                ? fp8e5m2_to_half_scaled(bytes[i], v_scale)
                : fp8e4m3_to_half_scaled(bytes[i], v_scale);
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
    wmma::store_matrix_sync(
        sO + (warp_id * TILES_PER_WARP * 16 + dt * 16),
        o_frag[dt], HD, wmma::mem_row_major);
  __syncthreads();

  for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
    int m = idx / HD, d = idx % HD;
    if (xqa) {
      const int q_per_kv = num_heads / num_kv_heads;
      if (m < q_per_kv) {
        float sum = s_sum[m];
        o_tile[m * HD + d] =
            __float2half(sum > 0.0f ? sO[m * HD + d] / sum : 0.0f);
      }
    } else {
      int gq = q_start + m;
      if (gq < Sq) {
        float sum = s_sum[m];
        o_tile[m * q_row_stride + d] =
            __float2half(sum > 0.0f ? sO[m * HD + d] / sum : 0.0f);
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
    // sO holds the pre-last-tile intermediate; stage the final registers.
    #pragma unroll
    for (int dt = 0; dt < TILES_PER_WARP; dt++)
      wmma::store_matrix_sync(
          sO + (warp_id * TILES_PER_WARP * 16 + dt * 16),
          o_frag[dt], HD, wmma::mem_row_major);
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
      int m = idx / HD, d = idx % HD;
      if (xqa ? (m < q_per_kv) : (q_start + m < Sq))
        op[m * HD + d] = __float2half(sO[m * HD + d]);
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

  // Log-sum-exp merge of the P partition partials. The merge is independent
  // of the partition count (no fixed-size arrays), so P may be raised to
  // fill the V100 SMs. Each thread owns a contiguous D chunk of one query
  // row; per-row weights are recomputed per thread from the (L1-hot)
  // max/sum partials, avoiding shared memory entirely.
  const int q_per_kv = num_heads / num_kv_heads;
  if (xqa) {
    // XQA: one query head per block; its partial lives under the shared KV
    // head, at row m = head_idx % q_per_kv. One D element per thread
    // (HD == 128 == blockDim).
    const int kv_head = head_idx / q_per_kv;
    const int m = head_idx % q_per_kv;
    const int gq = q_start;
    float gmax = -1e30f;
    for (int p = 0; p < num_partitions; p++) {
      const int pidx = (seq * num_partitions + p) * num_kv_heads + kv_head;
      gmax = fmaxf(gmax, max_partial[pidx * BM + m]);
    }
    const half* op_row =
        o_partial + ((int64_t)((seq * num_partitions) * num_kv_heads + kv_head)
                       * BM +
                     m) *
                    HD;
    half* o = O + ((int64_t)(q_base + gq) * num_heads + head_idx) * HD;
    const int64_t part_stride = (int64_t)num_kv_heads * BM * HD;
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
      float den = 0.0f, num = 0.0f;
      for (int p = 0; p < num_partitions; p++) {
        const int pidx =
            ((seq * num_partitions + p) * num_kv_heads + kv_head) * BM + m;
        const float w = __expf(max_partial[pidx] - gmax);
        den += w * sum_partial[pidx];
        num += w * __half2float(op_row[(int64_t)p * part_stride + d]);
      }
      o[d] = __float2half((den > 0.0f ? num / den : 0.0f));
    }
    return;
  }

  // Batched: 128 threads = BM(16) rows x 8 chunks of HD/8-wide D. Thread t
  // owns row gm = t/8 and D range [ (t%8)*(HD/8), +HD/8 ).
  const int gm = threadIdx.x / 8;
  const int gq = q_start + gm;
  if (gq >= Sq) return;
  const int chunk = HD / 8;
  const int d0 = (threadIdx.x % 8) * chunk;
  float gmax = -1e30f;
  for (int p = 0; p < num_partitions; p++) {
    const int pidx = (seq * num_partitions + p) * num_heads + head_idx;
    gmax = fmaxf(gmax, max_partial[pidx * BM + gm]);
  }
  float den = 0.0f;
  float num[32];  // HD/8 with HD <= 256
  #pragma unroll
  for (int i = 0; i < 32; i++) num[i] = 0.0f;
  const int64_t part_stride = (int64_t)num_heads * BM * HD;
  const int64_t op_base =
      ((int64_t)seq * num_partitions * num_heads + head_idx) * BM * HD +
      gm * HD + d0;
  for (int p = 0; p < num_partitions; p++) {
    const int pidx = (seq * num_partitions + p) * num_heads + head_idx;
    const float w = __expf(max_partial[pidx * BM + gm] - gmax);
    den += w * sum_partial[pidx * BM + gm];
    const half* op = o_partial + op_base + (int64_t)p * part_stride;
    for (int i = 0; i < chunk; i++) num[i] += w * __half2float(op[i]);
  }
  const float inv = den > 0.0f ? 1.0f / den : 0.0f;
  half* o = O + ((int64_t)(q_base + gq) * num_heads + head_idx) * HD + d0;
  for (int i = 0; i < chunk; i++) o[i] = __float2half(num[i] * inv);
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
  if (mode == 3) num_kv_heads /= 2;  // nvfp4: dim1 holds K rows + V rows
  const int BM = vllm::sm70_attn::BM;
  int P = (int)num_partitions;
  if (P > vllm::sm70_attn::MAX_PARTITIONS) P = vllm::sm70_attn::MAX_PARTITIONS;
  if (P < 1) P = 1;
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
    if (P > 1) \
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
    else if (mode == 4) { LAUNCH_PARTED(4, 128); }
    else if (mode == 1) { LAUNCH_PARTED(1, 128); }
    else { LAUNCH_PARTED(0, 128); }
  } else {
    if (mode == 2) { LAUNCH_PARTED(2, 256); }
    else if (mode == 3) { LAUNCH_PARTED(3, 256); }
    else if (mode == 4) { LAUNCH_PARTED(4, 256); }
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
  if (mode == 3) num_kv_heads /= 2;  // nvfp4: dim1 holds K rows + V rows

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
    else if (mode == 4) { LAUNCH_BATCHED_KERNEL(4, HD_VAL) } \
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
    if (gm >= T) continue;
    float val = 0.0f;
    if (gn < T && gm > gn)
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

// chunk_o kernel: o = scale * (exp(g) * (q @ h^T) + tril(exp(g_m - g_n) *
// (q @ k^T)) @ v).  Replaces Triton chunk_fwd_kernel_o (tl.dot does not
// emit WMMA on SM70).
// Grid: (NT, H, V/64), Block: 128 (4 warps).  Each warp owns 16 rows of
// the chunk; all 3 matmuls run on 16x16x16 WMMA.  Shared ~76KB.
namespace vllm {
namespace sm70_fla {

constexpr int CO_BT = 64;
constexpr int CO_BV = 64;
constexpr int CO_BK = 128;
constexpr int CO_SQK = CO_BK + 8;   // padded stride for q/k/h tiles
constexpr int CO_SV = CO_BV + 8;    // padded stride for v/Ag tiles

__global__ __launch_bounds__(128, 1)
void fla_chunk_o_kernel(
    const half* __restrict__ q,   // [T_total, Hg, K]
    const half* __restrict__ k,   // [T_total, Hg, K]
    const half* __restrict__ v,   // [T_total, H, V]
    const half* __restrict__ h,   // [NT_total, H, V, K]
    const half* __restrict__ g,   // [T_total, H] chunk-local cumsum
    half* __restrict__ o,         // [T_total, H, V]
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ chunk_indices,
    int H, int Hg, int K, int V, float scale) {
  const int i_t = blockIdx.x;
  const int i_h = blockIdx.y;
  const int i_v = blockIdx.z;

  const int i_n = chunk_indices[i_t * 2];
  const int i_tc = chunk_indices[i_t * 2 + 1];
  const int bos = cu_seqlens[i_n];
  const int eos = cu_seqlens[i_n + 1];
  const int T = eos - bos;

  const int hg = i_h / (H / Hg);
  const int HgK = Hg * K;
  const int HV = H * V;

  const half* q_head = q + (size_t)bos * HgK + hg * K;
  const half* k_head = k + (size_t)bos * HgK + hg * K;
  const half* v_head = v + (size_t)bos * HV + i_h * V + i_v * CO_BV;
  const half* g_head = g + (size_t)bos * H + i_h;
  const half* h_ch = h + ((size_t)i_t * H + i_h) * V * K + i_v * CO_BV * K;
  half* o_head = o + (size_t)bos * HV + i_h * V + i_v * CO_BV;

  const int warp_id = threadIdx.x / 32;
  const int m_base = warp_id * 16;

  // Shared budget (~42KB): sQ holds q (gated in place after pass 1), sB is
  // reused for k tiles, h tiles, v tile and the final result.
  __shared__ half sQ[CO_BT][CO_SQK];
  __shared__ half sB[CO_BT][CO_SQK];
  __shared__ half sA[CO_BT][CO_BT];
  __shared__ float sG[CO_BT];

  for (int idx = threadIdx.x; idx < CO_BT; idx += 128) {
    const int gt = i_tc * CO_BT + idx;
    sG[idx] = (gt < T) ? __half2float(g_head[(size_t)gt * H]) : -1e30f;
  }
  for (int idx = threadIdx.x; idx < CO_BT * CO_BK; idx += 128) {
    const int t = idx / CO_BK;
    const int d = idx % CO_BK;
    const int gt = i_tc * CO_BT + t;
    sQ[t][d] = (gt < T && d < K) ? q_head[(size_t)gt * HgK + d]
                                 : __float2half(0.0f);
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_A[4];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    wmma::fill_fragment(acc_A[i], 0.0f);
    wmma::fill_fragment(acc_o[i], 0.0f);
  }

  // Pass 1: A = q @ k^T (ungated).
  for (int k_base = 0; k_base < K; k_base += CO_BK) {
    const int actual_BK = min(CO_BK, K - k_base);
    for (int idx = threadIdx.x; idx < CO_BT * CO_BK; idx += 128) {
      const int t = idx / CO_BK;
      const int d = idx % CO_BK;
      const int gt = i_tc * CO_BT + t;
      sB[t][d] = (gt < T && d < actual_BK)
                     ? k_head[(size_t)gt * HgK + k_base + d]
                     : __float2half(0.0f);
    }
    __syncthreads();
    const int n_k_tiles = (actual_BK + 15) / 16;
    for (int kt = 0; kt < n_k_tiles; kt++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fq;
      wmma::load_matrix_sync(fq, &sQ[m_base][kt * 16], CO_SQK);
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
        wmma::load_matrix_sync(fb, &sB[nt * 16][kt * 16], CO_SQK);
        wmma::mma_sync(acc_A[nt], fq, fb, acc_A[nt]);
      }
    }
    __syncthreads();
  }

  // Gate q rows in place: inter-chunk term = (q*exp(g)) @ h^T.
  for (int idx = threadIdx.x; idx < CO_BT * CO_BK; idx += 128) {
    const int t = idx / CO_BK;
    const int d = idx % CO_BK;
    if (d < K)
      sQ[t][d] = __float2half(__half2float(sQ[t][d]) * __expf(sG[t]));
  }
  __syncthreads();

  // Pass 2: o_inter = (q*exp(g)) @ h^T.
  for (int k_base = 0; k_base < K; k_base += CO_BK) {
    const int actual_BK = min(CO_BK, K - k_base);
    for (int idx = threadIdx.x; idx < CO_BT * CO_BK; idx += 128) {
      const int r = idx / CO_BK;
      const int d = idx % CO_BK;
      sB[r][d] = (d < actual_BK) ? h_ch[(size_t)r * K + k_base + d]
                                 : __float2half(0.0f);
    }
    __syncthreads();
    const int n_k_tiles = (actual_BK + 15) / 16;
    for (int kt = 0; kt < n_k_tiles; kt++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fq;
      wmma::load_matrix_sync(fq, &sQ[m_base][kt * 16], CO_SQK);
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
        wmma::load_matrix_sync(fb, &sB[nt * 16][kt * 16], CO_SQK);
        wmma::mma_sync(acc_o[nt], fq, fb, acc_o[nt]);
      }
    }
    __syncthreads();
  }

  // Store ungated A as fp32 over sB (h tiles are dead; 64 rows x 66 float
  // = 16896 B matches sB exactly).
  float (*sBf)[66] = reinterpret_cast<float (*)[66]>(sB);
#pragma unroll
  for (int nt = 0; nt < 4; nt++)
    wmma::store_matrix_sync(&sBf[m_base][nt * 16], acc_A[nt], 66,
                            wmma::mem_row_major);
  __syncthreads();

  // Pairwise gating + lower-triangular mask -> fp16 sA.
  for (int idx = threadIdx.x; idx < CO_BT * CO_BT; idx += 128) {
    const int m = idx / CO_BT;
    const int n = idx % CO_BT;
    const int gm = i_tc * CO_BT + m;
    const int gn = i_tc * CO_BT + n;
    float val = 0.0f;
    if (gm < T && gn < T && m >= n)
      val = sBf[m][n] * __expf(sG[m] - sG[n]);
    sA[m][n] = __float2half(val);
  }
  __syncthreads();

  // Load v tile into sB (safe: A float view fully consumed).
  for (int idx = threadIdx.x; idx < CO_BT * CO_BV; idx += 128) {
    const int t = idx / CO_BV;
    const int d = idx % CO_BV;
    const int gt = i_tc * CO_BT + t;
    sB[t][d] = (gt < T) ? v_head[(size_t)gt * HV + d] : __float2half(0.0f);
  }
  __syncthreads();

  // acc_A is dead; reuse for A @ v: o[m, d] = sum_n A[m, n] * v[n, d].
  // Each of the 4 accumulators holds one d-tile (16 cols); the inner loop
  // sums the 4 n-tiles of A/v.
#pragma unroll
  for (int i = 0; i < 4; i++) wmma::fill_fragment(acc_A[i], 0.0f);
  for (int nt = 0; nt < CO_BT / 16; nt++) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::load_matrix_sync(fa, &sA[m_base][nt * 16], CO_BT);
#pragma unroll
    for (int dt = 0; dt < 4; dt++) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fv;
      wmma::load_matrix_sync(fv, &sB[nt * 16][dt * 16], CO_SQK);
      wmma::mma_sync(acc_A[dt], fa, fv, acc_A[dt]);
    }
  }
  // Combine both terms (same fragment shape: elementwise add) and store
  // the fp32 result over sQ (dead after pass 2).
  float (*sQf)[64] = reinterpret_cast<float (*)[64]>(sQ);
#pragma unroll
  for (int vt = 0; vt < 4; vt++) {
#pragma unroll
    for (int e = 0; e < acc_o[vt].num_elements; e++)
      acc_o[vt].x[e] += acc_A[vt].x[e];
    wmma::store_matrix_sync(&sQf[m_base][vt * 16], acc_o[vt], 64,
                            wmma::mem_row_major);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < CO_BT * CO_BV; idx += 128) {
    const int m = idx / CO_BV;
    const int d = idx % CO_BV;
    const int gm = i_tc * CO_BT + m;
    if (gm >= T) continue;
    o_head[(size_t)gm * HV + d] = __float2half(sQf[m][d] * scale);
  }
}

}  // namespace sm70_fla
}  // namespace vllm

// chunk_o launcher.  scale is passed as the bit pattern of a float inside
// an int64 (stable-libtorch TORCH_BOX mangles float scalar args).
torch::stable::Tensor fla_chunk_o_sm70(
    torch::stable::Tensor q,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor h,
    torch::stable::Tensor g,
    torch::stable::Tensor o,
    torch::stable::Tensor cu_seqlens,
    torch::stable::Tensor chunk_indices,
    int64_t scale_bits,
    int64_t NT_total) {
  const int T_total = q.size(0);
  const int Hg = q.size(1);
  const int K = q.size(2);
  const int H = v.size(1);
  const int V = v.size(2);
  const int NT = (int)NT_total;
  float scale;
  {
    uint32_t bits = (uint32_t)(scale_bits & 0xffffffffu);
    memcpy(&scale, &bits, sizeof(scale));
  }

  const torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  dim3 grid(NT, H, V / vllm::sm70_fla::CO_BV);
  vllm::sm70_fla::fla_chunk_o_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<const half*>(
          q.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(
          k.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(
          v.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(
          h.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(
          g.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<half*>(o.mutable_data_ptr<torch::headeronly::Half>()),
      cu_seqlens.mutable_data_ptr<int>(),
      chunk_indices.mutable_data_ptr<int>(),
      H, Hg, K, V, scale);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("fla_chunk_o_sm70 launch failed: %s grid=(%d,%d,%d)\n",
           cudaGetErrorString(err), (int)grid.x, (int)grid.y, (int)grid.z);
    assert(false);
  }
  (void)T_total;
  return o;
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
    float* __restrict__ h_out,       // [NT_total, H, V, K] (fp32, matches Triton)
    float* __restrict__ ht_out,      // [N, H, V, K] final state (per seq), may be null
    const float* __restrict__ init_state,  // [N, H, V, K] pre-chunk states, may be null
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
  // Number of 64-wide K sub-blocks; K must be a multiple of DH_KS (the
  // Python dispatch gates this kernel to K <= 256).
  const int NK = K / DH_KS;

  const half* k_head = k + (bos * Hg + hg) * K;
  const half* w_head = w + (bos * H + i_h) * K;
  const half* u_head = u + (bos * H + i_h) * V;
  const half* g_head = g + bos * H + i_h;
  half* vn_head = v_new + (bos * H + i_h) * V;
  float* h_head = h_out + ((boh * H + i_h) * V) * K;

  const int warp_id = threadIdx.x / 32;

  // Dynamic shared layout: fp32 state [4 subs][DH_BV][DH_KS] then a
  // half staging tile [DH_BV][DH_KS] for the WMMA loads. The state stays
  // fp32 across chunks (matching the Triton reference); fp16 is used only
  // for the WMMA dot operands.
  extern __shared__ float sH_state[];
  float* sHsub[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) sHsub[i] = sH_state + i * (DH_BV * DH_KS);
  half* sHh = reinterpret_cast<half*>(sH_state + 4 * DH_BV * DH_KS);

  __shared__ half sData[DH_BT][DH_SData];
  __shared__ float sV[DH_BT][DH_BV];
  __shared__ half sVh[DH_BT][DH_BV];
  __shared__ float sG[DH_BT];

  // Initial state: the sequence's pre-chunk state (decode-carried for
  // continuations, zeros for fresh sequences).
  if (init_state != nullptr) {
    const float* init_head = init_state + (int64_t)i_nh * V * K +
        i_v * DH_BV * K;
    for (int sub = 0; sub < NK; sub++) {
        for (int idx = threadIdx.x; idx < DH_BV * DH_KS; idx += 128) {
          int r = idx / DH_KS, c = idx % DH_KS;
          sHsub[sub][r * DH_KS + c] = init_head[r * K + sub * DH_KS + c];
        }
      }
    } else {
      for (int sub = 0; sub < NK; sub++) {
        for (int idx = threadIdx.x; idx < DH_BV * DH_KS; idx += 128) {
          int r = idx / DH_KS, c = idx % DH_KS;
          sHsub[sub][r * DH_KS + c] = 0.0f;
        }
      }
    }
  __syncthreads();

  // V computation: each warp handles 1 M-tile (16 rows of BT=64), 2 N-tiles (BV=32)
  const int m_tile_v = warp_id;

  for (int i_t = 0; i_t < NT; i_t++) {
    int cs = i_t * DH_BT;
    int ce = min(cs + DH_BT, T);

    // Store current state to h_out (fp32, chunk-start state)
    for (int sub = 0; sub < NK; sub++) {
      for (int idx = threadIdx.x; idx < DH_BV * DH_KS; idx += 128) {
        int r = idx / DH_KS, c = idx % DH_KS;
        h_head[i_t * HVK + (i_v * DH_BV + r) * K + sub * DH_KS + c] =
            sHsub[sub][r * DH_KS + c];
      }
    }
    __syncthreads();

    // Compute v = sum_i(w_i @ h_i^T)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> v_acc[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) wmma::fill_fragment(v_acc[i], 0.0f);

    for (int sub = 0; sub < NK; sub++) {
      // Load w[64, 64] into sData
      for (int idx = threadIdx.x; idx < DH_BT * DH_KS; idx += 128) {
        int t = idx / DH_KS, d = idx % DH_KS;
        int gt = cs + t;
        sData[t][d] = (gt < T) ? w_head[gt * HK + sub * DH_KS + d]
                              : __float2half(0.0f);
      }
      // Stage the state sub-block in fp16 for the WMMA operand
      for (int idx = threadIdx.x; idx < DH_BV * DH_KS; idx += 128) {
        int r = idx / DH_KS, c = idx % DH_KS;
        sHh[r * DH_KS + c] = __float2half(sHsub[sub][r * DH_KS + c]);
      }
      __syncthreads();

      // v += w @ h^T  (w[64,64] @ h^T[64,32] → v[64,32])
      // A=w row_major, B=h^T col_major from sHh[sub]
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
                                 &sHh[nt * 16 * DH_KS + kt * 16],
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

    // Gate: v *= exp(g_last - g), h *= exp(g_last). g is a natural-log
    // cumulative sum (chunk.py -> chunk_local_cumsum, USE_EXP2=False).
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

    for (int sub = 0; sub < NK; sub++) {
      for (int idx = threadIdx.x; idx < DH_BV * DH_KS; idx += 128) {
        int r = idx / DH_KS, c = idx % DH_KS;
        sHsub[sub][r * DH_KS + c] *= g_last_exp;
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

    for (int sub = 0; sub < NK; sub++) {
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
          // B = k[16, 16] from sData, row_major, chunk rows aligned with
          // A's chunk columns: k[kt*16 + p][n*16 + c].
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                         wmma::row_major> b_frag;
          wmma::load_matrix_sync(b_frag,
                                 &sData[kt * 16][n_tile * 16],
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
        sHsub[sub][r * DH_KS + c] += sV_flat[r * DH_KS + c];
      }
      __syncthreads();
    }
  }

  // Epilogue: h_out stores the state at each chunk START; the post-chunk
  // state of the final chunk is the sequence's final state.
  if (ht_out != nullptr) {
    float* ht_head = ht_out + i_nh * V * K + i_v * DH_BV * K;
    for (int sub = 0; sub < NK; sub++) {
      for (int idx = threadIdx.x; idx < DH_BV * DH_KS; idx += 128) {
        int r = idx / DH_KS, c = idx % DH_KS;
        ht_head[r * K + sub * DH_KS + c] =
            sHsub[sub][r * DH_KS + c];
      }
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

  auto h = torch::stable::empty({B, NT, H, V, K},
                                torch::stable::ScalarType::Float,
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

  // Dynamic shared: fp32 state (4*32*64) + fp16 staging tile (32*64).
  constexpr int kDynSmem = 4 * vllm::sm70_fla::DH_BV *
                               vllm::sm70_fla::DH_KS * sizeof(float) +
                           vllm::sm70_fla::DH_BV *
                               vllm::sm70_fla::DH_KS * sizeof(half);
  static bool smem_configured = false;
  if (!smem_configured) {
    cudaFuncSetAttribute(vllm::sm70_fla::fla_delta_h_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kDynSmem);
    smem_configured = true;
  }

  vllm::sm70_fla::fla_delta_h_kernel<<<grid, 128, kDynSmem, stream>>>(
      reinterpret_cast<const half*>(k.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(w.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(u.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(g.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<half*>(v_new.mutable_data_ptr<torch::headeronly::Half>()),
      h.mutable_data_ptr<float>(),
      output_final_state ? final_state.mutable_data_ptr<float>() : nullptr,
      initial_state.size(0) > 0 ? initial_state.mutable_data_ptr<float>() : nullptr,
      cu_seqlens.mutable_data_ptr<int>(),
      chunk_offsets.mutable_data_ptr<int>(),
      H, Hg, K, V);

  return {h, v_new, final_state};
}

// Fused single-chunk gated-delta-rule verify kernel (T <= FV_MAX_T).
// Collapses the 5-launch chain cumsum/kkt/solve_tril/wy/delta_h/chunk_o
// into one launch for the MTP-verify path (small T, one chunk per seq).
// Math (single chunk, S = (I + strict_tril(A_k))^-1, A_k[t,s] =
// k[t].k[s]*exp(g_cs[t]-g_cs[s])*beta[t]):
//   v_new[t]  = sum_s S[t,s] * (v[s] - k[s]^T h_prev)
//   h_new     = h_prev * exp(g_last) + sum_t exp(g_last-g_cs[t]) (v_new[t] x k[t])
//   o[t]      = scale * ( exp(g_cs[t]) (q[t]^T h_prev)
//               + sum_{s<=t} exp(g_cs[t]-g_cs[s]) (q[t].k[s]) v_new[s] )
// Grid: (cdiv(V,32), N*H), Block: 128.
constexpr int FV_MAX_T = 8;

__global__ __launch_bounds__(128, 1)
void fla_verify_kernel(
    const half* __restrict__ q,        // [T_total, Hg, K] (H/Hg heads share one q head)
    const half* __restrict__ k,        // [T_total, Hg, K]
    const half* __restrict__ v,        // [T_total, H, V]
    const half* __restrict__ beta,     // [T_total, H]
    const half* __restrict__ g_raw,    // [T_total, H] (pre-cumsum, natural log)
    const float* __restrict__ h_prev,  // [N, H, V, K], may be null (zeros)
    half* __restrict__ o,              // [T_total, H, V]
    float* __restrict__ h_new,         // [N, H, V, K]
    const int* __restrict__ cu_seqlens,
    int64_t scale_bits, int H, int Hg, int K, int V) {
  const float scale = __int_as_float((int32_t)scale_bits);
  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_h = i_nh % H;
  const int i_n = i_nh / H;

  int bos = cu_seqlens[i_n];
  int T = cu_seqlens[i_n + 1] - bos;
  if (T <= 0 || T > FV_MAX_T) return;

  const int hg = i_h / (H / Hg);

  const half* q_head = q + (bos * Hg + hg) * K;
  const half* k_head = k + (bos * Hg + hg) * K;
  const half* v_head = v + (bos * H + i_h) * V + i_v * 32;
  const half* b_head = beta + bos * H + i_h;
  const half* g_head = g_raw + bos * H + i_h;
  const float* h_head = h_prev ? h_prev + i_nh * V * K + i_v * 32 * K : nullptr;
  half* o_head = o + (bos * H + i_h) * V + i_v * 32;
  float* hw_head = h_new + i_nh * V * K + i_v * 32 * K;

  __shared__ half sK[FV_MAX_T][128];
  __shared__ half sQ[FV_MAX_T][128];
  __shared__ half sV[FV_MAX_T][32];
  __shared__ float sG[FV_MAX_T];   // cumulative g
  __shared__ float sBeta[FV_MAX_T];
  __shared__ float sA[FV_MAX_T][FV_MAX_T];   // gated kkt
  __shared__ float sQK[FV_MAX_T][FV_MAX_T];  // q[t].k[s]
  __shared__ float sS[FV_MAX_T][FV_MAX_T];   // (I + tril(A))^-1
  __shared__ float sKh[FV_MAX_T][32];  // k[t]^T h_prev
  __shared__ float sQh[FV_MAX_T][32];  // q[t]^T h_prev
  __shared__ float sVn[FV_MAX_T][32];  // v_new
  __shared__ float sGexp[FV_MAX_T];   // exp(g_cs[t])
  const int tid = threadIdx.x;

  // Load q/k/v/beta/g into shared
  for (int idx = tid; idx < FV_MAX_T * 128; idx += 128) {
    int t = idx >> 7, d = idx & 127;
    sK[t][d] = (t < T) ? __ldg(&k_head[t * (Hg * K) + d]) : __float2half(0.0f);
    sQ[t][d] = (t < T) ? __ldg(&q_head[t * (Hg * K) + d]) : __float2half(0.0f);
  }
  for (int idx = tid; idx < FV_MAX_T * 32; idx += 128) {
    int t = idx >> 5, d = idx & 31;
    sV[t][d] = (t < T) ? __ldg(&v_head[t * (H * V) + d]) : __float2half(0.0f);
  }
  for (int idx = tid; idx < FV_MAX_T; idx += 128) {
    if (idx < T) {
      sBeta[idx] = __half2float(__ldg(&b_head[idx * H]));
      sG[idx] = __half2float(__ldg(&g_head[idx * H]));
    } else {
      sBeta[idx] = 0.0f;
      sG[idx] = 0.0f;
    }
  }
  __syncthreads();
  // cumulative sum of raw g (per head; T small -> serial on thread 0)
  if (tid == 0) {
    for (int t = 1; t < T; t++) sG[t] += sG[t - 1];
    for (int t = 0; t < T; t++) sGexp[t] = expf(sG[t]);
  }
  __syncthreads();

  // kkt + qk dots (one thread per (t,s) pair)
  for (int idx = tid; idx < FV_MAX_T * FV_MAX_T; idx += 128) {
    int t = idx / FV_MAX_T, s = idx % FV_MAX_T;
    if (t < T && s <= t) {
      float dot = 0.0f;
      for (int d = 0; d < 128; d++) dot += __half2float(sK[t][d]) * __half2float(sK[s][d]);
      sA[t][s] = dot * expf(sG[t] - sG[s]) * sBeta[t];
      float qk = 0.0f;
      for (int d = 0; d < 128; d++) qk += __half2float(sQ[t][d]) * __half2float(sK[s][d]);
      sQK[t][s] = qk;
    } else if (t < T && s < T) {
      sA[t][s] = 0.0f;
      sQK[t][s] = 0.0f;
    }
  }
  __syncthreads();
  // S = (I + strict_tril(A))^-1 via forward substitution (thread 0, T <= 8)
  if (tid == 0) {
    for (int t = 0; t < T; t++) {
      for (int s = 0; s < T; s++) sS[t][s] = (s == t) ? 1.0f : 0.0f;
    }
    for (int t = 1; t < T; t++) {
      for (int s = 0; s < t; s++) {
        float acc = 0.0f;
        for (int u = s; u < t; u++) acc += sA[t][u] * sS[u][s];
        sS[t][s] = -acc;
      }
    }
  }
  __syncthreads();

  // state dots qh/kh: thread (t, vd)
  for (int idx = tid; idx < FV_MAX_T * 32; idx += 128) {
    int t = idx >> 5, vd = idx & 31;
    if (t < T) {
      float ak = 0.0f, aq = 0.0f;
      for (int d = 0; d < 128; d++) {
        float kd = __half2float(sK[t][d]);
        float qd = __half2float(sQ[t][d]);
        float hp = h_head ? __ldg(&h_head[vd * K + d]) : 0.0f;
        ak += kd * hp;
        aq += qd * hp;
      }
      sKh[t][vd] = ak;
      sQh[t][vd] = aq;
    }
  }
  __syncthreads();

  // v_new + h_new + o
  // v_new[t] = u[t] - w[t]^T h_prev with the production wy semantics:
  //   u[t] = sum_s S[t,s]*beta[s]*v[s]
  //   w[t] = exp(g_cs[t]) * sum_s S[t,s]*beta[s]*k[s]
  // so v_new[t,vd] = sum_s S[t,s]*beta[s]*(v[s,vd] - exp(g_cs[t])*kh[s,vd])
  for (int idx = tid; idx < FV_MAX_T * 32; idx += 128) {
    int t = idx >> 5, vd = idx & 31;
    if (t < T) {
      float et = sGexp[t];
      float vn = 0.0f;
      for (int s = 0; s < T; s++)
        vn += sS[t][s] * sBeta[s] * (__half2float(sV[s][vd]) - et * sKh[s][vd]);
      sVn[t][vd] = vn;
    }
  }
  __syncthreads();

  for (int idx = tid; idx < FV_MAX_T * 32 * 128; idx += 128) {
    int vd = (idx / 128) & 31;
    int t = (idx / (32 * 128));
    int d = idx & 127;
    if (t < T) {
      float hp = h_head ? __ldg(&h_head[vd * K + d]) : 0.0f;
      float hacc = hp * sGexp[T - 1];
      for (int tt = 0; tt < T; tt++)
        hacc += (sGexp[T - 1] / sGexp[tt]) * sVn[tt][vd] * __half2float(sK[tt][d]);
      hw_head[vd * K + d] = hacc;
    }
  }
  __syncthreads();

  for (int idx = tid; idx < FV_MAX_T * 32; idx += 128) {
    int t = idx >> 5, vd = idx & 31;
    if (t < T) {
      float oacc = sGexp[t] * sQh[t][vd];
      for (int s = 0; s <= t; s++)
        oacc += (sGexp[t] / sGexp[s]) * sQK[t][s] * sVn[s][vd];
      o_head[t * (H * V) + vd] = __float2half(oacc * scale);
    }
  }

}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
fla_verify_sm70(
    torch::stable::Tensor q, torch::stable::Tensor k,
    torch::stable::Tensor v, torch::stable::Tensor beta,
    torch::stable::Tensor g, torch::stable::Tensor h_prev,
    torch::stable::Tensor cu_seqlens, int64_t scale_bits,
    int64_t N) {
  int T = k.size(1);
  int H = v.size(2), Hg = k.size(2), K = k.size(3), V = v.size(3);
  const torch::stable::accelerator::DeviceGuard device_guard(
      k.get_device_index());
  auto o = torch::stable::empty({1, T, H, V}, v.scalar_type(), std::nullopt,
                                v.device());
  auto h_new = torch::stable::empty({N, H, V, K}, torch::stable::ScalarType::Float,
                                    std::nullopt, k.device());
  dim3 grid((V + 31) / 32, N * H);
  const cudaStream_t stream = get_current_cuda_stream();
  fla_verify_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<const half*>(q.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(k.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(v.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(beta.mutable_data_ptr<torch::headeronly::Half>()),
      reinterpret_cast<const half*>(g.mutable_data_ptr<torch::headeronly::Half>()),
      h_prev.size(0) > 0 ? h_prev.mutable_data_ptr<float>() : nullptr,
      reinterpret_cast<half*>(o.mutable_data_ptr<torch::headeronly::Half>()),
      h_new.mutable_data_ptr<float>(),
      cu_seqlens.mutable_data_ptr<int>(),
      scale_bits, H, Hg, K, V);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("fla_verify_sm70 launch failed: %s grid=(%d,%d) block=128 smem=0\n",
           cudaGetErrorString(err), (int)grid.x, (int)grid.y);
    assert(false);
  }
  return {o, h_new};
}
