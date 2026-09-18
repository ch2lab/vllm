// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// SM70 (V100) unified NVFP4 paged decode/verify — FlashDecoding split-KV
// kernel.
//
// Structure (cudagraph-safe, fixed grid):
//   partition kernel  grid = (splits, H, R), block = 128 threads (4 warps)
//     one (kv head, KV split) per block; the block owns the full qo-head
//     group of its kv head, so every KV byte is read exactly once per kv
//     head (no 12x qo-head re-reads). q rows (<= 8) are processed
//     sequentially over the same staged KV; row qq sits at sequence
//     position kv_len - q_rows + qq and attends causally to
//     [lo, kv_len - q_rows + qq + 1).
//     In-kernel NVFP4 dequant (direct 16B vectorized loads + bit ops +
//     LUT — no gather-based indexing, the sm_70 Triton miscompilation that
//     caps the legacy decode at ~1.8 GB/s) and synchronous 16B page
//     staging (cp.async needs sm_80+; not available on sm_70).
//     Emits per-(q row, qo head) partial online-softmax states
//     (acc | m | l) to a persistent fp32 workspace.
//   reduce kernel     grid = (H, R), block = 128 threads
//     log-sum-exp merge across splits into the final fp16 output.
//
// Layout (LBHNC raw bytes, identical to sm70_nvfp4_decode /
// sm70_nvfp4_dequant): one page (stride page) = K side | V side; each side
// = data region (H*N*DD) then scale region (H*N*SD); token row =
// [DD data bytes | SD scale bytes] with F = DD + SD. Nibble: even dim =
// low nibble of byte d//2, odd = high. One e4m3 scale byte per 16 dims.
// Dequant: x = E2M1[nib] * E4M3[sf] * layer_scale.

#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kHQ = 12;        // qo heads (per GPU, TP2 of 24)
constexpr int kH = 2;          // kv heads (per GPU, TP2 of 4)
constexpr int kRatio = 6;      // qo heads per kv head (kHQ / kH)
constexpr int kD = 256;        // head dim
constexpr int kN = 16;         // tokens per staging chunk (compile-time)
constexpr int kDD = kD / 2;    // 128 data bytes per token row
constexpr int kSD = kD / 16;   // 16 scale bytes per token row
constexpr int kF = kDD + kSD;  // 144 bytes per token row
constexpr int kMaxQ = 8;       // max rows per request
constexpr int kThreads = 128;
constexpr int kWarps = 4;
// Workspace slot stride: one (qq, i) row of partial state.
constexpr int kRowStride = kD + 2;

// E2M1 nibble -> half bits (index = nibble code 0..15, bit 3 = sign).
// Magnitudes: {0, .5, 1, 1.5, 2, 3, 4, 6}.
__constant__ unsigned c_e2m1_bits[16] = {
    0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600,
    0x8000, 0xB800, 0xBC00, 0xBE00, 0xC000, 0xC200, 0xC400, 0xC600,
};

__device__ __forceinline__ __half e2m1_half(unsigned code) {
  return *reinterpret_cast<const __half*>(&c_e2m1_bits[code & 0xF]);
}

// e4m3 scale byte -> float. Matches the Triton LUT semantics (e = 0 is the
// subnormal bin; the store kernel never emits e = 15).
__device__ __forceinline__ float e4m3_to_float(unsigned char b) {
  const bool sgn = b & 0x80;
  const int e = (b >> 3) & 0xF;
  const int m = b & 7;
  float v;
  if (e == 0) {
    v = m * 0.001953125f;  // 2^-9
  } else {
    v = (1.0f + m * 0.125f) * exp2f((float)e - 7.0f);
  }
  return sgn ? -v : v;
}

__global__ void __launch_bounds__(kThreads)
sm70_flashdecode_partition_kernel(
    const __half* __restrict__ q,          // [R, q_rows, kHQ, kD]
    const uint8_t* __restrict__ kv,        // raw LBHNC bytes
    const int* __restrict__ bt,            // [R, bt_cols]
    const int* __restrict__ kv_lens,       // [R]
    float* __restrict__ ws,                // [R, kH, splits, q_rows, kRatio, kD+2]
    float k_scale, float v_scale, float sm_scale,
    int window_left, int splits, int q_rows, int bt_cols, int n_page,
    int64_t page_stride, int64_t side_off) {
  const int r = blockIdx.z;
  const int h = blockIdx.y;
  const int sp = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  __shared__ __half q_smem[kRatio * kD];       // 3 KB (one q row)
  __shared__ uint8_t kv_raw[2][kN * kF];       // 4608 B (side, row*F)
  __shared__ float sf_smem[2][kN * kSD];       // scale bytes, layer-scaled
  __shared__ float score_sh[kWarps][kRatio * kN];  // warp partials [w][i][t]
  __shared__ float score_full[kRatio * kN];   // merged [i][t]
  __shared__ float p_sh[kRatio * kN];         // exp(s - m_new) [i][t]
  // [0:kR) page_m, [kR:2kR) page_l, [2kR:3kR) alpha, [3kR:4kR) w_o,
  // [4kR:5kR) w_p, [5kR:6kR) m_merged, [6kR:7kR) l_merged (per i).
  __shared__ float stat_sh[7 * kRatio];

  // Dim mapping: lane owns dims (warp*64 + 2*lane, +1); byte = warp*32+lane
  // within the 128B data row; scale groups: dim d -> group d//16, so
  // d0 = warp*64 + 2*lane -> warp*4 + lane//8.
  const int d0 = warp * 64 + 2 * lane;
  const int d1 = d0 + 1;
  const int d_byte = warp * 32 + lane;
  const int g_lo = warp * 4 + (lane >> 3);
  const int g_hi = warp * 4 + (2 * lane + 1) / 16;

  const int kv_len = __ldg(kv_lens + r);

  // Per-(r,h) workspace block stride (q_rows-dependent: only the active
  // rows are allocated, so q=1 decode needs a small persistent workspace).
  const int64_t ws_sp = (int64_t)q_rows * kRatio * kRowStride;

  const int64_t ws_base =
      ((int64_t)(r * kH + h) * splits + sp) * ws_sp;

  auto write_empty_row = [&](int qq) {
    float* row = ws + ws_base + qq * (kRatio * kRowStride);
    for (int idx = tid; idx < kRatio * kRowStride; idx += kThreads) {
      const int rem = idx % kRowStride;
      row[idx] = (rem == kD) ? -INFINITY
               : (rem == kD + 1) ? 0.0f
                                 : (rem < kD ? 0.0f : 0.0f);
    }
  };

  if (kv_len == 0) {
    for (int qq = 0; qq < q_rows; qq++) write_empty_row(qq);
    return;
  }
  int lo = window_left >= 0 ? kv_len - 1 - window_left : 0;
  if (lo < 0) lo = 0;
  if (lo >= kv_len) {
    for (int qq = 0; qq < q_rows; qq++) write_empty_row(qq);
    return;
  }

  // Token-level split range for this partition: [t_a, t_b). The chunk grid
  // is 16-token units; n_page is a multiple of 16, so a chunk never
  // straddles a block boundary.
  const int64_t t_a = (int64_t)lo + ((int64_t)kv_len - lo) * sp / splits;
  const int64_t t_b =
      (int64_t)lo + ((int64_t)kv_len - lo) * (sp + 1) / splits;
  if (t_a >= t_b) {
    for (int qq = 0; qq < q_rows; qq++) write_empty_row(qq);
    return;
  }
  const int c0 = (int)(t_a / kN);
  const int c1 = (int)((t_b - 1) / kN);

  // Stage one 16-token chunk of a block (both sides): 288 x 16B,
  // synchronous. The chunk starts at token n0 within block b (n0 a
  // multiple of kN; the block holds n_page tokens).
  auto stage_chunk = [&](int b, int n0) {
    const int64_t base =
        (int64_t)__ldg(bt + r * bt_cols + b) * page_stride;
    constexpr int kChunks = 2 * kN * (kF / 16);  // 288
    for (int c = tid; c < kChunks; c += kThreads) {
      const int side = c / (kN * kF / 16);  // 0 = K, 1 = V
      const int rem = c % (kN * kF / 16);
      const int n = rem / (kF / 16);
      const int f16 = rem % (kF / 16);  // 16B chunk in the 144B row
      int64_t off;
      if (f16 < kDD / 16) {  // data chunk
        off = (int64_t)h * (n_page * kDD) + (n0 + n) * kDD + f16 * 16;
      } else {  // scale chunk (exactly one 16B chunk)
        off = (int64_t)kH * (n_page * kDD) + (int64_t)h * (n_page * kSD) +
              (n0 + n) * kSD + (f16 - kDD / 16) * 16;
      }
      const uint4 v = *reinterpret_cast<const uint4*>(
          kv + base + side * side_off + off);
      *reinterpret_cast<uint4*>(kv_raw[side] + n * kF + f16 * 16) = v;
    }
    __syncthreads();
  };

  // Decode all scale bytes of the staged page (layer scale pre-multiplied).
  auto decode_scales = [&]() {
    constexpr int kScales = 2 * kN * kSD;  // 512
    for (int idx = tid; idx < kScales; idx += kThreads) {
      const int side = idx / (kN * kSD);
      const int rem = idx % (kN * kSD);
      const int n = rem / kSD;
      const int g = rem % kSD;
      sf_smem[side][n * kSD + g] =
          e4m3_to_float(kv_raw[side][n * kF + kDD + g]) *
          (side == 0 ? k_scale : v_scale);
    }
    __syncthreads();
  };

  // Load one q row (sm_scale folded in; half2 loads). q is [R, q_rows, kHQ, kD].
  auto load_q_row = [&](int qq) {
    const __half* src =
        q + ((int64_t)(r * q_rows + qq) * kHQ + h * kRatio) * kD;
    for (int idx = tid; idx < kRatio * kD / 2; idx += kThreads) {
      const int t = idx / (kD / 2);
      const int d2 = (idx % (kD / 2)) * 2;
      const __half2 v = *reinterpret_cast<const __half2*>(src + t * kD + d2);
      const __half2 s = __float2half2_rn(sm_scale);
      reinterpret_cast<__half2*>(&q_smem[0])[idx] = __hmul2(v, s);
    }
  };

  // Chunk loop (synchronous staging — cp.async needs sm_80+). This block
  // exclusively owns its (sp, qq) ws slots: start them empty, then merge
  // each chunk's partial in (online combine), so q_rows rows share one
  // pass over the staged KV.
  for (int qq = 0; qq < q_rows; qq++) {
    write_empty_row(qq);
  }
  __syncthreads();
  for (int c = c0; c <= c1; c++) {
    const int cbase = c * kN;
    const int block = cbase / n_page;
    stage_chunk(block, cbase - block * n_page);
    decode_scales();

    const int t0 = cbase > (int)t_a ? cbase : (int)t_a;
    const int t1_page = cbase + kN < (int)t_b ? cbase + kN : (int)t_b;

    for (int qq = 0; qq < q_rows; qq++) {
      const int hi = kv_len - q_rows + qq + 1;
      const int t1 = t1_page < hi ? t1_page : hi;

      // Per-row partial state (registers: kRatio acc pairs + scores).
      float acc_lo[kRatio];
      float acc_hi[kRatio];
#pragma unroll
      for (int i = 0; i < kRatio; i++) {
        acc_lo[i] = 0.0f;
        acc_hi[i] = 0.0f;
      }
      float m = -INFINITY;
      float l = 0.0f;
      if (tid < kRatio) {
        stat_sh[tid] = -INFINITY;
        stat_sh[kRatio + tid] = 0.0f;
      }

      if (t0 < t1) {
        load_q_row(qq);
        __syncthreads();

        const __half* qsm = &q_smem[0];
        const float* sfk = &sf_smem[0][0];
        const float* sfv = &sf_smem[1][0];

        // Phase 1: per-token scores (warp partials, reduced per token).
        for (int t = t0; t < t1; t++) {
          const int n = t - cbase;
          const unsigned char kb = kv_raw[0][n * kF + d_byte];
          const float K_lo =
              __half2float(e2m1_half(kb & 0xF)) * sfk[n * kSD + g_lo];
          const float K_hi =
              __half2float(e2m1_half(kb >> 4)) * sfk[n * kSD + g_hi];
          float sacc[kRatio];
#pragma unroll
          for (int i = 0; i < kRatio; i++) {
            const int ioff = i * kD;
            sacc[i] = __half2float(qsm[ioff + d0]) * K_lo +
                      __half2float(qsm[ioff + d1]) * K_hi;
          }
          const int trel = t - t0;
#pragma unroll
          for (int i = 0; i < kRatio; i++) {
            float v = sacc[i];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
              v += __shfl_xor_sync(0xFFFFFFFF, v, o);
            }
            if (lane == 0) {
              score_sh[warp][i * kN + trel] = v;
            }
          }
          __syncwarp();
        }
        __syncthreads();

        // Merge warp partials: 192 values over 128 threads.
        for (int v = tid; v < kRatio * kN; v += kThreads) {
          float s = 0.0f;
#pragma unroll
          for (int w = 0; w < kWarps; w++) {
            s += score_sh[w][v];
          }
          score_full[v] = s;
        }
        __syncthreads();

        // Per-head online softmax update.
        if (tid < kRatio) {
          const float* srow = &score_full[tid * kN];
          float m_tile = -INFINITY;
#pragma unroll
          for (int t = 0; t < kN; t++) {
            if (t0 + t < t1) {
              if (srow[t] > m_tile) m_tile = srow[t];
            }
          }
          const float m_new = fmaxf(m, m_tile);
          const float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
          float l_new = l * alpha;
          for (int t = 0; t < kN; t++) {
            if (t0 + t >= t1) {
              p_sh[tid * kN + t] = 0.0f;
              continue;
            }
            const float p = expf(srow[t] - m_new);
            p_sh[tid * kN + t] = p;
            l_new += p;
          }
          stat_sh[tid] = m_new;
          stat_sh[kRatio + tid] = l_new;
          stat_sh[2 * kRatio + tid] = alpha;
          m = m_new;
          l = l_new;
        }
        __syncthreads();

        // Rescale accumulators, then PV.
        const float* alpha_sh = &stat_sh[2 * kRatio];
        const float* psm = &p_sh[0];
#pragma unroll
        for (int i = 0; i < kRatio; i++) {
          acc_lo[i] *= alpha_sh[i];
          acc_hi[i] *= alpha_sh[i];
        }
        __syncthreads();
        for (int t = t0; t < t1; t++) {
          const int n = t - cbase;
          const int trel = t - t0;
          const unsigned char vb = kv_raw[1][n * kF + d_byte];
          const float V_lo =
              __half2float(e2m1_half(vb & 0xF)) * sfv[n * kSD + g_lo];
          const float V_hi =
              __half2float(e2m1_half(vb >> 4)) * sfv[n * kSD + g_hi];
#pragma unroll
          for (int i = 0; i < kRatio; i++) {
            const float p = psm[i * kN + trel];
            acc_lo[i] += p * V_lo;
            acc_hi[i] += p * V_hi;
          }
        }
      }

      // Merge this page's partial into the running ws slot for row qq
      // (log-sum-exp combine of the existing slot and the page partial).
      {
        float* row = ws + ws_base + qq * (kRatio * kRowStride);
        if (tid < kRatio) {
          const float m_ws = row[tid * kRowStride + kD];
          const float l_ws = row[tid * kRowStride + kD + 1];
          const float m_pg = stat_sh[tid];
          const float l_pg = stat_sh[kRatio + tid];
          const float M = fmaxf(m_ws, m_pg);
          float w_o, w_p, l_m;
          if (M == -INFINITY) {
            w_o = 0.0f;
            w_p = 0.0f;
            l_m = 0.0f;
          } else {
            w_o = (m_ws == -INFINITY) ? 0.0f : expf(m_ws - M);
            w_p = (m_pg == -INFINITY) ? 0.0f : expf(m_pg - M);
            l_m = w_o * l_ws + w_p * l_pg;
          }
          stat_sh[3 * kRatio + tid] = w_o;
          stat_sh[4 * kRatio + tid] = w_p;
          stat_sh[5 * kRatio + tid] = M;
          stat_sh[6 * kRatio + tid] = l_m;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kRatio; i++) {
          const float w_o = stat_sh[3 * kRatio + i];
          const float w_p = stat_sh[4 * kRatio + i];
          row[i * kRowStride + d0] =
              w_o * row[i * kRowStride + d0] + w_p * acc_lo[i];
          row[i * kRowStride + d1] =
              w_o * row[i * kRowStride + d1] + w_p * acc_hi[i];
        }
        if (tid < kRatio) {
          row[tid * kRowStride + kD] = stat_sh[5 * kRatio + tid];
          row[tid * kRowStride + kD + 1] = stat_sh[6 * kRatio + tid];
        }
      }
      __syncthreads();  // q_smem reuse
    }
  }
}

__global__ void __launch_bounds__(kThreads)
sm70_flashdecode_reduce_kernel(
    __half* __restrict__ out,        // [R, q_rows, kHQ, kD]
    const float* __restrict__ ws,    // [R, kH, splits, q_rows, kRatio, kD+2]
    int splits, int q_rows) {
  const int r = blockIdx.y;
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t ws_sp = (int64_t)q_rows * kRatio * kRowStride;

  for (int qq = 0; qq < q_rows; qq++) {
    for (int slot = tid; slot < kRatio * kD; slot += kThreads) {
      const int i = slot / kD;
      const int d = slot % kD;
      const int64_t base =
          ((int64_t)(r * kH + h) * splits) * ws_sp +
          (int64_t)qq * (kRatio * kRowStride);
      // Per-split m, l and acc at (qq, i, d).
      float m_max = -INFINITY;
      for (int s = 0; s < splits; s++) {
        const float m =
            ws[base + s * ws_sp + i * kRowStride + kD];
        if (m > m_max) m_max = m;
      }
      __half o;
      if (m_max == -INFINITY) {
        o = __float2half(0.0f);
      } else {
        float denom = 0.0f;
        float sum = 0.0f;
        for (int s = 0; s < splits; s++) {
          const float* w =
              ws + base + s * ws_sp + i * kRowStride;
          const float w_s = expf(w[kD] - m_max);
          denom += w_s * w[kD + 1];
          sum += w_s * w[d];
        }
        o = __float2half(denom > 0.0f ? sum / denom : 0.0f);
      }
      out[((int64_t)(r * q_rows + qq) * kHQ + h * kRatio + i) * kD + d] = o;
    }
  }
}

}  // namespace

void sm70_flashdecode(torch::stable::Tensor& q, torch::stable::Tensor& kv,
                      torch::stable::Tensor& block_table,
                      torch::stable::Tensor& kv_lens,
                      torch::stable::Tensor& out, double k_scale,
                      double v_scale, double sm_scale, int64_t window_left,
                      int64_t splits, int64_t q_rows,
                      torch::stable::Tensor& workspace) {
  torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
  // dtypes are asserted by the python wrapper; only shapes are checked here.
  STD_TORCH_CHECK(q.size(1) == q_rows && q_rows >= 1 && q_rows <= kMaxQ,
                  "sm70_flashdecode: q dim 1 must equal q_rows (1..8)");
    STD_TORCH_CHECK(q.size(2) == kHQ && q.size(3) == kD,
                  "sm70_flashdecode: supports HQ=12 (TP2), D=256");
  const int64_t H = kv.size(1) / 2;
  const int64_t n_page = kv.size(2);
  STD_TORCH_CHECK(
      H == kH && n_page >= kN && n_page % kN == 0 && kv.size(3) == kF,
      "sm70_flashdecode: supports H=2, F=144, page = multiple of 16");
  const int64_t R = q.size(0);
  const int64_t bt_cols = block_table.size(1);
  STD_TORCH_CHECK(block_table.size(0) == R && kv_lens.size(0) == R,
                  "sm70_flashdecode: table/len mismatch");
  STD_TORCH_CHECK(out.size(0) == R && out.size(1) == q_rows &&
                      out.size(2) == kHQ && out.size(3) == kD,
                  "sm70_flashdecode: out shape mismatch");
  const int64_t ws_need =
      R * kH * splits * q_rows * kRatio * kRowStride;
  STD_TORCH_CHECK(workspace.numel() >= ws_need,
                  "sm70_flashdecode: workspace too small");
  STD_TORCH_CHECK(splits >= 1, "sm70_flashdecode: splits < 1");

  cudaStream_t stream = get_current_cuda_stream(q.get_device_index());

  dim3 grid(splits, kH, R);
  sm70_flashdecode_partition_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()),
      reinterpret_cast<const uint8_t*>(kv.data_ptr()),
      reinterpret_cast<const int*>(block_table.data_ptr()),
      reinterpret_cast<const int*>(kv_lens.data_ptr()),
      reinterpret_cast<float*>(workspace.data_ptr()),
      static_cast<float>(k_scale), static_cast<float>(v_scale),
      static_cast<float>(sm_scale), static_cast<int>(window_left),
      static_cast<int>(splits), static_cast<int>(q_rows),
      static_cast<int>(bt_cols), static_cast<int>(n_page), kv.stride(0),
      (int64_t)H * n_page * kF);

  dim3 rgrid(kH, R);
  sm70_flashdecode_reduce_kernel<<<rgrid, kThreads, 0, stream>>>(
      reinterpret_cast<__half*>(out.mutable_data_ptr()),
      reinterpret_cast<const float*>(workspace.data_ptr()),
      static_cast<int>(splits), static_cast<int>(q_rows));

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "sm70_flashdecode launch failed: ",
                  cudaGetErrorString(err));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_flashdecode_ops) {
  sm70_flashdecode_ops.def(
      "sm70_flashdecode(Tensor q, Tensor kv_cache, Tensor block_table, "
      "Tensor kv_lens, Tensor out, float k_scale, float v_scale, "
      "float sm_scale, int window_left, int splits, int q_rows, "
      "Tensor workspace) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_flashdecode_ops) {
  sm70_flashdecode_ops.impl("sm70_flashdecode",
                            TORCH_BOX(&::sm70_flashdecode));
}
