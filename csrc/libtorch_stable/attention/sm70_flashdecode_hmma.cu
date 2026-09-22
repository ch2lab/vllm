
#include <cute/tensor.hpp>

#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

// Named (not anonymous) namespace: cutlass mma_traits_sm70.hpp contains its
// own anonymous namespace, and two anonymous namespaces in one nvcc TU make
// the generated host-stub _NV_ANON_NAMESPACE reference ambiguous.
namespace fdhmma {

using namespace cute;

constexpr int kHQ = 12;
constexpr int kH = 2;
constexpr int kRatio = 6;
constexpr int kD = 256;
constexpr int kN = 16;
constexpr int kMaxQ = 8;
constexpr int kDD = kD / 2;
constexpr int kSD = kD / 16;
constexpr int kF = kDD + kSD;
constexpr int kThreads = 128;
constexpr int kPartMaxThreads = kRatio * 32;   // 192
constexpr int kPartWarps = kRatio;
constexpr int kLaneDims = 8;
constexpr int kRowStride = kD + 2;
constexpr int kDChunks = 4;                     // 64-dim chunks

using SmemTileAtom = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

using FmAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
using FmTiled = TiledMMA<FmAtom, Layout<Shape<_1, _2, _1>>,
                         Tile<_8, Int<kN>, _4>>;

__device__ __forceinline__ float e2m1_f(unsigned code) {
  const unsigned m = code & 7;
  float mag;
  if (m < 5) {
    mag = m * 0.5f;
  } else if (m < 7) {
    mag = (m == 5) ? 3.0f : 4.0f;
  } else {
    mag = 6.0f;
  }
  return (code & 8) ? -mag : mag;
}

__device__ __forceinline__ float e4m3_to_float(unsigned char b) {
  const bool sgn = b & 0x80;
  const int e = (b >> 3) & 0xF;
  const int m = b & 7;
  float v;
  if (e == 0) {
    v = m * 0.001953125f;
  } else {
    v = (1.0f + m * 0.125f) * exp2f((float)e - 7.0f);
  }
  return sgn ? -v : v;
}

__global__ void __launch_bounds__(kPartMaxThreads, 4)
sm70_flashdecode_hmma_partition_kernel(
    const __half* __restrict__ q,
    const uint8_t* __restrict__ kv,
    const int* __restrict__ bt,
    const int* __restrict__ kv_lens,
    float* __restrict__ ws,
    float k_scale, float v_scale, float sm_scale,
    int window_left, int splits, int q_rows, int bt_cols, int n_page,
    int64_t page_stride, int64_t side_off) {
  const int r = blockIdx.z / q_rows;
  const int qq = blockIdx.z % q_rows;
  const int h = blockIdx.y;
  const int sp = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // 8 rows so the (8,x) gemm A-operand row 7 lands inside the allocation
  // (its outputs are discarded; only kRatio rows carry real heads).
  __shared__ __half q_smem[kMaxQ * kD];             // 4 KB, this row
  __shared__ uint8_t kv_raw[kN * kF];                  // 2.3 KB V packed
  __shared__ __half k_tile_storage_full[kN * kD];    // 8 KB (gemm operand)
  // Score partials: gemm warp g computes S[8 heads x 16 tok] for dim chunk
  // g (M-rows = the 6 head q-vectors + 2 zero pad rows, a real GEMM slice,
  // not a stride-0 broadcast); softmax sums the kDChunks partials per head
  // row. kMaxQ rows per chunk because the C-tile is 8 rows tall.
  __shared__ float s_shA[kDChunks][kMaxQ][kN];
  // v3h-style dequant tables: e2t[byte] = the 2 nibble magnitudes,
  // e4t[side*256+byte] = e4m3 value x layer scale (exact in fp32).
  __shared__ float2 e2t[256];
  __shared__ float e4t[512];

  auto sK = make_tensor(make_smem_ptr(
      reinterpret_cast<half_t*>(k_tile_storage_full)),
      tile_to_shape(SmemTileAtom{}, Shape<Int<kN>, Int<kD>>{}));

  const int kv_len = __ldg(kv_lens + r);
  const int64_t ws_sp = (int64_t)q_rows * kRatio * kRowStride;
  const int64_t ws_base =
      ((int64_t)(r * kH + h) * splits + sp) * ws_sp;

  float* slot = ws + ws_base + qq * (kRatio * kRowStride);
  auto write_empty = [&]() {
    for (int idx = tid; idx < kRatio * kRowStride; idx += kPartMaxThreads) {
      const int rem = idx % kRowStride;
      slot[idx] = (rem == kD) ? -INFINITY
                : (rem == kD + 1) ? 0.0f : 0.0f;
    }
  };

  int lo = window_left >= 0 ? kv_len - 1 - window_left : 0;
  if (lo < 0) lo = 0;
  write_empty();
  if (kv_len == 0 || lo >= kv_len) return;
  const int64_t t_a = (int64_t)lo + ((int64_t)kv_len - lo) * sp / splits;
  const int64_t t_b =
      (int64_t)lo + ((int64_t)kv_len - lo) * (sp + 1) / splits;
  if (t_a >= t_b) return;
  const int c0 = (int)(t_a / kN);
  const int c1 = (int)((t_b - 1) / kN);
  const int hi = kv_len - q_rows + qq + 1;

  // ---- Q into smem once (this row, 6 heads), sm_scale folded ----
  {
    const __half* src =
        q + ((int64_t)(r * q_rows + qq) * kHQ + h * kRatio) * kD;
    for (int idx = tid; idx < kMaxQ * kD / 2; idx += kPartMaxThreads) {
      const int t = idx / (kD / 2);
      const int d2 = (idx % (kD / 2)) * 2;
      const __half2 s = __float2half2_rn(sm_scale);
      const __half2 v = (t < kRatio)
          ? *reinterpret_cast<const __half2*>(src + t * kD + d2)
          : __half2();  // padded M-rows: zeros, outputs discarded
      reinterpret_cast<__half2*>(&q_smem[0])[idx] = __hmul2(v, s);
    }
    for (int b = tid; b < 256; b += kPartMaxThreads) {
      e2t[b] = make_float2(e2m1_f(b & 0xF), e2m1_f(b >> 4));
      const float sfv = e4m3_to_float(static_cast<unsigned>(b));
      e4t[b] = sfv * k_scale;
      e4t[256 + b] = sfv * v_scale;
    }
  }
  __syncthreads();

  // ---- gemm setup: A = q_smem as [8 head-rows, 256]; one gemm warp per
  // 64-dim chunk.
  FmTiled tiled_mma;
  const int lane_w = tid & 31;
  auto thr_mma = tiled_mma.get_thread_slice(lane_w);
  const int d_base = kLaneDims * lane;
  float acc[kLaneDims];
#pragma unroll
  for (int j = 0; j < kLaneDims; j++) acc[j] = 0.0f;
  float m_run = -INFINITY, l_run = 0.0f;

  for (int c = c0; c <= c1; c++) {
    const int cbase = c * kN;
    const int blk = cbase / n_page;
    const int n0_in = cbase - blk * n_page;
    const int64_t base =
        (int64_t)__ldg(bt + r * bt_cols + blk) * page_stride;
    // ---- V: packed bytes -> smem (PV reads it in place below). K is
    // dequantized straight from global into the fp16 tile in the next
    // loop, so it needs no smem staging round-trip.
    for (int cc = tid; cc < kN * (kF / 16); cc += kPartMaxThreads) {
      const int n = cc / (kF / 16);
      const int f16 = cc % (kF / 16);
      const int ntok = n0_in + n;
      int64_t off;
      if (f16 < kDD / 16) {
        off = (int64_t)h * (n_page * kDD) + ntok * kDD + f16 * 16;
      } else {
        off = (int64_t)kH * (n_page * kDD) +
              (int64_t)h * (n_page * kSD) + ntok * kSD +
              (f16 - kDD / 16) * 16;
      }
      const uint4 src = *reinterpret_cast<const uint4*>(
          kv + base + side_off + off);
      *reinterpret_cast<uint4*>(&kv_raw[n * kF + f16 * 16]) = src;
    }
    // ---- K: global -> registers -> fp16 gemm tile (uint4 of packed
    // bytes = 32 dims = two 16-dim scale groups) ----
    for (int e = tid; e < kN * 8; e += kPartMaxThreads) {
      const int n = e >> 3;
      const int u = e & 7;
      const int ntok = n0_in + n;
      const uint4 d4 = *reinterpret_cast<const uint4*>(
          kv + base + (int64_t)h * (n_page * kDD) + ntok * kDD + u * 16);
      const uint8_t* sc =
          kv + base + (int64_t)kH * (n_page * kDD) +
          (int64_t)h * (n_page * kSD) + ntok * kSD;
      const float sfs[2] = {e4t[sc[2 * u]], e4t[sc[2 * u + 1]]};
      const unsigned words[4] = {d4.x, d4.y, d4.z, d4.w};
#pragma unroll
      for (int w4 = 0; w4 < 4; w4++) {
        const float sf = sfs[w4 >> 1];
        const unsigned wd = words[w4];
        const float2 n01 = e2t[wd & 0xFF], n23 = e2t[(wd >> 8) & 0xFF];
        const float2 n45 = e2t[(wd >> 16) & 0xFF], n67 = e2t[wd >> 24];
        const __half2 hv[4] = {
            __floats2half2_rn(n01.x * sf, n01.y * sf),
            __floats2half2_rn(n23.x * sf, n23.y * sf),
            __floats2half2_rn(n45.x * sf, n45.y * sf),
            __floats2half2_rn(n67.x * sf, n67.y * sf)};
        *reinterpret_cast<uint4*>(&sK(n, 32 * u + 8 * w4)) =
            *reinterpret_cast<const uint4*>(hv);
      }
    }
    __syncthreads();
    // ---- score gemm: warp g < kDChunks computes S[8 heads x 16 tok] for
    // dim chunk g (M-rows = 6 head q-vectors + 2 zero pad rows) ----
    if (warp < kDChunks) {
      auto accT = thr_mma.partition_C(make_tensor(
          make_gmem_ptr(&s_shA[warp][0][0]),
          make_shape(_8{}, Int<kN>{}), make_stride(Int<kN>{}, _1{})));
      auto rAcc = thr_mma.make_fragment_C(accT);
      clear(rAcc);
      auto sKc = cute::local_tile(sK, make_shape(_16{}, _64{}),
                                  make_coord(0, warp));
      auto gB = thr_mma.partition_B(sKc);
      auto rK = thr_mma.make_fragment_B(gB);
      copy(gB, rK);
      auto mA_chunk = make_tensor(
          make_gmem_ptr(reinterpret_cast<const half_t*>(
              &q_smem[warp * 64])),
          make_shape(_8{}, _64{}), make_stride(Int<kD>{}, _1{}));
      auto rQc = thr_mma.make_fragment_A(thr_mma.partition_A(mA_chunk));
      copy(thr_mma.partition_A(mA_chunk), rQc);
      gemm(tiled_mma, rQc, rK, rAcc);
      copy(rAcc, accT);
    }
    __syncthreads();
    // ---- softmax + PV over this chunk's 16 scores (head = warp) ----
    const int rt0 = cbase > (int)t_a ? 0 : (int)(t_a - cbase);
    const int rt1p = cbase + kN < (int)t_b ? kN : (int)(t_b - cbase);
    int rt1 = rt1p;
    if (cbase + rt1 > hi) rt1 = hi - cbase;
    if (rt1 > kN) rt1 = kN;
    if (rt1 > rt0) {
      const float* p0 = &s_shA[0][warp][0];
      const float* p1 = &s_shA[1][warp][0];
      const float* p2 = &s_shA[2][warp][0];
      const float* p3 = &s_shA[3][warp][0];
      float m_tile = -INFINITY;
      for (int t = rt0; t < rt1; t++)
        m_tile = fmaxf(m_tile, (p0[t] + p1[t]) + (p2[t] + p3[t]));
      const float m_new = fmaxf(m_run, m_tile);
      const float alpha = (m_run == -INFINITY) ? 0.0f
                        : expf(m_run - m_new);
      l_run *= alpha;
#pragma unroll
      for (int j = 0; j < kLaneDims; j++) acc[j] *= alpha;
      m_run = m_new;
      for (int t = rt0; t < rt1; t++) {
        const float p = expf((p0[t] + p1[t]) + (p2[t] + p3[t]) - m_new);
        l_run += p;
        // PV from raw bytes: this lane's 8 dims = 4 contiguous packed bytes
        // sharing one scale group.
        const uint32_t vb4 = *reinterpret_cast<const uint32_t*>(
            &kv_raw[t * kF + (d_base >> 1)]);
        const float sf = e4t[256 + kv_raw[t * kF + kDD + (d_base >> 4)]];
        const float2 v01 = e2t[vb4 & 0xFF], v23 = e2t[(vb4 >> 8) & 0xFF];
        const float2 v45 = e2t[(vb4 >> 16) & 0xFF], v67 = e2t[vb4 >> 24];
        acc[0] = fmaf(p * sf, v01.x, acc[0]);
        acc[1] = fmaf(p * sf, v01.y, acc[1]);
        acc[2] = fmaf(p * sf, v23.x, acc[2]);
        acc[3] = fmaf(p * sf, v23.y, acc[3]);
        acc[4] = fmaf(p * sf, v45.x, acc[4]);
        acc[5] = fmaf(p * sf, v45.y, acc[5]);
        acc[6] = fmaf(p * sf, v67.x, acc[6]);
        acc[7] = fmaf(p * sf, v67.y, acc[7]);
      }
    }
    __syncthreads();   // tiles reuse next chunk
  }

  // ---- direct write of this split's partial (single writer per slot) ----
  {
    float* rp = slot + warp * kRowStride + d_base;
#pragma unroll
    for (int j = 0; j < kLaneDims; j++) rp[j] = acc[j];
    if (lane == 0) {
      slot[warp * kRowStride + kD] = m_run;
      slot[warp * kRowStride + kD + 1] = l_run;
    }
  }
}

// Two-stage LSE reduce: stage 1 splits the O(splits) read loop across
// kSplitChunks more CTAs (24 CTAs at R=1 latency-bound the whole reduce at
// ~40GB/s effective; 4x CTAs -> ~4x), emitting per-chunk (m, l, acc)
// triples; stage 2 merges the kSplitChunks triples (4 iterations) into out.
constexpr int kSplitChunks = 4;

__global__ void __launch_bounds__(kThreads, 4)
sm70_fd_hmma_reduce1_kernel(
    const float* __restrict__ ws,  // [R, kH, splits, q_rows, kRatio, kD+2]
    float* __restrict__ part,       // [kSplitChunks, R, kH, q_rows, kRatio, kD+2]
    int splits, int q_rows, int R) {
  const int h = blockIdx.x;
  const int r = blockIdx.y;
  const int zs = (kRatio * kD + kThreads - 1) / kThreads;  // 12
  const int ch = blockIdx.z / zs;
  const int slot = (blockIdx.z % zs) * kThreads + threadIdx.x;
  if (slot >= kRatio * kD) {
    return;
  }
  const int i = slot / kD;
  const int d = slot % kD;
  const int64_t psz = (int64_t)q_rows * kRatio * kRowStride;
  const int cs = (splits + kSplitChunks - 1) / kSplitChunks;
  const int s0 = ch * cs;
  const int s1 = min(s0 + cs, splits);
  // ws block for (r, h) starts at (r*kH + h)*splits splits; this CTA
  // covers splits [s0, s1) within it.
  const int64_t w_off =
      (((int64_t)(r * kH + h) * splits) + s0) * psz +
      (int64_t)i * kRowStride;
  const int64_t p_off =
      ((int64_t)ch * R + r) * kH * psz + h * psz + (int64_t)i * kRowStride;

  for (int qq = 0; qq < q_rows; qq++) {
    const float* wq =
        ws + w_off + (int64_t)qq * (kRatio * kRowStride);
    float m_max = -INFINITY;
#pragma unroll 4
    for (int s = s0; s < s1; s++) {
      m_max = fmaxf(m_max, wq[(s - s0) * psz + kD]);
    }
    float l = 0.0f;
    float acc = 0.0f;
    if (m_max != -INFINITY) {
#pragma unroll 4
      for (int s = s0; s < s1; s++) {
        const float* w = wq + (s - s0) * psz;
        const float w_s = expf(w[kD] - m_max);
        l += w_s * w[kD + 1];
        acc += w_s * w[d];
      }
    }
    float* p = part + p_off + (int64_t)qq * (kRatio * kRowStride);
    p[d] = acc;
    p[kD] = m_max;
    p[kD + 1] = l;
  }
}

__global__ void __launch_bounds__(kThreads, 4)
sm70_fd_hmma_reduce2_kernel(
    __half* __restrict__ out,  // [R, q_rows, kHQ, kD]
    const float* __restrict__ part,
    int splits, int q_rows, int R) {
  const int r = blockIdx.y;
  const int h = blockIdx.x;
  const int slot = blockIdx.z * kThreads + threadIdx.x;
  if (slot >= kRatio * kD) {
    return;
  }
  const int i = slot / kD;
  const int d = slot % kD;
  const int64_t psz = (int64_t)q_rows * kRatio * kRowStride;

  for (int qq = 0; qq < q_rows; qq++) {
    const int64_t pbase =
        h * psz + (int64_t)qq * (kRatio * kRowStride) +
        (int64_t)i * kRowStride;
    // Merge the kSplitChunks (m, l, acc) triples.
    float m = -INFINITY, l = 0.0f, sum = 0.0f;
#pragma unroll
    for (int ch = 0; ch < kSplitChunks; ch++) {
      const float* p =
          part + ((int64_t)ch * R + r) * kH * psz + pbase;
      const float mc = p[kD];
      const float M = fmaxf(m, mc);
      const float w_o = (m == -INFINITY) ? 0.0f : expf(m - M);
      const float w_c = (mc == -INFINITY) ? 0.0f : expf(mc - M);
      sum = sum * w_o + p[d] * w_c;
      l = l * w_o + p[kD + 1] * w_c;
      m = M;
    }
    __half o;
    if (m == -INFINITY) {
      o = __float2half(0.0f);
    } else {
      sum = (l > 0.0f) ? sum / l : 0.0f;
      o = __float2half(sum);
    }
    out[((int64_t)(r * q_rows + qq) * kHQ + h * kRatio + i) * kD + d] = o;
  }
}

}  // namespace fdhmma

using namespace fdhmma;

void sm70_flashdecode_hmma(torch::stable::Tensor& q, torch::stable::Tensor& kv,
                      torch::stable::Tensor& block_table,
                      torch::stable::Tensor& kv_lens,
                      torch::stable::Tensor& out, double k_scale,
                      double v_scale, double sm_scale, int64_t window_left,
                      int64_t splits, int64_t q_rows,
                      torch::stable::Tensor& workspace) {
  torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
  // dtypes are asserted by the python wrapper; only shapes are checked here.
  STD_TORCH_CHECK(q.size(1) == q_rows && q_rows >= 1 && q_rows <= kMaxQ,
                  "sm70_flashdecode_hmma: q dim 1 must equal q_rows (1..8)");
    STD_TORCH_CHECK(q.size(2) == kHQ && q.size(3) == kD,
                  "sm70_flashdecode_hmma: supports HQ=12 (TP2), D=256");
  const int64_t H = kv.size(1) / 2;
  const int64_t n_page = kv.size(2);
  STD_TORCH_CHECK(
      H == kH && n_page >= kN && n_page % kN == 0 && kv.size(3) == kF,
      "sm70_flashdecode_hmma: supports H=2, F=144, page = multiple of 16");
  const int64_t R = q.size(0);
  const int64_t bt_cols = block_table.size(1);
  STD_TORCH_CHECK(block_table.size(0) == R && kv_lens.size(0) == R,
                  "sm70_flashdecode_hmma: table/len mismatch");
  STD_TORCH_CHECK(out.size(0) == R && out.size(1) == q_rows &&
                      out.size(2) == kHQ && out.size(3) == kD,
                  "sm70_flashdecode_hmma: out shape mismatch");
  // Workspace layout: [partition ws | reduce stage-1 partials].
  // Partials: [kSplitChunks, R, kH, q_rows, kRatio, kRowStride].
  const int64_t ws_part =
      R * kH * splits * q_rows * kRatio * kRowStride;
  const int64_t ws_red =
      kSplitChunks * R * kH * q_rows * kRatio * kRowStride;
  const int64_t ws_need = ws_part + ws_red;
  STD_TORCH_CHECK(workspace.numel() >= ws_need,
                  "sm70_flashdecode_hmma: workspace too small");
  STD_TORCH_CHECK(splits >= 1, "sm70_flashdecode_hmma: splits < 1");

  cudaStream_t stream = get_current_cuda_stream(q.get_device_index());

  dim3 grid(splits, kH, R * q_rows);  // v3h: per-row CTA grid
  dim3 block(kRatio * 32);
  sm70_flashdecode_hmma_partition_kernel<<<grid, block, 0, stream>>>(
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

  const float* ws_ptr = reinterpret_cast<const float*>(workspace.data_ptr());
  float* part_ptr =
      reinterpret_cast<float*>(workspace.data_ptr()) + ws_part;
  const int zslots = (kRatio * kD + kThreads - 1) / kThreads;

  dim3 rgrid1(kH, R, kSplitChunks * zslots);
  sm70_fd_hmma_reduce1_kernel<<<rgrid1, kThreads, 0, stream>>>(
      ws_ptr, part_ptr, static_cast<int>(splits),
      static_cast<int>(q_rows), static_cast<int>(R));

  dim3 rgrid2(kH, R, zslots);
  sm70_fd_hmma_reduce2_kernel<<<rgrid2, kThreads, 0, stream>>>(
      reinterpret_cast<__half*>(out.mutable_data_ptr()), part_ptr,
      static_cast<int>(splits), static_cast<int>(q_rows),
      static_cast<int>(R));

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "sm70_flashdecode launch failed: ",
                  cudaGetErrorString(err));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_fd_hmma_ops) {
  sm70_fd_hmma_ops.def(
      "sm70_flashdecode_hmma(Tensor q, Tensor kv_cache, Tensor block_table, "
      "Tensor kv_lens, Tensor out, float k_scale, float v_scale, "
      "float sm_scale, int window_left, int splits, int q_rows, "
      "Tensor workspace) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_fd_hmma_ops) {
  sm70_fd_hmma_ops.impl("sm70_flashdecode_hmma",
                            TORCH_BOX(&::sm70_flashdecode_hmma));
}
