#include <cute/tensor.hpp>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
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
    int64_t page_stride, int64_t side_off,
    unsigned long long* __restrict__ prof) {
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

  unsigned long long ts = 0, t_stage = 0, t_deq = 0, t_gemm = 0;
  for (int c = c0; c <= c1; c++) {
    ts = clock64();
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
    t_deq += clock64() - ts; ts = clock64();
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
    t_gemm += clock64() - ts;
    const unsigned long long tp = clock64();
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
    if (threadIdx.x == 0) {
      atomicAdd(&prof[0], t_stage);
      atomicAdd(&prof[1], t_deq);
      atomicAdd(&prof[2], t_gemm);
      atomicAdd(&prof[3], tp ? clock64() - tp : 0ULL);
      atomicAdd(&prof[4], (unsigned long long)(c1 - c0 + 1));
    }
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

}
int main() {
  using fdhmma::sm70_flashdecode_hmma_partition_kernel;
  const int ctx = 131072, N = 16, F = 144, H = 2, HQ = 12, D = 256;
  const int nb = ctx / N;
  const int splits = 120;
  uint8_t *kv; cudaMalloc(&kv, (size_t)nb * 2 * H * N * F);
  __half *q; cudaMalloc(&q, HQ * D * 2);
  int *bt; cudaMalloc(&bt, nb * 4);
  int *kl; cudaMalloc(&kl, 4);
  float *ws; cudaMalloc(&ws, (size_t)H * splits * 6 * (D + 2) * 8);
  unsigned long long *prof; cudaMalloc(&prof, 5 * 8);
  cudaMemset(prof, 0, 40);
  int *h_bt = (int*)malloc(nb * 4);
  for (int i = 0; i < nb; i++) h_bt[i] = i;
  cudaMemcpy(bt, h_bt, nb * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(kl, &ctx, 4, cudaMemcpyHostToDevice);
  dim3 grid(splits, H, 1); dim3 block(192);
  auto launch = [&]() {
    sm70_flashdecode_hmma_partition_kernel<<<grid, block>>>(
        q, kv, bt, kl, ws, 1.f, 1.f, 1.f/16, -1, splits, 1, nb, N,
        (int64_t)2 * H * N * F, (int64_t)H * N * F, prof);
  };
  for (int i = 0; i < 3; i++) launch();
  cudaDeviceSynchronize();
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < 10; i++) launch();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  unsigned long long hp[5];
  cudaMemcpy(hp, prof, 40, cudaMemcpyDeviceToHost);
  printf("per-call: %.3f ms\n", ms / 10);
  const char* names[5] = {"vstage", "kdeq ", "gemm ", "pv   "};
  unsigned long long tot = 0;
  for (int i = 0; i < 4; i++) tot += hp[i];
  for (int i = 0; i < 4; i++)
    printf("%s: %5.1f%%  %6.0f cyc/CTA-call\n", names[i],
           100.0 * hp[i] / tot, (double)hp[i] / 10 / (splits * H));
  return 0;
}
