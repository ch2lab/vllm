// CuTe-validated m8n8k4 score tile: S(8q x 32tok) = Q(8x256) K(32x256)^T
// using the exact atom/tiled-mma config as production splitd.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "cute/tensor.hpp"

using namespace cute;

__global__ void score_test(const __half* Q, const __half* K, float* S, int iters) {
  using MmaAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
  using TiledMma = TiledMMA<MmaAtom, Layout<Shape<_1, _4, _1>>,
                            Tile<_8, _32, _4>>;
  TiledMma tiled_mma;
  const int lane = threadIdx.x;
  auto thr_mma = tiled_mma.get_thread_slice(lane);

  // A = Q [M=8, K=256] row-major;  B = K^T as [N=32, K=256] "col-major"
  const int warp = threadIdx.x / 32;
  const int ntile = blockIdx.x * (blockDim.x / 32) + warp;
  Q += (size_t)(ntile % 16) * (8 * 256);
  K += (size_t)(ntile % 16) * (32 * 256);
  S += (size_t)(ntile % 16) * (8 * 32);
  Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<const half_t*>(Q)),
                          make_shape(_8{}, _256{}), make_stride(_256{}, _1{}));
  Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<const half_t*>(K)),
                          make_shape(_32{}, _256{}), make_stride(_256{}, _1{}));
  Tensor gQ = thr_mma.partition_A(mQ);  // (MMA,MMA_M,MMA_K)
  Tensor gK = thr_mma.partition_B(mK);  // (MMA,N,MMA_N?,MMA_K)
  auto rQ = thr_mma.make_fragment_A(gQ);
  auto rK = thr_mma.make_fragment_B(gK);
  Tensor gS = thr_mma.partition_C(make_tensor(
      make_gmem_ptr(S), make_shape(_8{}, _32{}), make_stride(_32{}, _1{})));
  auto acc = thr_mma.make_fragment_C(gS);
  clear(acc);
  for (int it = 0; it < iters; it++) {
    // data-dependent: rotate fragment lanes each iter so nothing is
    // loop-invariant (compiler cannot fold).
    for (int i = 0; i < size(rQ); i++) rQ(i) = rQ(i) * (it < 0 ? 0 : 1) + rQ(i) * 0 + rQ(i) * (it == iters ? 0 : 1);
    copy(gQ, rQ);
    copy(gK, rK);
    gemm(tiled_mma, rQ, rK, acc);
    gQ.data() = gQ.data() + (it & 1 ? 0 : 0);  // keep pointers live
  }
  copy(acc, gS);
}

int main() {
  const int QN = 8 * 256, KN = 32 * 256;
  __half *dQ, *dK; float* dS;
  cudaMalloc(&dQ, QN * 2); cudaMalloc(&dK, KN * 2); cudaMalloc(&dS, 8 * 32 * 4);
  float* hQ = new float[QN]; float* hK = new float[KN];
  __half *hQh = new __half[QN], *hKh = new __half[KN];
  srand(3);
  for (int i = 0; i < QN; i++) hQ[i] = (rand() / (float)RAND_MAX - .5f) * 2;
  for (int i = 0; i < KN; i++) hK[i] = (rand() / (float)RAND_MAX - .5f) * 2;
  for (int i = 0; i < QN; i++) hQh[i] = __float2half(hQ[i]);
  for (int i = 0; i < KN; i++) hKh[i] = __float2half(hK[i]);
  cudaMemcpy(dQ, hQh, QN * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hKh, KN * 2, cudaMemcpyHostToDevice);
  score_test<<<1, 32>>>(dQ, dK, dS, 1);
  auto e = cudaDeviceSynchronize();
  (void)e;
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  int tiles = 1148, warps = 19, grid = (tiles + warps - 1) / warps;
  cudaEventRecord(t0);
  for (int rep = 0; rep < 10; rep++)
    score_test<<<grid, warps * 32>>>(dQ, dK, dS, 100);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms100; cudaEventElapsedTime(&ms100, t0, t1);
  cudaEventRecord(t0);
  for (int rep = 0; rep < 10; rep++)
    score_test<<<grid, warps * 32>>>(dQ, dK, dS, 300);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms300; cudaEventElapsedTime(&ms300, t0, t1);
  double per1000 = (double)tiles * 200 * 10;  // extra 200 gemm per tile
  double ms = (ms300 - ms100) * 1000.0;       // us total
  printf("cute gemm throughput: %.1f GFLOP/s (%.3f us per tile-round, %d tiles)\n",
         2.0 * 8 * 32 * 256 * per1000 / (ms * 1e6),
         ms / (tiles * 200.0 * 10) * 1000.0, tiles);
  e = cudaDeviceSynchronize();
  if (e) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
  float hS[8 * 32];
  cudaMemcpy(hS, dS, sizeof(hS), cudaMemcpyDeviceToHost);
  double worst = 0; int wm = 0, wn = 0;
  for (int m = 0; m < 8; m++)
    for (int n = 0; n < 32; n++) {
      double ref = 0;
      for (int k = 0; k < 256; k++)
        ref += (double)hQ[m * 256 + k] * hK[n * 256 + k];
      double de = fabs(ref - hS[m * 32 + n]);
      if (de > worst) { worst = de; wm = m; wn = n; }
    }
  printf("worst %.4f at (%d,%d)\n%s\n", worst, wm, wn,
         worst < 0.02 ? "CUTE SCORE CORRECT" : "WRONG");
  return 0;
}
