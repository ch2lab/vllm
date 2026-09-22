#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
__device__ __forceinline__ uint32_t pack2(float x, float y) {
  __half2 h = __floats2half2_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&h);
}
__global__ void probe(float* out, int mode, int kseg) {
  const int lane = threadIdx.x & 31;
  const int t = lane & 7;
  float av = (mode == 0) ? (float)(t + 1) : 1.f;   // A row marker
  float bv = (mode == 0) ? 1.f : (float)(1u << (t + 1));  // B col marker (pow2)
  float zero = 0.f;
  uint32_t a0 = pack2(zero, zero), a1 = pack2(zero, zero);
  uint32_t b0 = pack2(zero, zero), b1 = pack2(zero, zero);
  if (mode == 0) { if (kseg==0) { a0 = pack2(av, zero); a1 = pack2(av, zero); }
                  else { a0 = pack2(zero, av); a1 = pack2(zero, av); } }
  else           { if (kseg==0) { b0 = pack2(bv, zero); b1 = pack2(bv, zero); }
                   else { b0 = pack2(zero, bv); b1 = pack2(zero, bv); } }
  float d[8] = {0,0,0,0,0,0,0,0};
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
  for (int i = 0; i < 8; i++) out[lane * 8 + i] = d[i];
}
int main() {
  float* d_out; cudaMalloc(&d_out, 256 * 4);
  for (int mode = 0; mode < 2; mode++) for (int kseg = 0; kseg < 2; kseg++) {
    probe<<<1, 32>>>(d_out, mode, kseg);
    cudaDeviceSynchronize();
    float h[256];
    cudaMemcpy(h, d_out, 256 * 4, cudaMemcpyDeviceToHost);
    printf("=== %s kseg%d ===\n", mode ? "Bcol" : "Arow", kseg);
    for (int l = 0; l < 8; l++) {
      printf("lane%2d:", l);
      for (int i = 0; i < 8; i++) {
        int v = (int)h[l * 8 + i];
        if (mode) { int p = 0, x = v; while (x > 1 && !(x & 1)) { x >>= 1; p++; } printf(" %s%d", p?"c":"?", p ? p - 1 : 0); }
        else printf(" r%d", v - 1);
      }
      printf("\n");
    }
  }
  return 0;
}
