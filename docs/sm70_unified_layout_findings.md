# SM70 (V100) NVFP4 — 1x Layout / Prefill-Decode Design Findings

Date: 2026-09-15 (unattended spike). Box 8.178 (2x V100-SXM2-32GB), branch sm70.
Goal: design a layout MORE excellent than 1Cat (1x unified), serving prefill+decode.
Method: measured benchmark on the real ops (Marlin / QPN2 / dequant / cuBLAS), synthetic
4-bit weights, real layer shapes. No 1Cat model run (read-only ref).

## V100 constraints (primary sources)
- Only FP16 tensor cores (NO INT4/INT8 TC — those are Turing+). 4-bit GEMM MUST
  dequant->FP16 + HMMA. (GTC 2019 TC guide; Volta whitepaper.)
- HBM2 900 GB/s; L2 only 6 MB (weights >> L2, so small-M is HBM-bound).
- FP16 TC peak ~125 TF/s; ridge ~139 FLOP/byte.
- Marlin (Ampere-tuned: cp.async + deep smem pipelines) hits only ~25% occupancy on
  V100 (no cp.async, tighter smem/regs).

## Benchmark: prefill landscape (FFN k=5120 n=8704, qkv, down; M=1..1024)
Active path today: decode(M<=32)=QPN2, prefill(M>32)=Marlin. Both 4-bit, 2x storage.

  M   | Marlin   cuBLAS16  | who faster
  128 | 0.445    0.321     | cuBLAS 1.39x
  256 | 0.438    0.351     | cuBLAS 1.25x
  512 | 0.802    0.625     | cuBLAS 1.28x
  1024| 1.592    1.151     | cuBLAS 1.38x
Decode: QPN2 M=1 0.048ms (1.9TF/s, ~520 GB/s) vs cuBLAS 0.150ms -> QPN2 3x faster.

### Finding 1: Marlin prefill is the WORST option on V100.
  cuBLAS FP16 beats our Marlin at every prefill M (1.25-1.39x). At large M a fused
  4-bit kernel does the SAME FP16-TC work PLUS the dequant on CUDA cores -> it cannot
  beat cuBLAS (strictly more work, same TC). "Write a fused GEMM to beat cuBLAS" is
  the WRONG direction.

### Finding 2: QPN2 decode is optimal (keep it).
  4-bit weight (4x less traffic) + high occupancy -> 3x faster than cuBLAS at decode.
  No lever (decode is at the TP2 ceiling: ~50% GEMM + ~37% allreduce).

### Finding 3: the 1x scratch tax is the dequant, and it is 2.5x above floor.
  1Cat (and the 1x staged path) dequant 4-bit->FP16 per prefill call, then cuBLAS.
  The dequant (sm70_nvfp4_dequant) runs at ~300 GB/s eff, but the HW copy floor is
  ~774 GB/s (fill 813 / copy 774). => ~2.5x headroom in the dequant kernel
  (likely under-vectorized loads). FFN dequant = 0.377ms ~= the FFN GEMM (0.351ms)
  at M=256, so the tax is large at small prefill M and shrinks at large M
  (GEMM grows with M; dequant is M-independent).

### Finding 4 (REFUTED): naive stream-overlap does NOT hide the tax.
  dequant (stream A) || cuBLAS GEMM (stream B): speedup 1.00-1.12x. Both are
  HBM/SM-bound -> they contend, not overlap. "Hide the dequant under the GEMM"
  is not a reliable lever on V100.

## Refined design (data-driven)
Storage 1x 4-bit (drop the Marlin layout; keep the 4-bit that QPN2 + dequant use):
  - decode (M<=32): QPN2 (unchanged, 3x faster than cuBLAS).
  - prefill (M>32): dequant(4-bit->FP16) + cuBLAS FP16  [faster than Marlin].
  - tax: the dequant, per layer.
Net vs current (2x+Marlin): 1x => TP1 fits (20.56GB); prefill GEMM faster (cuBLAS>Marlin);
  residual = the dequant tax.
More-excellent-than-1Cat delta: optimize the dequant kernel (~2.5x, 300->700+ GB/s)
  => smaller tax than 1Cat pays. (The structural overlap idea is out, per Finding 4.)

## Decision point for user
This PIVOTS away from the Marlin prefill path (inverts prior Marlin work). Confirm
before implementing. 1x memory + cuBLAS prefill is the clear win; the "more excellent"
is the dequant optimization (bounded ~2.5x).

## Plan (if confirmed)
1. dequant kernel optimization: vectorized 4-bit->FP16 (target ~700+ GB/s). Bench vs 300.
2. Repoint prefill (M>32) from Marlin to dequant+cuBLAS; keep QPN2 for decode.
3. Drop Marlin layout from storage (2x->1x); keep Marlin code available until confirmed.
4. Verify TP1 fits (20.56GB) + prefill/decode throughput; run the service (systemd).

## Revision (post Triton-dequant probe) — dequant headroom is NOT trivially 2.5x
- A naive vectorized Triton dequant came out SLOWER than the C++ kernel (184 vs 296 GB/s).
  The 4-bit-read (2 codes/byte) + per-16-group E4M3-scale structure makes it hard to reach
  the 774 GB/s copy floor. The C++ at ~300 GB/s is the practical current; getting ~2x more
  needs a purpose-built kernel (16B vector 4-bit loads + explicit 16-group scale handling).
- => The dequant delta over 1Cat is BOUNDED (~1.5-2x on the tax), not a fundamental leap.
- => "More excellent than 1Cat" is LIMITED: 1Cats 1x+staged is already the right V100

## Revision (post Triton-dequant probe) - dequant headroom is NOT trivially 2.5x
- Naive vectorized Triton dequant came out SLOWER than the C++ kernel (184 vs 296 GB/s).
  The 4-bit-read (2 codes/byte) + per-16-group E4M3-scale structure makes it hard to reach
  the 774 GB/s copy floor. The C++ at ~300 GB/s is the practical current; getting ~2x more
  needs a purpose-built kernel (16B vector 4-bit loads + explicit 16-group scale handling).
- The dequant delta over 1Cat is therefore BOUNDED (~1.5-2x on the tax), not a fundamental leap.
- 1Cat 1x+staged is already the right V100 architecture; our only delta is dequant speed.
  The BIG win is fixing OUR current design (2x + Marlin prefill) to 1x + staged cuBLAS:
  TP1 fits AND prefill GEMM gets faster (cuBLAS beats Marlin 1.25-1.39x).
- Untested possibility (follow-up): overlap the dequant with GDN/attention compute (NOT the
  GEMM, which saturates SMs). GDN linear-attn is ~40% of prefill and uses different resources;
  dequant||GDN may overlap where dequant||GEMM did not.

## Revision (dequant root-caused, 2026-09-15) - the 3x dequant was a BUG
- Answered the question "is the slow dequant a Python problem / does C++ help?": NO.
  The dequant is ALREADY a CUDA kernel (torch.ops._C.sm70_nvfp4_dequant). A
  pipelined (dispatch-overlapped, no per-iter sync) loop measured the same
  ~300-320 GB/s as single-shot, so host dispatch/allocation overhead is
  negligible. The slow part is the kernel, not the language or the dispatch.
- A purpose-built vectorized kernel (16 codes/thread, global-L1 LUT via
  __restrict__ arg, FP16 __hmul math, 16-byte store) that measured ~900 GB/s
  was WRONG: it used a misaligned 4-byte read (uint32 reinterpret) of the
  2-byte-aligned __half output array, which scrambled the output (verified
  against a torch reference: the values were garbage). With the store fixed to
  aligned __half2 field assignment, the identical kernel is ~315 GB/s - no
  gain over the original ~320.
- => The CORRECT dequant is ~320 GB/s on a V100-SXM2, same as the original and
  the same order as 1Cat dequant. The "2.5x headroom (300 -> 700)" in Finding 3
  was a hypothesis (the copy floor) and was NOT demonstrated by a correct
  kernel. Any "3x / 900 GB/s" numbers were a garbage-output artifact of the
  misaligned-store bug.
- => "More excellent than 1Cat" is NOT via dequant speed (we match 1Cat at
  ~320 GB/s). The 1x migration value is unchanged but narrower: TP1 fits (the
  hard goal, 20.56 GB on a 32 GB V100) + faster prefill GEMM (cuBLAS beats
  Marlin 1.25-1.39x). The dequant tax stays ~320 GB/s (the 1x scratch tax).
- Lesson: verify a fast kernel is CORRECT (vs an independent reference), not
  just fast - the 900 GB/s was a measurement of a buggy kernel.
