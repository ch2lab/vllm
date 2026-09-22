# SM70 Flash-V100 (sm70_v37) Prefill Port — Design

Date: 2026-09-17
Status: proposed
Scope: vllm fork, branch `sm70`, SM70/V100 target
Related: M1 in-kernel decode (shipped), M2 transient-shadow prefill (shipped), M3 FP8 compat (separate work item, out of scope here)

## 1. Purpose and success criteria

Close the long-context prefill gap vs the 1Cat V100 production stack (measured
~1.8–2× slower at 27K–54K context) by porting 1Cat's hand-tuned SM70
full-attention prefill kernel (`sm70_v37`) into this fork, replacing the
flashinfer FA2 paged-prefill call for the target model's 16 full-attention
layers.

Success criteria (all must hold on the running server, target config
`/root/run/run2.sh` + `Qwen3.8-27B.yaml` unchanged):

1. 54K cold prefill TTFT ≤ 38 s (beats 1Cat's 37.5 s measured; current 76.4 s).
2. 27K cold prefill TTFT ≤ 13 s (current 25.7 s; 1Cat 16.9 s).
3. ≤8K context prefill no slower than current (we already lead at 6K:
   1,601 vs 1,374 tok/s).
4. Decode throughput unchanged: code256 ≥ 84 tok/s (current 85.7; M1
   in-kernel decode path untouched).
5. No OOM: workspace + staging within the ~11 GiB free headroom/GPU.
6. Mooncake prefix-cache behavior unchanged (nvfp4 KV byte format intact).
7. Greedy determinism: token-identical output vs gate-off run except
   fp-implementation divergence of the replaced attention (precedent:
   A/B token-identity runs show mid-text divergence is expected and
   acceptable when quality stays coherent — same bar as M1).

## 2. Background and evidence

### 2.1 Measured kernel breakdown (nsys 2024.6, 54K prefill, TP2 V100, 200 W power cap)

Our fork (vllm-dev, DFlash2 K=5, nvfp4 KV, chunk 4096), 54K window
(76.2 s wall, 151.1 s total GPU kernel time across both GPUs):

| Bucket | Share | Time | Notes |
|---|---|---|---|
| flashinfer FA2 `BatchPrefillWithPagedKVCacheKernel` | 64.6% | 97.7 s | **all target model**: 416 launches = 16 layers × 13 chunks × 2 GPUs, avg 229 ms/launch. Draft model contributes ≈0 (DFlash2 prefill = KV precompute only, no softmax attention; its attention runs only at decode/verify, sub-ms). |
| cutlass s884gemm GEMM | 20.9% | 31.6 s | target (+ small draft share). Our GEMM is faster than 1Cat's. |
| `gdn_forward_vlk_varlen_kernel` (GDN) | 5.1% | 7.7 s | already 1Cat FlashQLA-SM70 port (`gdn_sm70_fwd.cu`). |
| elementwise | 3.9% | 5.9 s | |
| NCCL allreduce | 2.2% | 3.4 s | TP2. |
| nvfp4 dequant (qpn2) | 1.6% | 2.4 s | |

1Cat 1.5.0 (fp8_e5m2 KV, no spec decode, chunk 8192), 54K window
(37.3 s wall, 74.5 s total GPU time):

| Bucket | Share | Time |
|---|---|---|
| `turbomind::gemm::gemm_kernel` | 65.9% | 49.1 s |
| `flash::sm70_d256_splitd_dense_kernel` (v37 dense) | 16.9% | 12.6 s |
| `flash_attention_forward_paged_d256` | 6.0% | 4.5 s |
| NCCL | 4.0% | 3.0 s |
| `tilelang_fused_chunk_gdr_fwd_kernel` (GDN) | 2.9% | 2.1 s |

Conclusions:

- The entire prefill gap is in full-attention kernels: 97.7 s (ours) vs
  17.1 s (1Cat) for the same attention work → 1Cat's kernel is ~5.7×
  faster. FA2 on sm70 with head_dim 256 is the root cause (D256 is a poor
  fit for FA2 tile layouts on V100).
- GDN is not a bottleneck (5.1%; we already run 1Cat's kernel). GEMM: we
  lead 1Cat (31.6 s vs 49.1 s). Neither needs porting.
- One-time startup artifact (a 52×608 ms PyTorch-MemEffAttention burst at
  trace t=21–37 s) is pre-request (54K request starts at t≈100 s); not in
  steady state; not an M4 target.

### 2.2 The port target: 1Cat `sm70_v37`

Source (1Cat-vLLM tree, verified against the installed 1.5.0 env that ran
the benchmark): `csrc/attention/sm70_v37/` — `prefill.cu` (666 ln),
`tail.cu` (854 ln), `bridge.cu` (335 ln), plus `gemm_with_softmax.h`,
`epilogue_visitor_with_softmax.h`, `reduce_softmax_final.h`,
`register.cpp`, `README.md`. ~1,855 lines of self-contained CUDA (no vLLM
Python/C++ dependency in the kernels).

Contract (from 1Cat's README + our profiling):

- Prefill-only, one request, FP16 I/O, FP32 accumulation.
- Shape per launch: **Hq=6, Hkv=1, D=256** — exactly our per-GPU GQA group
  (HQ=12 / H_kv=2 per GPU ⇒ 6 groups of 6 Q-heads per KV-head; head_dim 256).
- Causal; scale 1/16; Q is 64–8192 in multiples of 64; Q ≤ KV ≤ 262,144;
  KV multiple of 32; 16-byte-aligned contiguous FP16 Q/K/V/out.
- Dataflow: the 6 Q-heads are packed into the GEMM row dimension; the
  preceding KV is a non-causal prefix (M128/N256/K32 sm70 tensor-core GEMM
  with FP32 logits → tile-local probs), the current chunk is an exactly
  masked tail; FP32 online partial state combined before one final FP16
  conversion.
- The explicit E4M3 bridge (1Cat production: fp8 KV → per-layer scale
  expansion) is a **storage-format adapter**; for us the staged KV is
  already FP16 (nvfp4 → fp16 dequant), so the bridge is a no-op path
  (or reads our nvfp4 pages directly — see §4.4).
- Workspace: 768 MiB score cache per device (8192 q × 6 heads × 8192
  prefix columns × 2 B) + transient FP32 slab + 128 MiB headroom; per-device
  lock + CUDA events serialize shared use; admission failure keeps the
  exact fallback.
- CUDA-graph capture is excluded from this operator (eager only) — our
  prefill steps are eager (chunked prefill); decode is unaffected.
- 1Cat's own gates/validation: env `VLLM_FLASH_V100_PREFILL_D256_GQA_V37`,
  route counters, FP64-oracle tests, graph-replay tests.

### 2.3 Why dense-only (no paged-d256 port in phase 1)

v37 consumes the full KV of a request from a dense contiguous FP16
workspace (prefix + tail in one launch). Staging 54K tokens =
54K × 256 × 2 sides × 2 B ≈ 55 MiB/GPU; the config ceiling 262,144 tokens
= 268 MiB/GPU — both well inside headroom. 1Cat additionally used a paged
d256 kernel (6.0% of their time) for their fp8 bridge workspace layout;
with our per-step dense staging that path is unnecessary. The paged
`fused_mha_forward_paged` (D256) port is deferred to phase 2 only if
admission misses prove frequent (§12).

## 3. Scope

In:

- Port of the `sm70_v37` CUDA kernel family into `csrc/libtorch_stable`,
  built via `build_sm70.sh` (mandatory build rule).
- Dense nvfp4→fp16 KV staging op (variant of the existing shadow-dequant
  kernel family; writes a contiguous per-request FP16 KV).
- Prefill policy in `vllm/v1/attention/backends/flashinfer.py`: env gate,
  admission, workspace management, per-request/per-group kernel launch,
  FA2 fallback.
- Tests (unit + offline A/B + server A/B) per §8.

Out (explicit):

- M1 decode, DFlash2, Mooncake, nvfp4 KV byte format, GEMM, GDN: untouched.
- 1Cat's paged-d256 / decode kernels: not ported.
- M3 (FP8 KV compatibility): separate work item.
- 1Cat's `tilelang` GDN variant / turbomind GEMM: not ported (we already
  match or beat them there).
- No changes to `/root/run/run2.sh` or `Qwen3.8-27B.yaml`.

## 4. Architecture

### 4.1 Components

| Component | Location | Change |
|---|---|---|
| v37 kernels | `csrc/libtorch_stable/attention/sm70_v37/` (new dir: prefill.cu, tail.cu, bridge.cu, 3 headers, register.cpp) | port from 1Cat, attribution header (MIT, 1Cat-vLLM) |
| Kernel registration | `csrc/libtorch_stable/` build files + `build_sm70.sh` target | add TUs to `_C_stable_libtorch` |
| Dense staging op | `vllm/v1/attention/ops/sm70_nvfp4_dequant.py` (new dense-output variant, Triton, same LUT/scale logic as the validated paged op) | new |
| Prefill policy | `vllm/v1/attention/backends/flashinfer.py` | new branch in the existing prefill section (where the FA2 paged call currently runs over `shadow_kv`), gated |
| Env gate | `vllm/envs.py` | `VLLM_SM70_FLASH_V100_PREFILL` (default "0" during bring-up; flipped to "1" after §8.3 passes) |

### 4.2 Data flow (per prefill step, per full-attention layer)

```
nvfp4 paged KV (existing pool)
   │  dense staging op (per request, once per step — not per layer):
   │  unpack nibbles + 2-level e4m3 scales → FP16
   ▼
dense FP16 workspace  [kv_len, 2, 256] per (request, kv-head group)   (~55 MiB @54K)
   │  per GQA group (6 groups/layer):
   ▼  Q = layer's 6 Q-heads of this request's prefill tokens (fp16, from the
      in-flight forward — same buffer the FA2 call currently reads)
v37 launch: sm70_d256_gqa_v37_fwd(q, k_dense, v_dense, out, workspace)
   ▼
out  → same destination tensor the FA2 path writes (downstream unchanged)
```

- Per-step staging cost ≈ current shadow dequant (~1.2 s/GPU at 54K) —
  replaced, not added (the paged shadow for prefill becomes unnecessary
  when the gate is on and admission passes; the paged shadow is kept for
  fallback steps and decode).
- Multi-request prefill batches: v37 is batch-1 ⇒ loop over prefill
  requests (each request = up to 6 group launches per layer). Config
  ceiling is 8 seqs × chunk 4096 — loop cost is metadata only.
- Non-causal callers (DFlash draft attention, if it ever runs in prefill
  attention — currently it does not) and q_len=1 decode: excluded by
  admission; they keep the FA2/M1 paths.

### 4.3 Admission & fallback

A request's prefill chunk uses v37 iff ALL hold:

1. Gate on: `VLLM_SM70_FLASH_V100_PREFILL=1` and SM70 family and the stable
   libtorch extension loaded (operator `sm70_d256_gqa_v37_fwd` present).
2. Causal, target-model full-attention layer (not draft, not GDN).
3. q_len ∈ [64, 8192], q_len % 64 == 0 (bridge zero-pads leading rows when
   q_len < 64 per 1Cat's design; q_len < 64 after padding rules ⇒ FA2).
4. kv_len ≤ 262,144, kv_len % 32 == 0.
5. Dense workspace available for (request, kv_len) — otherwise FA2.
6. Head group matches (HQ/HKV = 6/1 per group, D=256, fp16, scale 1/16).

Any miss ⇒ the existing FA2 paged-prefill call for that request (exact
current behavior). A per-layer/per-request route counter (logged at
DEBUG + a summary INFO at step end) records v37 vs fallback — route
counters, not the env var, are the execution evidence (1Cat's rule).

### 4.4 Memory budget (per GPU)

| Item | Size |
|---|---|
| v37 score cache (1Cat spec) | 768 MiB (allocated once, per device, event-serialized) |
| v37 FP32 transient slab | ≤ 128 MiB headroom |
| Dense staging @ 54K | ~55 MiB |
| Dense staging @ 262K ceiling | ~268 MiB |
| Total worst case | ~1.1 GiB |
| Free headroom (measured) | ~11 GiB |

No change to `gpu-memory-utilization` behavior; allocations use the
existing vLLM allocator (torch caching allocator) so they count against
the pool like the current shadow buffer does.

### 4.5 Kernel adaptation notes (port deltas vs 1Cat source)

1. **Bridge**: keep the file structure; the E4M3 bridge path is
   compile-disabled for our build (KV arrives FP16 from staging). The
   bridge stays in-tree behind the same internal flag for reference.
2. **Layout**: 1Cat's production layout is their fp8 bridge workspace;
   ours is plain HND-contiguous FP16 per kv-head group. Map the kernel's
   K/V pointers to `[kv_len, 256]` slices; alignment is 16 B by construction
   (torch allocations).
3. **Naming**: operator keeps 1Cat's `sm70_d256_gqa_v37_fwd` name (their
   README mandates it to prevent old-library confusion) + route counters.
4. **Attribution**: 1Cat-vLLM MIT — keep per-file headers.

## 5. Build

- TUs added to the `libtorch_stable` stable-ABI build; compiled with
  `bash /data/src/vllm/build_sm70.sh _C_stable_libtorch` (mandatory build
  script; no direct cmake/ninja). First run recompiles incrementally
  (~30 s with ccache for the changed TUs).
- Kernel flags: sm_70 gencode only (V100), -O3, `--use_fast_math` NOT set
  (FP32-accumulation contract).
- If the stable-libtorch route hits an ABI issue with the raw pointer
  kernels, fallback is a torch cpp_extension JIT load at first use (same
  pattern as `sm70_gdn.py`) — decide at implementation, prefer stable lib.

## 6. Prefill policy (Python)

New fields on the SM70 section of `FIPrefill` (mirror existing
`shadow_kv` field style): dense workspace ptr per request, kv_len,
admission result. `build()`: for each prefill request compute
(kv_len, q_len), run admission, allocate/round the dense staging region;
`forward()`: per layer — staged KV + Q slice → v37 launch (via the
registered op), or fall through to the existing FA2 call. All on the
existing SM70 NVFP4 KV gate (`VLLM_SM70_NVFP4_KV`) so non-NVFP4 runs stay
byte-identical to upstream.

Gate default: `0` during bring-up. Flip to `1` only after §8.3 passes on
the live server. `VLLM_SM70_FLASH_V100_PREFILL=0` must yield a
byte-identical code path to today (no behavior delta).

## 7. Testing & validation plan

1. **Kernel unit tests** (ported from 1Cat `tests/v1/attention/
   test_sm70_v37_prefill.py` + `tests/kernels/attention/...`, adapted):
   causal attention vs PyTorch FP64 oracle; q/kv edge alignment
   (64/65/8191/8192, kv 32-multiples and misaligned); leading-query
   padding; E4M3 byte coverage replaced by nvfp4 scale coverage (unit and
   non-unit scales, per-group); graph-replay with changing live lengths
   (eager fallback path). Run: `.venv pytest` equivalent
   (`/data/vllm-dev/bin/python -m pytest tests/... -v`).
2. **Offline A/B** (existing harness `/tmp/profile_ours.py`, gate off/on,
   same DFlash2 + nvfp4 config): 6.7K / 27K / 54K TTFT. Expectations:
   54K 76.4 s → ≤ 38 s; 27K 25.7 s → ≤ 13 s; 6.7K 4.3 s → ≤ 4.5 s (no
   regression). Also a second 54K nsys run with the gate on: verify the
   FA2 bucket shrank to ~fallback-only and v37 time ≈ 12–17 s total.
3. **Server A/B** (target config, live server on port 8002):
   - prefill bench (`/tmp/prefill_bench.py`) fresh prompts: 6.7K/27K/54K.
   - decode regression: code256 ≥ 84 tok/s, short256 ≥ 28 tok/s.
   - determinism: 3 fixed greedy prompts, token-identity vs gate-off run
     (bar = M1 precedent: coherent quality, fp-divergence mid-text OK).
   - memory: no OOM at 54K; `nvidia-smi` headroom check before/after.
   - Mooncake: prefix-hit on a repeated 54K prompt still loads KV
     (nvfp4 bytes unchanged); put-path failures remain the pre-existing
     degraded-store issue (not a regression criterion).
4. **Acceptance**: all of §1; then flip gate default to `1` in
   `vllm/envs.py` and update `SM70_PHASEB_STATE.md`.

## 8. Rollback

- Runtime: `VLLM_SM70_FLASH_V100_PREFILL=0` (one env line) — full return
  to today's FA2 prefill; no state migration.
- Code: the policy branch is additive (new fields, new op, new TUs);
  reverting the commit set restores today's tree exactly.
- No rollback decisions without explicit user confirmation (standing
  rule) — a gate-off restart is the first response to any anomaly.

## 9. Risks

| Risk | Mitigation |
|---|---|
| ptxas hang on large 4D-F32 PTX (M2 hard-won lesson) | v37 already compiles and runs on sm70 in 1Cat production; keep per-program live state small as written; if the stable-lib build hangs, drop to JIT-compiled extension with the same source |
| Numerics: fp16 probs + FP32 accum vs FA2 | FP64-oracle unit tests with the same tolerance class 1Cat used; server A/B quality/determinism bar (§7.3) |
| Workspace/allocator surprise under Mooncake+spec decode | §4.4 budget vs 11 GiB headroom; measure at 54K before flipping the default |
| Multi-request prefill loop overhead (up to 8 reqs) | loop is metadata-only; if overhead shows in A/B at 8-req batches, add a batched launch in phase 2 |
| API drift 1.5.0 → 1.4.x | kernels are standalone tensor I/O; the Python launcher is ours; no 1.5.0-only imports in the ported set |
| 200 W power cap distorts future comparisons | keep cap as-is (part of target run2.sh); compare on the cap, always |

## 10. Deliverables (files)

- `csrc/libtorch_stable/attention/sm70_v37/{prefill.cu, tail.cu, bridge.cu, gemm_with_softmax.h, epilogue_visitor_with_softmax.h, reduce_softmax_final.h, register.cpp}`
- `csrc/libtorch_stable/CMakeLists.txt` (+ `build_sm70.sh` if it enumerates TUs)
- `vllm/v1/attention/ops/sm70_nvfp4_dequant.py` (dense staging variant)
- `vllm/v1/attention/backends/flashinfer.py` (policy branch + fields + route counters)
- `vllm/envs.py` (`VLLM_SM70_FLASH_V100_PREFILL`)
- `tests/v1/attention/test_sm70_v37_prefill.py` (ported/adapted)
- `SM70_PHASEB_STATE.md` update at acceptance

## 11. Open items resolved by this spec

- Port scope: sm70_v37 only (paged-d256 deferred) — decided by profile
  data (§2.1) + dense-staging budget (§2.3/§4.4).
- Draft model: no change (prefill contribution ≈ 0; decode via M1/FA2 as
  today).
- FP8 (M3): unaffected, separate queue.

## 12. Phase 2 candidates (only if phase 1 leaves a gap)

- Port `fused_mha_forward_paged` (D256) for prefix attention when dense
  staging admission fails (long-context, many concurrent 262K requests).
- Batched v37 launch for multi-request prefill.
- Route decode-side non-causal DFlash attention onto a tuned path (only
  if verify-step latency becomes visible — currently sub-ms).
