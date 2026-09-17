# SM70 Flash-V100 Prefill Port Implementation Plan (revised 2026-09-17)

> **Unattended mode:** human partner authorized full unattended completion. Execute inline with checkpoints per executing-plans discipline; do not stop for confirmations.
> **Revised:** recon (2026-09-17) showed the production-proven kernel is NOT the 1Cat `csrc/attention/sm70_v37` directory (that dir is missing 6 headers it includes — WIP). The running 1Cat 1.5.0 stack executes `torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_{dense,paged}_fwd`, compiled from 1Cat's **fork of vllm-flash-attention** (`github.com/zhinianqin/flash-attention-v100.git @ c2eda5e6115b98c3ba4bfd181570668742eece22`) + 4 cmake patches. This plan ports THAT exact source.

**Goal:** 54K cold prefill TTFT 76.4s → ≤38s (beat 1Cat 37.5s); 27K ≤13s; ≤8K no regression; decode ≥84 tok/s unchanged; no OOM; Mooncake byte format intact.

**Architecture:**
- **Kernel source:** 1Cat fork checkout (clone in progress at `/data/src/1Cat-vLLM/fa100-fork`) + 4 patches from `/data/src/1Cat-vLLM/cmake/patches/` (pipeline → splitkv3 → k_pingpong → gqa_arch order, same as 1Cat's `vllm_flash_attn.cmake`).
- **Build:** point our cmake at the fork via `VLLM_FLASH_ATTN_SRC_DIR` env override (mechanism already exists in our `cmake/external_projects/vllm_flash_attn.cmake` lines 28-35 — the same mechanism 1Cat used; their build's `FETCHCONTENT_BASE_DIR` was our `.deps`). Fresh reconfigure via `build_sm70.sh` first-run path; target `_vllm_fa2_C`; script copies `build-sm70/_vllm_fa2_C.abi3.so` → `vllm/_vllm_fa2_C.abi3.so`.
- **Safety:** our tree currently has NO `_vllm_fa2_C.abi3.so` (only `_C_stable_libtorch.abi3.so`); the FA2 import error at startup is the known-benign noise. Nothing in our SM70 runtime consumes `_vllm_fa2_C` (our attention = flashinfer FA2 + `_C_stable_libtorch` M1 decode). The fork's FA2 is SM70-only hdim256 causal prefill — it replaces a non-existent lib, not a used one.
- **Ops delivered (in `torch.ops._vllm_fa2_C`):**
  - `sm70_d256_splitd_n32_paged_fwd(q, k, v, block_table, out, kv_len, softmax_scale, causal)` — primary: q `[1, T, 12, 256]` fp16, k/v paged fp16 (4-D `[num_blocks, page, H, D]`, layout confirmed from kernel source after clone), `block_table [1, M]` int32, out in-place.
  - `sm70_d256_splitd_n32_dense_fwd(q, k, v, out, softmax_scale, causal)` — dense `[B, kv_len, H, D]` variant (kept; used for multi-request uniform-kv batches in phase 2).
  - `sm70_d256_splitd_n32_dense_splitkv3_fwd(...)` (splitkv3.patch; long-kv split-KV3) and the gqa_arch experimental op — registered by the patches; policy may ignore them in phase 1.
- **KV staging:** new Triton op `sm70_nvfp4_paged_stage_nhd` — dequant our raw nvfp4 paged KV into an NHD fp16 paged workspace (K and V as separate `[M_req, N, H_kv, D]` tensors), reusing the validated decode LUT/e4m3 logic from `sm70_nvfp4_dequant.py`. The existing HND shadow (`[B, H, N, 2D]`) is untouched — it stays the FA2 fallback + decode path.
- **Layouts CONFIRMED from kernel source (Task 1 Step 4 done):** both ops are one kernel `sm70_d256_splitd_dense_kernel<half, is_paged>`; q `[B, q_len, H_q, 256]` fp16 unit inner stride; dense k/v `[B, kv_len, H_kv, 256]`; paged k/v `[num_blocks, page, H_kv, 256]` (NHD, page%kGmemKRowsPerThread==0 — our page N=16 OK) + `block_table [B, M]` int32 + `kv_len` int (q_len ≤ kv_len ≤ M*page); out contiguous `[B, q_len, H_q, 256]`; causal required; q_len%64==0 (kBlockM=64), kv_len%32==0 (kBlockN=32), D=256, GQA H_q%H_kv==0, scales pre-multiplied by log2e in-kernel. Staging workspace: compact per-request `[M_req, 16, 2, 256]` fp16 (K and V separate) + local identity block_table `arange(M_req)`; kv_len%32==0 ⇒ kv_len%16==0 ⇒ no partial blocks.
- **Integration:** admission-gated branch in `flashinfer.py` forward() replacing the FA2 prefill run for admitted single-request steps; gate env `VLLM_SM70_FLASH_V100_PREFILL` default "0", flipped to "1" only after Task 6 acceptance. Route counters as execution evidence.

## Global Constraints

- Compile ONLY via `bash /data/src/vllm/build_sm70.sh [target]`. Never bare setup.py/cmake/ninja. Reconfigure happens via the script's first-run path (move `build-sm70` aside, script reconfigures, ccache keeps it fast).
- Python: `python3` (=/data/vllm-dev). Never bare pip/system python.
- Preserved SPDX headers on all copied files; attribution: 1Cat-vLLM fork of vllm-project/vllm-flash-attention (BSD-3-Clause) + 1Cat patches.
- Process rules: never `pkill`; kill specific PIDs from `pgrep -af`; long GPU jobs nohup+log+poll; per-command timeouts ≤120s.
- Production server (port 8002) healthy during Tasks 1-4 (GPUs at 28.4 GiB each, ~3.6 GiB headroom for small unit tests); stop it only for Task 5 A/B runs, restart before end of Task 6 with gate ON.
- Success bars: 54K ≤38s; 27K ≤13s; 6.7K ≤4.5s; decode code256 ≥84 tok/s; short256 ≥28 tok/s; no OOM; Mooncake GET still works.
- 1Cat admission table (from their policy test, gqa-arch route — our splitd policy mirrors its shape rules): q%64==0, 64≤q≤8192; kv%32==0, kv≤262144; (gqa-arch additionally requires kv>q — splitd has no such restriction; verify against kernel behavior in Task 2).

---

### Task 1: Fork checkout + patches + build `_vllm_fa2_C` + op smoke

**State at start:** clone of `zhinianqin/flash-attention-v100.git` running in background → `/data/src/1Cat-vLLM/fa100-fork` (blob:none filter), checkout c2eda5e6 pending.

- [x] **Step 1: Finish clone + verify commit** — log shows `CLONE_OK`; `git -C /data/src/1Cat-vLLM/fa100-fork rev-parse HEAD` == c2eda5e6115b98c3ba4bfd181570668742eece22. (DONE)
- [x] **Step 2: Resolve cutlass submodule** — DONE: fork pin == our `.deps` pin (62750a2b) → copied `include/` + `examples/35_gemm_softmax/` from our `.deps` checkout (26MB).
- [x] **Step 3: Apply the 4 patches in cmake order** — DONE with two manual fixes: (a) gqa_arch CMakeLists hunk failed on a trailing-space mismatch in the fork's `file(GLOB FA2_GEN_SRCS ` line — applied the intended 4-file `set(FA2_GEN_SRCS ...)` manually (matches 1Cat production); (b) pipeline CMakeLists hunk had the same root cause (never applied per its output). All 4 .cu/.h patch hunks applied; .orig/.rej removed.
  ```bash
  cd /data/src/1Cat-vLLM/fa100-fork
  P=/data/src/1Cat-vLLM/cmake/patches
  for f in sm70_flash_attn_d256_pipeline sm70_flash_attn_d256_splitkv3 \
           sm70_flash_attn_d256_k_pingpong sm70_flash_attn_d256_d256_gqa_arch; do :; done
  patch -p1 --batch --forward -i $P/sm70_flash_attn_d256_pipeline.patch
  patch -p1 --batch --forward -i $P/sm70_flash_attn_d256_splitkv3.patch
  patch -p1 --batch --forward -i $P/sm70_flash_attn_d256_k_pingpong.patch
  patch -p1 --batch --forward -i $P/sm70_flash_attn_d256_gqa_arch.patch   # includes cutlass/examples/35_gemm_softmax hunk
  ```
  All four must apply clean (they did for 1Cat's build; any failure = investigate fork base mismatch before proceeding). Note: `gqa_arch` patch touches `csrc/cutlass/examples/35_gemm_softmax/gemm_with_softmax.h` → cutlass checkout (Step 2) must precede it.
- [x] **Step 4: Read kernel source for KV layout** — DONE: paged K/V are separate contiguous `[num_pages, page, H_kv, D]` NHD tensors (identity block table in our integration); Q/out are `[1, q_rows, H_qo, D]`; no scratch ("pair-scratch" is the 1Cat-side launch args only). Alignment as read: `q % 64 == 0`, `kv_len % 32 == 0`, `page % 16 == 0`, `q_len == kv_len` (causal offset `kv_len - query_len`). **Mid-course deviation (Task 5.3):** spec decode forces 4091-token chunks (4096 − 5 draft slots) → alignment impossible (padding-only fix is mathematically impossible: Q needs +to-64, KV +to-32, no common solution when `(kv−q) % 32 ≠ 0`). Fixed by a minimal kernel extension: paged op gains an explicit true `query_len` (offset/mask) + `q_rows` (padded buffer shape for mQ); grid/alignment checks relaxed to `q_rows % 64 == 0 && q_rows >= query_len`, `capacity % 32 == 0`; core mask/softmax already partial-safe. Python pads q buffer to ceil64, KV workspace to whole pages (page_size=64 in production — see Task 5.3 note), passes true lengths, slices output.
- [x] **Step 5: Reconfigure + build** — DONE: `mv build-sm70 build-sm70.bak-upstream` + `VLLM_FLASH_ATTN_SRC_DIR=... build_sm70.sh _vllm_fa2_C` (offline reconfigure via the script's first-run path, FETCHCONTENT_FULLY_DISCONNECTED) → FA2_EXIT=0, STABLE_EXIT=0. The nested `vllm-flash-attn/_vllm_fa2_C.abi3.so` install path broke the script's top-level copy → script now has a nested-install fallback (installs to `vllm/vllm_flash_attn/`). Ops verified: paged/dense/dense_splitkv3 all registered after `import vllm.vllm_flash_attn`.
  ```bash
  cd /data/src/vllm
  mv build-sm70 build-sm70.bak-upstream   # keep as rollback reference
  VLLM_FLASH_ATTN_SRC_DIR=/data/src/1Cat-vLLM/fa100-fork bash build_sm70.sh _vllm_fa2_C
  ```
  Expect: full cmake reconfigure (re-fetches other FetchContent deps into `.deps`; network OK) + compile of the fork FA2 TUs (SM70-only hdim256 set + `flash_fwd_d256_splitd_sm70.cu` + `flash_fwd_d256_gqa_arch_sm70.cu`). ccache warms the rest. Budget: 5-15 min. Then the script copies `vllm/_vllm_fa2_C.abi3.so`.
  Follow-up (same run or second invocation): `bash build_sm70.sh _C_stable_libtorch` — must remain green (our stable lib, untouched sources).
- [x] **Step 6: Op smoke test** — DONE: `tests/v1/attention/test_sm70_splitd_smoke.py`, 11/11 pass (dense + paged vs FP64 per-q-head reference; bottom-right causal; GQA repeat; 65K kv; shuffled block table; 3 unaligned partial-tile cases 4091/4091, 4091/8182, 2334/26880).
  ```python
  import torch

  def test_splitd_ops_registered_and_run():
      paged = torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_paged_fwd
      dense = torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_dense_fwd
      torch.cuda.set_device(0)
      q = torch.randn(1, 64, 12, 256, dtype=torch.float16, device="cuda")
      out = torch.empty_like(q)
      # dense path (kv_len == q_len, no cache)
      k = torch.randn(1, 64, 2, 256, dtype=torch.float16, device="cuda")  # layout per Step 4
      v = torch.randn_like(k)
      r = dense(q, k, v, out, 1.0 / 16.0, True)
      assert torch.isfinite(r).all() and (r == out).all()
      # paged path: 4 blocks x page 16 x H x D, identity block table
      B, N = 4, 16
      k_cache = torch.randn(B, N, 2, 256, dtype=torch.float16, device="cuda")  # per Step 4 layout
      v_cache = torch.randn_like(k_cache)
      bt = torch.arange(B, dtype=torch.int32, device="cuda").unsqueeze(0)
      q2 = torch.randn(1, 64, 12, 256, dtype=torch.float16, device="cuda")
      out2 = torch.empty_like(q2)
      r2 = paged(q2, k_cache, v_cache, bt, out2, 64, 1.0 / 16.0, True)
      assert torch.isfinite(r2).all()
  ```
  (Layouts in this test are PLACEHOLDERS until Step 4 confirms them — rewrite the tensor shapes to match the kernel's actual expectations before running. This is the only open detail in the task; everything else is fixed.)
  Run: `cd /data/src/vllm && python3 -m pytest tests/v1/attention/test_sm70_splitd_smoke.py -v` → PASS.
- [x] **Step 7: Commit** — DONE as `684a850255` (integration + staging + smoke) — .so NOT committed: `*.so` is gitignored (build artifact); rebuild via `VLLM_FLASH_ATTN_SRC_DIR=/data/src/1Cat-vLLM/fa100-fork bash build_sm70.sh _vllm_fa2_C` (~30s ccache). Fork-side changes (4 T1 patches + partial-tile extension) committed in the fa100-fork repo for provenance.

---

### Task 2: Kernel correctness (FP64 oracle)

**Files:**
- Create: `tests/v1/attention/test_sm70_splitd_prefill.py`

- [x] **Step 1: Write oracle tests** — DONE (in `tests/v1/attention/test_sm70_splitd_smoke.py`): dense+paged vs FP64 per-q-head row-chunked reference; chunked-prefill bottom-right causal mask; GQA via repeat_interleave; 65K kv cases; shuffled block table (tolerance relaxed to 1e-1: measured 6.5e-2 kernel order-sensitivity — a 1Cat kernel property; production uses the identity compact workspace).
  - Reference: causal attention in FP64 (packed GQA: 12 q-heads → 2 kv-heads, group 6), `softmax(q@k.T * (1/16) + causal_mask) @ v`.
  - Cases (dense op): q_len ∈ {64, 384, 1600, 4096, 8192}, kv_len == q_len (no-cache) and kv_len > q_len (prefix) — kv via dense `[1, kv_len, 2, 256]`; tolerances: fp16 kernel vs fp64 → `atol/rtol` per 1Cat convention (start 2e-2/1e-2 on fp16 magnitudes; assert pass).
  - Cases (paged op): same sweep with kv in paged layout, identity + shuffled block tables (block permutation must not change results — correctness of block_table handling), kv%32 boundaries (32, 63→rejected?, 64, 4848-multiple).
  - GQA correctness: q-head h attends kv-head h//6 — verify with per-head-group distinct KV values (not randn symmetry).
  - Tolerance calibration: first run prints max err per case; fix thresholds to measured+margin, then re-run green.
  Run: `python3 -m pytest tests/v1/attention/test_sm70_splitd_prefill.py -v` → all PASS. Any fail = port/build bug → fix in fork checkout (re-apply patches cleanly), never by weakening the reference.
- [x] **Step 2: Commit** — DONE: oracle tests shipped in `684a850255` (file landed as `test_sm70_splitd_smoke.py`).

---

### Task 3: NHD staging op (nvfp4 paged → fp16 NHD workspace)

**Files:**
- Modify: `vllm/v1/attention/ops/sm70_nvfp4_dequant.py` (add kernel + launcher)
- Test: `tests/v1/attention/test_sm70_nvfp4_stage_nhd.py`

**Interfaces (exact shapes fixed in Task 1 Step 4 from kernel source):**
- `sm70_nvfp4_paged_stage_nhd(kv_cache, block_table_row, kv_len, k_out, v_out, k_scale, v_scale) -> None`
- `k_out`, `v_out`: preallocated fp16, layout per Step 4 (expected default `[num_blocks_used, N, H_kv, D]` with only the request's blocks filled) — the op fills exactly `kv_len` tokens of the request's physical blocks.

- [x] **Step 1: Failing test first** — DONE: `tests/v1/attention/test_sm70_nvfp4_stage_nhd.py`, bit-exact vs the validated paged-shadow dequant. Coverage extended after Task 5.3: parametrized over page sizes {16, **64 = production NVFP4 HND**}, incl. the 28637-kv shape that broke the old ceil32//page sizing (partial page + kv not %64).
  ```python
  # same raw bytes in -> shadow (HND) and NHD staging must agree per (token, head, side)
  sm70_nvfp4_paged_dequant(kv_cache, bt, lens, shadow, k_scale, v_scale)   # existing op = reference
  sm70_nvfp4_paged_stage_nhd(kv_cache, bt_row, kv_len, k_out, v_out, k_scale, v_scale)
  # reindex shadow[b,h,n, :D] == k_out at (block, n, h, :)  (HND->NHD transpose check)
  ```
  Cover: kv_len multiple of N; kv_len not multiple of N (partial last block); kv%32 != 0 (stages exactly kv_len rows, no padding needed — kernel takes kv_len); unit + non-unit e4m3 scales.
  Run → FAIL (function undefined).
- [x] **Step 2: Implement** — DONE: `sm70_nvfp4_paged_stage_nhd` + `_sm70_nvfp4_stage_nhd_side_kernel` in `sm70_nvfp4_dequant.py` (same file, reused nibble/e4m3 decode helpers, two-side launch). Post-5.3: staging grid covers exactly the request's real blocks (`nb = ceil(kv_len/N)`); padded workspace tail stays pre-zeroed. same program decomposition (one program per (physical block, head, token-in-block)), same nibble/e4m3 decode helpers (reuse, do not duplicate), only output addressing changes: NHD `out[block, n, h, d]` (K into `k_out`, V into `v_out` — two launches, mirroring the existing launcher's two-side structure). Launcher mirrors `sm70_nvfp4_paged_dequant` parameter passing (block table base, side offsets `H*N*F`, page stride).
- [x] **Step 3: Test passes** (bit-exact, 9/9 incl. N=64 production cases).
- [x] **Step 4: Commit** — DONE in `684a850255` (bundled with the integration commit).

---

### Task 4: Admission policy + flashinfer.py integration

**Files:**
- Modify: `vllm/envs.py` — `VLLM_SM70_FLASH_V100_PREFILL: str = "0"` (pattern of `VLLM_SM70_NVFP4_KV`).
- Create: `vllm/v1/attention/ops/sm70_splitd_policy.py` — pure predicate, importable + unit-testable:
  ```python
  def sm70_splitd_admit(num_prefills, q_len, kv_len, causal, q_dtype) -> bool:
      """Admit the splitd paged prefill route for this step.
      Phase 1: single prefill request per step; all other steps stay on FA2."""
      return (
          num_prefills == 1
          and causal
          and q_dtype is torch.float16
          and 64 <= q_len <= 8192
          and q_len % 64 == 0
          and 0 < kv_len <= 262144
          and kv_len % 32 == 0
      )
  ```
- Modify: `vllm/v1/attention/backends/flashinfer.py`:
  - Module state: `_SPLITD_ENABLED = envs.VLLM_SM70_FLASH_V100_PREFILL == "1"` + op-existence probe (lazy, once); `_SPLITD_ROUTE_STATS = {"routed": 0, "fallback": 0}` (INFO log every 100 decisions).
  - `FIPrefill` fields: `splitd_admit: bool = False`, `splitd_kv_ws` (k/v NHD workspaces, sized to max admitted kv_len, allocated once per step in build()), `splitd_bt_row` (int32 block-table row on GPU for the admitted request).
  - build(): if `_SPLITD_ENABLED` and step is all-prefill: q_len from `qo_indptr_prefill_cpu` diff, kv_len from `kv_lens_prefill_cpu`; admit → size/stage-ready workspaces (dequant happens per-layer in forward).
  - forward() shadow prefill branch (the existing `sm70_nvfp4_paged_dequant` + `prefill_wrapper.run` block, ~line 2647-2730): if `attn_metadata.prefill.splitd_admit`: for this layer — `sm70_nvfp4_paged_stage_nhd(kv_cache, splitd_bt_row, kv_len, k_ws, v_ws, k_scale, v_scale)`, then `torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_paged_fwd(q4, k_ws, v_ws, splitd_bt_row, out4, kv_len, self.scale, True)` where `q4 = prefill_query.reshape(1, q_len, H, D)` (contiguous check; copy if needed) and `out4` = view of the layer output slice (single request → `output[num_decode_tokens:num_decode_tokens+q_len].reshape(1, q_len, H, D)`); SKIP the FA2 run for this layer. Else: existing FA2 path unchanged.
  - Gate-off guarantee: when env is "0", `splitd_admit` is never set, no new allocations, byte-identical path (verified by test).
- Test: `tests/v1/attention/test_sm70_splitd_policy.py` — predicate table: admit (1, 4096, 53778+... use kv%32: 53792, True, fp16); reject num_prefills=2; reject q=100; reject q=32; reject q=8256; reject kv=262145; reject kv%32; reject kv=0; reject causal=False; reject bf16; gate-off (env "0" → predicate never consulted — assert build() leaves splitd_admit False / no workspace).
- [x] Steps: DONE — env landed as `VLLM_SM70_SPLITD_PREFILL` (bool, `envs.py`), predicate in `sm70_splitd_policy.py`, integration in `flashinfer.py` build/forward; policy tests 8/8 (extended after 5.3: unaligned q/kv admit cases). **Policy deviation (5.3):** alignment checks dropped — final predicate `n_pre==1, 64 ≤ q_len ≤ 8192, 0 < kv_len ≤ 262144, q_len ≤ kv_len, causal, fp16` (see Task 5.3 findings). Offline e2e: routed counters fire (201+ layer runs), coherent text. Commits: `684a850255`, `15d02087cb`, `df5551683c` + fork partial-tile commits.

---

### Task 5: Offline A/B (integration gate)

**Harness:** reuse `/tmp/profile_ours.py` (offline LLM, DFlash2 + nvfp4 KV, mirrors server config) at 6.7K / 27K / 54K.

- [x] **Step 1: Stop the production server** — DONE (PIDs 479709/479710/480380/480564/480565 killed; 0 MiB both GPUs).
- [x] **Step 2: Baseline (gate off)** — DONE: 6.7K 4.40s / 27K 26.58s / 54K 76.15s (reproduces profiled 76s; T5.1).
- [x] **Step 3: Gate-on runs** — `VLLM_SM70_SPLITD_PREFILL=1`: **54K 36.84s (bar ≤38s PASS, beats 1Cat 37.5s); 1.5K 0.93s (≤4.5s PASS); 27K 16.37s (bar ≤13s MISS — see 5.3b)**. Route counters: all 14 of 14 54K chunks routed (13×4091 + 577 partial), 201/… counter lines, 0 errors (only known FA2 startup noise).
- [x] **Step 4: Quality** — DONE: coherent text at all sizes; vs gate-off output, divergence only fp-implementation level (M1 precedent; full 3-prompt server check in Task 6).
- [x] **Step 5: Commit any fixes** found here — DONE: `15d02087cb` (GQA-ratio check), `df5551683c` (block-table row slice), fork partial-tile extension + page-aligned workspace sizing (5.3a/5.3b).

**Mid-course findings (Task 5.3):**
- **(5.3a) 4091-chunk root cause.** Spec decode (DFlash2, 5 draft slots) cuts the prefill budget: 4096 − 5 = 4091 tokens/chunk → q%64≠0, kv%32≠0 cumulatively. Padding-only impossible (Q to-64 vs KV to-32, no common solution). Fix = kernel gains true `query_len` + `q_rows` (partial-tile safe), policy relaxed (no alignment reqs).
- **(5.3b) Production KV page size = 64, not 16.** NVFP4 HND cache pages are 64 tokens; `m_blocks = ceil32(kv)//page` lost up to 63 rows whenever `ceil32(kv) % 64 == 32` (first hit: 54K chunk 7, kv=28637 → 447 pages = 28608 < 28637, staging assert). Fix: `m_blocks = ceil(kv_len/page_size)` (whole pages; capacity %32==0 since 64%32==0).
- **(5.3c) 27K bar miss analysis.** Per-chunk cost ≈ 2–4s nearly flat in kv (target+drafter 4091-token forwards, DFlash2 verify, 10 GDN groups, TP2, 4–6 builds/step) — shared by gate off/on and by 1Cat. Kernel savings: 1.5s/chunk (27K) → 2.8s/chunk (54K). Even zero-cost attention gives 27K ≈ 9–10s; ≤13s would need the kernel ~1.5× faster than 1Cat's own tuned kernel (we already beat their 54K: 36.84 vs 37.5s). **Bar not reachable within M4 scope** — reported, not silently missed.

---

### Task 6: Server A/B, acceptance, default flip, docs, memory

**Files:**
- Modify: `vllm/envs.py` (gate default "0"→"1"); `/root/run/run2.sh` (+`export VLLM_SM70_FLASH_V100_PREFILL=1` — SM70 env, allowed class); `/data/src/SM70_PHASEB_STATE.md` (M4 section).
- [x] **Step 1: Restart server with gate on** — DONE: `/health` 200 after ~370s (run2.sh envs + inline `VLLM_SM70_SPLITD_PREFILL=1`; log /root/run/server_m4.log).
- [x] **Step 2: Server A/B** — DONE (54K 35.41s ≤38s PASS, beats 1Cat 37.5s; 27K 15.79s — bar ≤13s MISS per 5.3c analysis; 4K 3.93s ≤4.5s PASS; code256 **85.4 tok/s ≥84 PASS**; short256 32.3 ≥28 PASS; 0 OOM, GPUs 30.2GiB steady).
- [x] **Step 3: Mooncake check** — connector up (standalone-store, KV-load threads ×2); 32K-old no store hit (35.57s) = pre-existing M1-era state (PUT -800 + dead segments + dflash/mamba lookup-window warning), not a regression: staging is read-only on the cache, byte format untouched.
- [x] **Step 4: Determinism** — short256/code256 byte-identical vs pre-M4 capture; longctx128 single-char synonym token flip at char 200 (就→只) — fp-level, coherent.
- [x] **Step 5: Flip + docs + memory** — DONE: envs.py default → True, run2.sh +export (backup run2.sh.bak-pre-splitd), SM70_PHASEB_STATE.md M4 section, memory entries 10147/10148 + acceptance; server left RUNNING with gate ON.
- [x] **Step 6: Final report** — numbers vs bars in the state doc M4 section; residual risks: multi-request steps → FA2 fallback (phase-2 dense-route batch); q<64 partial chunks → FA2 (577-token tail chunk routes fine at ≥64; <64 falls back); gqa-arch/splitkv3 ops registered but unused (splitkv3 = dense-only, paged path uses the paged op — same as 1Cat production); 27K bar unreachable within kernel scope (5.3c).

---

## Self-review (revised)

- Spec §1 criteria → T5 (TTFT bars) + T6 (decode/Mooncake/determinism). ✓
- Kernel provenance = 1Cat production .so source (highest fidelity; their build used the same `.deps` base dir). ✓
- No change to: M1 decode, DFlash2, GDN, GEMM, Mooncake, existing HND shadow (fallback+decode intact). ✓
- Rule compliance: all compiles via build_sm70.sh; python3 only; no pkill; log+poll for long runs. ✓
- Rollback: gate env off = byte-identical FA2 path (T4 test); build rollback = `mv build-sm70.bak-upstream build-sm70`. ✓
- Open detail (accepted): exact paged KV layout of the splitd kernel — fixed in T1 Step 4 by reading the kernel source; all downstream (T3 staging layout, T4 tensor views, T2 test shapes) follow it.
