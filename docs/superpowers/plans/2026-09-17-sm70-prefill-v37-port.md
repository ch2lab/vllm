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

- [ ] **Step 1: Finish clone + verify commit** — log shows `CLONE_OK`; `git -C /data/src/1Cat-vLLM/fa100-fork rev-parse HEAD` == c2eda5e6115b98c3ba4bfd181570668742eece22.
- [ ] **Step 2: Resolve cutlass submodule** — `git -C fa100-fork ls-tree HEAD csrc/cutlass` → pin P. If P != our `.deps` pin (62750a2b...), `git -C fa100-fork submodule update --init csrc/cutlass` (network; blob-less clone still fetches submodule data on demand). If equal, `cp -a` or symlink our `.deps/vllm-flash-attn-src/csrc/cutlass` into the fork (record which).
- [ ] **Step 3: Apply the 4 patches in cmake order**
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
- [ ] **Step 4: Read kernel source for KV layout** — in `csrc/flash_attn/src/flash_fwd_d256_splitd_sm70.cu` find the paged KV pointer arithmetic (`sm70_d256_splitd_paged_fwd` impl in `flash_api.cpp` + kernel launch): record exact expected layout (k/v `[num_blocks, page, H, D]` strides, HND or NHD, K/V separation, page-size constraints) and the Q layout/stride assumptions. This fixes the staging op output layout (Task 3) and the policy constraints. Also record scratch/workspace allocations inside the op (1Cat's "pair-scratch").
- [ ] **Step 5: Reconfigure + build**
  ```bash
  cd /data/src/vllm
  mv build-sm70 build-sm70.bak-upstream   # keep as rollback reference
  VLLM_FLASH_ATTN_SRC_DIR=/data/src/1Cat-vLLM/fa100-fork bash build_sm70.sh _vllm_fa2_C
  ```
  Expect: full cmake reconfigure (re-fetches other FetchContent deps into `.deps`; network OK) + compile of the fork FA2 TUs (SM70-only hdim256 set + `flash_fwd_d256_splitd_sm70.cu` + `flash_fwd_d256_gqa_arch_sm70.cu`). ccache warms the rest. Budget: 5-15 min. Then the script copies `vllm/_vllm_fa2_C.abi3.so`.
  Follow-up (same run or second invocation): `bash build_sm70.sh _C_stable_libtorch` — must remain green (our stable lib, untouched sources).
- [ ] **Step 6: Op smoke test** — create `tests/v1/attention/test_sm70_splitd_smoke.py`:
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
- [ ] **Step 7: Commit**
  ```bash
  cd /data/src/vllm && git add vllm/_vllm_fa2_C.abi3.so tests/v1/attention/test_sm70_splitd_smoke.py
  git commit -m "sm70: build _vllm_fa2_C from 1Cat flash-attention-v100 fork + SM70 d256 splitd patches (splitd/dense + splitkv3 + gqa-arch ops)"
  ```
  (The .so is a build artifact; committing it keeps the sm70 branch self-contained on this machine. CMakeLists unchanged — the fork comes in via the VLLM_FLASH_ATTN_SRC_DIR env override recorded in this commit message + the plan doc.)

---

### Task 2: Kernel correctness (FP64 oracle)

**Files:**
- Create: `tests/v1/attention/test_sm70_splitd_prefill.py`

- [ ] **Step 1: Write oracle tests**
  - Reference: causal attention in FP64 (packed GQA: 12 q-heads → 2 kv-heads, group 6), `softmax(q@k.T * (1/16) + causal_mask) @ v`.
  - Cases (dense op): q_len ∈ {64, 384, 1600, 4096, 8192}, kv_len == q_len (no-cache) and kv_len > q_len (prefix) — kv via dense `[1, kv_len, 2, 256]`; tolerances: fp16 kernel vs fp64 → `atol/rtol` per 1Cat convention (start 2e-2/1e-2 on fp16 magnitudes; assert pass).
  - Cases (paged op): same sweep with kv in paged layout, identity + shuffled block tables (block permutation must not change results — correctness of block_table handling), kv%32 boundaries (32, 63→rejected?, 64, 4848-multiple).
  - GQA correctness: q-head h attends kv-head h//6 — verify with per-head-group distinct KV values (not randn symmetry).
  - Tolerance calibration: first run prints max err per case; fix thresholds to measured+margin, then re-run green.
  Run: `python3 -m pytest tests/v1/attention/test_sm70_splitd_prefill.py -v` → all PASS. Any fail = port/build bug → fix in fork checkout (re-apply patches cleanly), never by weakening the reference.
- [ ] **Step 2: Commit** — `git add tests/v1/attention/test_sm70_splitd_prefill.py && git commit -m "sm70: splitd prefill kernel FP64-oracle correctness tests (dense + paged)"`

---

### Task 3: NHD staging op (nvfp4 paged → fp16 NHD workspace)

**Files:**
- Modify: `vllm/v1/attention/ops/sm70_nvfp4_dequant.py` (add kernel + launcher)
- Test: `tests/v1/attention/test_sm70_nvfp4_stage_nhd.py`

**Interfaces (exact shapes fixed in Task 1 Step 4 from kernel source):**
- `sm70_nvfp4_paged_stage_nhd(kv_cache, block_table_row, kv_len, k_out, v_out, k_scale, v_scale) -> None`
- `k_out`, `v_out`: preallocated fp16, layout per Step 4 (expected default `[num_blocks_used, N, H_kv, D]` with only the request's blocks filled) — the op fills exactly `kv_len` tokens of the request's physical blocks.

- [ ] **Step 1: Failing test first** — bit-exact vs the validated paged-shadow dequant:
  ```python
  # same raw bytes in -> shadow (HND) and NHD staging must agree per (token, head, side)
  sm70_nvfp4_paged_dequant(kv_cache, bt, lens, shadow, k_scale, v_scale)   # existing op = reference
  sm70_nvfp4_paged_stage_nhd(kv_cache, bt_row, kv_len, k_out, v_out, k_scale, v_scale)
  # reindex shadow[b,h,n, :D] == k_out at (block, n, h, :)  (HND->NHD transpose check)
  ```
  Cover: kv_len multiple of N; kv_len not multiple of N (partial last block); kv%32 != 0 (stages exactly kv_len rows, no padding needed — kernel takes kv_len); unit + non-unit e4m3 scales.
  Run → FAIL (function undefined).
- [ ] **Step 2: Implement** — new Triton kernel alongside `_sm70_nvfp4_paged_dequant_side_kernel` in the same file: same program decomposition (one program per (physical block, head, token-in-block)), same nibble/e4m3 decode helpers (reuse, do not duplicate), only output addressing changes: NHD `out[block, n, h, d]` (K into `k_out`, V into `v_out` — two launches, mirroring the existing launcher's two-side structure). Launcher mirrors `sm70_nvfp4_paged_dequant` parameter passing (block table base, side offsets `H*N*F`, page stride).
- [ ] **Step 3: Test passes** (bit-exact).
- [ ] **Step 4: Commit** — `git commit -m "sm70: NHD nvfp4->fp16 paged staging for splitd prefill kernel"`

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
- [ ] Steps: failing policy tests → implement predicate + env + integration → `python3 -m pytest tests/v1/attention/test_sm70_splitd_policy.py -v` PASS → minimal offline end-to-end (single 16K prompt, gate on, route counter shows routed>0, coherent text) → commit.

---

### Task 5: Offline A/B (integration gate)

**Harness:** reuse `/tmp/profile_ours.py` (offline LLM, DFlash2 + nvfp4 KV, mirrors server config) at 6.7K / 27K / 54K.

- [ ] **Step 1: Stop the production server** (GPUs needed free-ish): `pgrep -af "vllm serve|VLLM::"` → kill specific PIDs (APIServer → workers → EngineCore → tracker) → `nvidia-smi` shows 0 MiB on both GPUs.
- [ ] **Step 2: Baseline (gate off)** — confirm 54K ≈ 76s (reproduces profiled baseline).
- [ ] **Step 3: Gate-on runs** — `VLLM_SM70_FLASH_V100_PREFILL=1`; 54K ×2 (stability), 27K, 6.7K. Check route counter INFO lines show full v37 coverage on long-context layers; bars: 54K ≤38s, 27K ≤13s, 6.7K ≤4.5s. Miss → nsys the gate-on run (harness in /tmp), fix dominant cause, re-run.
- [ ] **Step 4: Quality** — 3 greedy 128-token prompts gate on vs off: coherent; divergence only fp-implementation level (M1 precedent).
- [ ] **Step 5: Commit any fixes** found here.

---

### Task 6: Server A/B, acceptance, default flip, docs, memory

**Files:**
- Modify: `vllm/envs.py` (gate default "0"→"1"); `/root/run/run2.sh` (+`export VLLM_SM70_FLASH_V100_PREFILL=1` — SM70 env, allowed class); `/data/src/SM70_PHASEB_STATE.md` (M4 section).
- [ ] **Step 1: Restart server with gate on** (run2.sh envs, from /tmp) → `/health` 200 (~4-6 min).
- [ ] **Step 2: Server A/B** — prefill bench 6.7K/27K/54K; decode code256 ≥84 tok/s; short256 ≥28 tok/s; `nvidia-smi` before/after 54K (no OOM).
- [ ] **Step 3: Mooncake check** — repeat 54K prompt: second TTFT drops (GET path intact, nvfp4 bytes unchanged — staging is read-only on the cache).
- [ ] **Step 4: Determinism** — 3 fixed greedy prompts vs pre-M4 capture: coherent, fp-level divergence only.
- [ ] **Step 5: Flip + docs + memory** — envs default, run2.sh, state doc, memory entries; leave server RUNNING with gate ON.
- [ ] **Step 6: Final report** — numbers table vs bars, residual risks (multi-request steps → FA2 fallback = phase-2 dense-route batch; partial-chunk q_len<64 → FA2; gqa-arch/splitkv3 ops registered but unused), next-phase candidates.

---

## Self-review (revised)

- Spec §1 criteria → T5 (TTFT bars) + T6 (decode/Mooncake/determinism). ✓
- Kernel provenance = 1Cat production .so source (highest fidelity; their build used the same `.deps` base dir). ✓
- No change to: M1 decode, DFlash2, GDN, GEMM, Mooncake, existing HND shadow (fallback+decode intact). ✓
- Rule compliance: all compiles via build_sm70.sh; python3 only; no pkill; log+poll for long runs. ✓
- Rollback: gate env off = byte-identical FA2 path (T4 test); build rollback = `mv build-sm70.bak-upstream build-sm70`. ✓
- Open detail (accepted): exact paged KV layout of the splitd kernel — fixed in T1 Step 4 by reading the kernel source; all downstream (T3 staging layout, T4 tensor views, T2 test shapes) follow it.
