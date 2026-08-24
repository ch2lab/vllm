# V1 PP+MTP Fixed-Frame Broadcast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom V1 PP+MTP conditional multi-broadcast protocol with a fixed-shape, single-frame protocol that remains symmetric when requests are cancelled or discarded.

**Architecture:** PP1 always broadcasts one contiguous frame per PP step. The frame has fixed capacity, carries generation and row metadata, and includes sampled, cursor, and draft payloads. PP0 always participates, validates the frame, and aligns rows by stable request key before applying the payload.

**Tech Stack:** Python, PyTorch distributed/NCCL, vLLM V1 worker, pytest.

## Global Constraints

- Preserve NVFP4 KV cache, FlashInfer attention, PP2, MTP2, and async scheduling.
- Do not modify the V2 runner or attention backend selection.
- Do not conditionally skip the PP frame collective based on local request state.
- Bound communication waits and raise a protocol error with generation and rank context on timeout or metadata mismatch.

---

### Task 1: Define the PP frame contract

**Files:**
- Modify: `vllm/v1/worker/pp_spec_broadcast.py`
- Test: `tests/v1/worker/test_pp_spec_broadcast.py` (create if absent)

**Interfaces:**
- Produces a fixed-frame dataclass/packing contract containing generation, row keys, row flags, cursors, sampled tokens, and draft tokens.
- Consumes `max_num_seqs` and `num_spec_tokens` from the runner.

- [ ] **Step 1: Inspect existing `PPReceiveRound` and test conventions.**
- [ ] **Step 2: Write unit tests for fixed frame shape, generation monotonicity, and pack/unpack round trips.**
- [ ] **Step 3: Run the focused tests and confirm the new contract tests fail.**
- [ ] **Step 4: Implement the smallest frame metadata and packing helpers without distributed calls.**
- [ ] **Step 5: Run the focused tests and confirm they pass.**

### Task 2: Replace three conditional broadcasts with one fixed broadcast

**Files:**
- Modify: `vllm/v1/worker/gpu_model_runner.py:5223-5404`
- Modify: `vllm/v1/worker/pp_spec_broadcast.py`
- Test: `tests/v1/worker/test_pp_spec_broadcast.py`

**Interfaces:**
- PP1 produces one contiguous frame every step.
- PP0 launches one async receive every step and publishes only a complete frame.

- [ ] **Step 1: Add a failing test proving a chunked-prefill sender and non-chunked receiver still perform one collective.**
- [ ] **Step 2: Add fixed-capacity buffers based on `max_num_seqs` and configured speculative width.**
- [ ] **Step 3: Pack sampled, cursor, and draft data into the frame on PP1, including row keys and flags.**
- [ ] **Step 4: Make PP0 always post the matching receive and remove local conditional skip branches.**
- [ ] **Step 5: Validate generation, frame dimensions, and row keys before applying the frame.**
- [ ] **Step 6: Run the focused tests and confirm cancellation/shape-divergence cases pass.**

### Task 3: Make row application cancellation-safe

**Files:**
- Modify: `vllm/v1/worker/gpu_model_runner.py` around `_pp_finish_receive_and_backfill`
- Test: `tests/v1/worker/test_pp_spec_broadcast.py`

**Interfaces:**
- Consumes a validated frame and the local request map.
- Produces aligned `prev_sampled_token_ids`, draft IDs, cursors, and placeholder bookkeeping.

- [ ] **Step 1: Add tests for cancelled rows, reordered rows, missing local rows, and new local rows.**
- [ ] **Step 2: Match rows by stable request key instead of row index.**
- [ ] **Step 3: Apply discard flags and placeholders only after frame validation.**
- [ ] **Step 4: Run focused tests and verify no local mask can change collective participation.**

### Task 4: Add bounded wait and protocol diagnostics

**Files:**
- Modify: `vllm/v1/worker/gpu_model_runner.py`
- Test: `tests/v1/worker/test_pp_spec_broadcast.py`

**Interfaces:**
- Uses the existing distributed Work API with a configured bounded wait.
- Raises an error containing PP rank, generation, expected shape, and received metadata.

- [ ] **Step 1: Add a failing test for a Work handle that does not complete before the configured timeout.**
- [ ] **Step 2: Implement bounded waiting and explicit protocol errors.**
- [ ] **Step 3: Run focused tests and verify the error contains actionable context.**

### Task 5: End-to-end PP2+MTP2 verification

**Files:**
- Modify: `/root/run/Qwen3.8-27B-nvfp4-pp2-mtp2.yaml` only if test configuration needs a documented transport setting.
- Test/log: `/tmp/opencode/vllm_ppv1_fixed_frame.log`

- [ ] **Step 1: Run unit tests for the frame and row-alignment logic.**
- [ ] **Step 2: Start V1 PP2+MTP2 with FlashInfer autotune disabled only if required by startup stability.**
- [ ] **Step 3: Verify a normal completion on port 8002.**
- [ ] **Step 4: Cancel an in-flight request and verify a subsequent request completes.**
- [ ] **Step 5: Run concurrent requests with batch reshuffling and inspect PP generation diagnostics.**
- [ ] **Step 6: Run the required project tests and record any remaining limitations.**
