# V1 PP+MTP Dataflow (Sampled/Draft/Cursor Broadcast)

> Purpose: document the *actual* business flow of the custom V1 PP+MTP path in
> `gpu_model_runner.py`, so deadlock/perf analysis is grounded in code, not
> guesses. All line numbers are from the current working tree.

## Topology

- PP=2 (ranks PP0, PP1), async scheduling ON, MTP2 (num_spec_tokens=2),
  NVFP4 KV.
- `PP1` = last rank = runs the sampler and the MTP drafter.
- `PP0` = non-last rank = runs the same transformer forward (verification) but
  NO sampler and NO drafter.
- The custom fork adds a per-step broadcast protocol so PP0 can:
  1. advance positions identically to PP1 (cursor / C4 backfill),
  2. embed the *real* draft tokens (not `-1` placeholders) into its
     verification inputs.

## Thread model

- **output-proc thread** runs `sample_tokens()` (both ranks).
- **driver thread** runs `execute_model()` (both ranks).
- These two threads overlap steps under async scheduling
  (sample_tokens(N) || execute_model(N+1)).

## Per-step flow on PP1 (last rank, sample_tokens)

`sample_tokens()` (`gpu_model_runner.py:4914`):

1. If `execute_model_state is None` (no-op step): returns early
   (`:4917-4925`), **sends nothing**.
2. Else (real step): `_sample()` (`:4950`), `_update_states_after_model_execute`
   (`:4952`; historical versions ran the removed token-0 sanitizer here),
   then **PP broadcast** (`:4960-4968`):
   `_pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids.clone())`.
3. Draft proposal runs (`:4977-5118`), then **PP draft broadcast**
   (`:5125-5132`): `_pp_broadcast_draft_token_ids()`.
4. Returns `ModelRunnerOutput` / `AsyncGPUModelRunnerOutput`.

### `_pp_broadcast_prev_sampled_token_ids` (`:5224`)

- `payload_present = not self._is_all_reqs_chunked_prefill()` (`:5247`).
- Always (header enabled): broadcast control `(1,1)` int32 = `1/0` (`:5248-5250`).
- If `payload_present`:
  - pad sampled grid to `[num_reqs, num_spec+1]` with `-1` (`:5259-5264`),
  - historical versions ran `sanitize_token_zero_col` here (`:5274`),
  - broadcast sampled `[num_reqs, num_spec+1]` (`:5279`),
  - broadcast cursor `[num_reqs, 1]` = `num_tokens_no_spec` (`:5285-5299`).
- Else (chunked):
  - broadcast all-`-1` sampled `[num_reqs, num_spec+1]` (`:5275-5278`),
  - broadcast zero cursor `[num_reqs, 1]` (`:5294-5298`).
- **All four sends are synchronous** (`dist.broadcast`), executed in order on
  PP1's sample_tokens thread.

### `_pp_broadcast_draft_token_ids` (`:5301`)

- Broadcast draft `[num_reqs, num_spec]` (real drafts, or all-`-1` when None),
  **unconditionally** (my change removed the chunked early-return at `:5318`).
- Also synchronous.

## Per-step flow on PP0 (non-last rank)

### `sample_tokens()` no-op branch (`:4917-4925`)

- If `execute_model_state is None`:
  `_pp_receive_prev_sampled_token_ids_to_input_batch()` (`:4922`) runs,
  then returns kv-connector-only output.

### `_pp_receive_prev_sampled_token_ids_to_input_batch` (`:5340`)

- `gen += 1` (`:5358`).
- Launches **4 async broadcasts** in fixed order:
  1. control `(1,1)` (header) (`:5369-5375`),
  2. recv `[num_reqs, num_spec+1]` (`:5377-5383`),
  3. cursor `[num_reqs, 1]` (`:5384-5392`),
  4. draft `[num_reqs, num_spec]` if `num_spec_tokens` (`:5395-5406`).
- Publishes `PPReceiveRound(gen, recv, cursor, draft, recv_work, cursor_work,
  draft_work, control, control_work)` atomically (`:5411-5426`), sets
  `_pp_round_event`.
- **Never waits** (my change moved the control wait out of here).

### `execute_model()` (`:4540...`)

- At top: `_pp_finish_receive_and_backfill()` (`:4571-4572`).

### `_pp_finish_receive_and_backfill` (`:5428`)

1. **Chunked-prefill branch** (`:5447-5471`): if `_is_all_reqs_chunked_prefill()`,
   does placeholder bookkeeping locally and `return`s — **drops the round
   without waiting** any `Work`.
2. Else waits for round (`:5475-5496`): if `_pp_pending_round is None`, spins on
   `_pp_round_event.wait(timeout=30)`; on timeout logs "PP sampled-token round
   N never published", invalidates mapping, returns.
3. Header (`:5506-5535`): `round.control_work.wait()` then
   `if not bool(round.control.item())`: placeholder bookkeeping and `return`
   (header=0 chunked round).
4. Payload (`:5536-5567`): `round.recv_work.wait()`, `round.cursor_work.wait()`,
   `round.draft_work.wait()`, sets `_draft_token_ids`/`_pp_waited_gen`.
5. C4 backfill (`:5568+`): `gather_valid_sampled_tokens_per_req(recv)`,
   persists real tokens into `token_ids_cpu`, aligns cursors, builds
   `prev_req_id_to_index`.

## Pairing / ordering contract (the crux)

Both ranks must execute **the same multiset of NCCL broadcasts in the same
order** for every PP "round". Current per-rank emission per step:

| rank | real step (execute_model_state non-None) | no-op step (execute_model_state None) |
|------|------------------------------------------|----------------------------------------|
| PP1  | 4 sync sends (header, sampled, cursor, draft) via sample_tokens | **0 sends** (early return at `:4925`) |
| PP0  | 0 (consumer waits round published earlier) | 4 async receives via sample_tokens no-op branch |

- PP0's producer (4 async receives) fires **only on no-op steps**.
- PP1's sender (4 sync sends) fires **only on real steps**.
- PP0's consumer fires **on real steps** (waits the round published during a
  previous no-op step).

### Observed deadlock (multi-sample py-spy)

- PP0 stuck forever at `round.control_work.wait()` (`:5508`) — an async NCCL
  broadcast that never completes.
- PP1 healthy (sampler/drafter running).

**Hypothesis under investigation:** when PP1 hits a no-op step (its
`execute_model_state is None`) it sends **nothing**, but PP0's producer on a
matching no-op step still launched 4 async receives -> those NCCL ops are never
paired -> PP0's consumer waits on `control_work` forever. Under async overlap,
PP0's no-op step can pair with PP1's no-op step (pipeline bubble / both-idle),
not only with PP1's real step.

> This asymmetry was *introduced* by my "always-emit" change on the send side
> combined with "never wait in producer" on the receive side. Baseline waited
> synchronously in the producer, which converted this into a recoverable 30 s
> timeout instead of a permanent wait.

## Root cause (confirmed by PPDBG repro, 2026-08-20)

The hang is **not** a PP collective deadlock. RECV/SEND/DRAFT counts stayed
perfectly paired (equal) even while the service hung; py-spy showed both
workers cycling through normal phases (sample, finish_receive, attn build) at
high CPU while no request ever completed. It is a **token-0 storm self-sustaining
loop** in the MTP path:

1. A trigger step makes PP1's sampled grid all `-1` (step 67951 in the repro:
   `SEND data=[[-1,-1,-1]]`, previously healthy tokens `[1510,36,6964]`).
2. `sanitize_token_zero_col` (sampled only) keeps `-1`; the **draft** grid then
   comes out as `[0,0]` (`SEND_DRAFT vals=[0,0]`).
3. `[-1, 0, 0]` is scattered into the next step's input (`INPUT_DUMP
   ids=[-1,0,0,...]`, `NEG_INPUT ... ids=[-1, 0, 0, ...]`).
4. MTP argmax on that input keeps producing `[0,0]`; draft 0 is **not sanitized**,
   so the 0 enters the input again → infinite loop. `num_tokens_no_spec`
   (`nct`) grows without bound (67932 → 79592 → ...), request never finishes.

So there are two distinct defects:
- **D1 (trigger):** a step where PP1's sampled grid becomes all `-1` (invalid
  logits / NVFP4 decode corruption at large cursor, or a request-state edge).
- **D2 (amplifier):** draft tokens are never sanitized, so a 0 draft is
  broadcast (`SEND_DRAFT vals=[0,0]`) and re-fed, making the loop self-sustain
  and pin the engine.

Historical versions ran `sanitize_token_zero_col` only on `output_token_ids`
(sampled) in `_update_states_after_model_execute` (`:1743`) — never on
`_draft_token_ids`.

### PPDBG evidence (repro2 log)

- `RECV(launch)` / `SEND shape` / `SEND_DRAFT` counts stayed equal even while
  hung (3300 vs 3299 at capture, an in-flight skew, no permanent divergence).
- First `vals=[0,0]` for the offending request at its 21st SEND_DRAFT; before
  that drafts were healthy `[248045,74455]`, `[1156,369]`, `[1510,36]`...
- `INPUT_DUMP ids=[1,0,0]` (PP0) vs `ids=[0,0,0]` (PP1): PP0 and PP1 diverge by
  one leading `1` — the 0 storm, not a collective mismatch.

### Root cause refinement (diag log, second repro) — cursor drift, not decode corruption

The second repro (67.9K-prompt request `bcaeeffba`) shows the real trigger is
**`num_tokens_no_spec` (the input cursor, printed as `step=`) drifting away
from `num_computed_tokens` (printed as `nct=`)**, not a sudden logits/NVFP4
failure:

- Prompt = 70575, prefill in 2 chunks (67968 + 2607), then decode.
- Healthy requests keep `step ≈ nct` (diff 1-3, the MTP spec-occupancy
  difference) — see the v3 log: step=20 ↔ nct=21, step=25 ↔ nct=25.
- The offending request drifts: at the trigger step `step=[70701,...]` while
  `nct=70799` — **a ~98-token gap**. `num_tokens_no_spec` lags
  `num_computed_tokens`.
- The step before the trigger everything is healthy (`INPUT_DUMP
  ids=[4577,63,14]`, sampled `[63,14,63]`, draft `[1882,12]`); the next step
  sampled becomes all `-1` and draft `[0,0]`.

Mechanism: MTP optimistically advances `num_computed_tokens` by
`1 bonus + num_drafts` each step, but `num_tokens_no_spec` (the cursor the
next input grid is built from) advances by fewer; the gap grows every step.
When the lag is large enough, MTP reads/builds inputs at a stale position →
sampled grid goes all-`-1` (invalid) → draft `0` is broadcast un-sanitized →
0-storm self-sustains and pins the engine.

So the user's intuition is correct: **the request keeps generating past where
its input cursor was correctly advanced — "it kept generating when there was
no (correctly positioned) input left".** The defect is cursor/position
advancement desync, not PP collective mismatch and not KV decode corruption
(no NaN/-inf logits observed).

### A/B attribution (experiment A, async off) — the self-dev async PP path is the cause

Ran the same 67.9K-prompt workload with `async-scheduling: false` (this
bypasses the self-dev async PP broadcast; the MTP spec decode itself stays on,
and the sync PP path — official-style — handles sampled tokens via scheduler
return instead of the async GPU broadcast):

- nct advanced past 100218 (prompt 70575 + ~30K generated tokens),
  **0 SAMPLE_BAD, 0 draft `[0,0]`, 0 never-published, 31 requests completed**.
- With `async-scheduling: true` (self-dev async PP broadcast active), the same
  workload hit the 0-storm after ~130 generated tokens.

**Attribution: the self-dev async PP broadcast path is the cause of the
0-storm hang, not MTP or the sync PP path.** The sync path (async off) runs
MTP+PP correctly for the same 67.9K prompt.

### Fix direction (chosen)

Not a rewrite. Fix the cursor/position desync inside the self-dev async PP
broadcast path (`_pp_broadcast_prev_sampled_token_ids` / receiver / C4
backfill interplay) — where `num_tokens_no_spec` drifts from
`num_computed_tokens`. The sync path is the reference for correct behavior.

## Upstream survey (2026-08-20) — official answer is V2, not V1

- **Issue #49355** "[Bug]: MTP speculative decoding is broken with pipeline
  parallelism (PP>1) — three distinct failures": upstream reporter hit the same
  class of bugs. Failure 1 = `GPUModelRunner has no attribute 'drafter'` on
  non-last ranks; Failure 2 = `PP+async expects sampled_token_ids to have shape
  [num_reqs,1]` (PP receive path assumes 1 sampled/req but MTP emits
  `[num_reqs, num_spec+1]`); Failure 3 = device-side out-of-bounds gather in
  `hidden_states[logits_indices]` on the last PP rank (cursor/indices desync).
- **Official answer (njhill, vLLM maintainer): use V2 model runner
  `VLLM_USE_V2_MODEL_RUNNER=1`. V1 is not fixed for MTP+PP.**
- Community V1 PRs (all OPEN, unmerged): #49442 (drafter only on last rank),
  #49443 (broadcast `input_batch.prev_sampled_token_ids` — the previous,
  verified step — instead of current `sampler_output.sampled_token_ids`),
  #52179 (PP spec cadence sync). #50514 is a **V2** EAGLE3+PP implementation.
- Upstream `origin/main` V1 PP broadcast contract:
  `_pp_broadcast_prev_sampled_token_ids` / `_pp_receive_...` assume
  `[num_reqs, 1]`, set `input_batch.prev_sampled_token_ids = recv`, build
  `prev_req_id_to_index` there, and advance `num_tokens_no_spec[i] = pos + 1`
  (one token per request per step). **MTP's multi-token-per-step output is not
  representable in that contract** — hence upstream pushes V2.

### V2 record (deferred, not the path we are taking now)

- `VLLM_USE_V2_MODEL_RUNNER=1` is upstream's official MTP+PP path.
- In this environment V2 was verified NOT usable: startup FlashInfer autotune
  PP-collective mismatch hang; with autotune off, CUDA-graph capture failed at
  `qwen3_5_mtp.py:185` (`input_ids < 0` check under stream capture). Recorded
  for later; not pursued now.

## Current Investigation Update (2026-08-20)

### Confirmed token-0 defects removed

Token ID `0` is a valid vocabulary item. The custom `sanitize_token_zero_col`
path was invalid for two independent reasons:

1. It changed valid sampled/draft data whenever the model emitted token `0`.
2. Its `replace.any()` converted a CUDA bool to a Python bool and synchronized
   the device. py-spy caught PP1 blocked in `sanitize_token_zero_col ->
   _local_scalar_dense_cuda -> cuStreamSynchronize`, while PP0 and EngineCore
   waited for the PP response.

The sanitize function and both call sites were removed. A second bookkeeping
path that replaced list entries equal to `0` was also removed; it additionally
used `start_idx` before assignment. Token `0` is now preserved unchanged.

### New runtime evidence: cursor semantics are inconsistent

The diagnostic records the first occurrence of the token sequence
`[nonzero, 0, 0, 0]` once per request. Two independent long-request events
showed:

```text
pattern=[1092, 0, 0, 0]
computed=121773 no_spec=121777 prompt=121670

pattern=[1042, 0, 0, 0]
computed=121772 no_spec=121776 prompt=121680
prev_draft=2 scheduled=3 spec=2
```

The second event occurred exactly at `121776 = 43 * 2832`, matching the
attention block size printed during startup. This implicates a cross-block
state/slot transition, but the first event shows the same failure class is not
limited to that absolute boundary.

After reverting the experimental PP1 `sampled_ids = [-1] * v` change, another
event showed the opposite cursor error:

```text
pattern=[17, 0, 0, 0]
computed=48140 no_spec=48123 prompt=48110
prev_draft=2 scheduled=3 spec=2
```

Thus neither fixed `+1` nor direct `+v` bookkeeping is a valid solution. The
two variants move `num_tokens_no_spec` to opposite sides of
`num_computed_tokens`.

### Current root-cause boundary

The custom C4 path writes sampled values using
`num_tokens_no_spec` (`gpu_model_runner.py:5634-5649`), while the next input
read is indexed by `num_computed_tokens` (the C4 comments explicitly describe
this at `:5590-5595`). These are not interchangeable cursors. Their observed
divergence explains both failure directions:

```text
PP1 write cursor: num_tokens_no_spec
next input read:  num_computed_tokens
```

The resulting stale or misplaced token/KV state can produce a repeated tail,
then invalid sampled output and eventual zero draft acceptance. The current
investigation must resolve this cursor contract before changing token values or
adding further defenses.

### Diagnostic controls

`PP_TOKEN_DIAG=1` enables one-shot diagnostics only:

- `[TOK0_PATTERN]`: `[nonzero, 0, 0, 0]`, including position counters.
- `[TOKEN_REPEAT]`: automatically detects a repeated period from 1 through 16,
  requiring three consecutive copies, and records only the first match.

The diagnostics are disabled by default so normal async execution does not
maintain token histories or construct position-context dictionaries.

### Newly confirmed control-flow bug

`GPUModelRunner._prepare_input_ids()` has a common-case optimization when the
batch order is unchanged. Before this fix, that branch copied the latest
sampled token and returned unconditionally. With MTP scheduled tokens, the same
method's draft scatter is later in the method, so the non-last PP rank never
copied the received draft grid into `input_ids` for the common batch case.

That leaves the verification input using stale values at draft positions. It
can present as cursor/KV drift because the sampled grid and local input grid no
longer describe the same token sequence. The fix keeps the early return only
when `total_num_spec_tokens == 0`; MTP batches continue to the existing draft
scatter path. A regression test covers this control-flow invariant:
`test_prepare_input_ids_common_spec_path_reaches_draft_scatter`.

This is a separate, concrete defect from the remaining absolute-cursor
semantics. The service must be retested after this fix before changing
`num_tokens_no_spec` or `num_computed_tokens` arithmetic.

### Rejected experiment: forcing last-rank bookkeeping to grid width

The first post-fix long log showed the previous draft-scatter defect was no
longer the immediate issue: PP0 and PP1 `INPUT_DUMP` values matched, including
the draft columns. PP1 still emitted multi-token grids while its bookkeeping
often advanced `num_tokens_no_spec` by one. An experiment changed the async
placeholder width from one to the sampled width:

```text
BK v=3 ... ntns=48126
NCT ... nct=48135
```

```python
sampled_ids = [-1] * v
```

This experiment was rejected by the next long log. It made the sampler-rank
cursor run ahead instead:

computed=50971 no_spec=50975
```

The change has been reverted. The remaining issue is the interaction between
optimistic scheduler counts, C4 backfill, and Mamba/GDN state transitions; the
placeholder width cannot be selected from `v` alone.

The latest repro also shows the first zero storm at `computed=50971`, five
tokens before `50976 = 18 * 2832`, with identical PP0/PP1 input grids. This is
the current state-copy investigation target.

## Trace infrastructure (2026-08-20, runtime capture)

`vllm/v1/worker/pp_trace.py` adds an opt-in per-request ring buffer that dumps
only on anomaly, so normal runs write nothing and no GPU->CPU sync is added.

- Enable: `PP_TRACE_DIR=/tmp/vllm_trace` (+ optional `PP_TRACE_ROUNDS=500`).
- Anomaly trigger needs `PP_TOKEN_DIAG=1` (feeds the `TOK0_PATTERN` /
  `TOKEN_REPEAT` detectors which call `_trace.trigger`).
- Dump: `<PP_TRACE_DIR>/<req_id>.r<rank>.txt`, one line per round:
  - `STATE` (`_prepare_inputs`, gpu_model_runner.py:2499): `accepted`
    (=`num_accepted_tokens.np`, the value the GDN metadata builder receives),
    `state_idx` (`mamba_state_idx`), `nct` (`num_computed_tokens_cpu`).
  - `ACCEPT` (`_bookkeeping_sync`, :4210): `v` (real rejection accepted count
    `(st != -1).sum()`), `computed`, `no_spec`, `nct2` (same `accepted`).
  - All values are CPU-side (numpy / dict / python lists); no `.gpu...tolist()`.
- Every trace point is a no-op unless `PP_TRACE_DIR` is set.

### Two real captures (same session, corrected key `...f0d83...`)

1. **True token-0** (`chatcmpl-bed5b78257ca5d62`, `pattern=[0]`, drift 24):
   prompt 142080, 2000 tokens generated. Zero appears right after
   `state_idx` 50->51 transition:
   ```
   3099 ACCEPT v=1 computed=144388 nct2=3     <- real accept 1, accepted buffer 3
   3101 STATE accepted=1 state_idx=50 nct=144391
   3144 ACCEPT v=1 computed=144431 nct2=1     <- v == nct2 == 1
   3146 STATE accepted=1 state_idx=51 nct=144434   <- cross-block transition
   ... then token 0
   ```
   The cross-block frame: `nct=144431, v=1, scheduled=3, draft=2` computes
   `num_tokens_running_state = 144431+3-2 = 144432`,
   `new_num_computed = 144432+1-1 = 144432`, `aligned_new_computed = 144432`
   (= 51*2832 exactly), `dest_block_idx = 50`, `src_block_idx = 50`
   -> judged **same-block**, `accepted` reset to 1, no copy.
2. **False positive** (`chatcmpl-9c8aac71c51d9945`, `pattern=[15]`): finished
   normally, `tail=[1905,1,10598,30,561,1156,2640,328]` (no 0). A normal-text
   token repeating 3x, not a defect. `TOKEN_REPEAT` is a noisy detector; only
   `TOK0_PATTERN`/`pattern=[0]` indicate real decay.

### What this round established (confirmed)

- `num_accepted_tokens.np` (`nct2`, what GDN metadata gets at forward) is the
  **previous** frame's postprocess result; `v` is the **current** frame's real
  rejection count. They are legitimately two frames apart; `v=1 nct2=3` is NOT
  an inconsistency bug.
- `num_computed_tokens - num_tokens_no_spec` is not a monotonic drift; it
  oscillates (trace 2) and is the normal MTP spec-occupancy gap. The two cursors
  are not interchangeable, but their difference alone does not explain the zero.
- The zero appears **right after a `state_idx` cross-block transition** in every
  real capture (this one: 50->51 at 144434; earlier: 18->19 at 50976; also the
  `121776 = 43*2832` boundary from a prior capture). Cross-block GDN state
  migration is the prime suspect.
- GDN forward rollback uses `num_accepted_tokens = diff(query_start_loc)`
  (gdn_attn.py:558), derived from scheduling, independent of
  `num_accepted_tokens.np`; the fused postprocess migration uses the frame's
  real `v`. These are two separate accepted-token consumers.

### What is NOT yet confirmed (open)

- The exact postprocess decision (`needs_copy`, `accept_token_bias`,
  `dest_block_idx`) at the cross-block frame cannot be rebuilt from the current
  trace: `stage_postprocess_inputs_to_gpu` (mamba_utils.py:1497) writes
  `state_idx/scheduled/computed/draft` into GPU-staging numpy views, but those
  are not traced. `num_computed` there is the GPU-corrected value, not the CPU
  optimistic one shown in `STATE`.
- Whether the "same-block judgment on a block-boundary-aligned frame" is the
  corruption trigger, or whether the corruption is earlier (a low-acceptance
  frame `v=1` near a block edge) still needs a decisive test.
- Code alone cannot pin this: the branch is value-dependent at runtime.

### Next decisive step (if pursued)

Add a `POST` trace in `stage_postprocess_inputs_to_gpu` (or read
`mamba_state_idx`/`num_scheduled`/`num_computed`/`num_draft` CPU views at that
point) so a reproduced cross-block frame can be replayed exactly through
`postprocess_mamba_fused_kernel`'s decision arithmetic (mamba_utils.py:403-435)
and the failing frame identified deterministically.

## Root-cause chain closed (2026-08-20, POST + STOP_DIAG captures)

Two new diagnostics closed the loop:

- **`POST` trace** in `stage_postprocess_inputs_to_gpu` (mamba_utils.py): records
  the exact fused-postprocess inputs `state_idx / scheduled / computed / draft`
  per frame (all CPU numpy).
- **`STOP_DIAG`** in `check_stop` (v1/core/sched/utils.py): logs the real stop
  reason. It proved the "premature stop" is **EOS**: `eos last=248046
  eos=248046 n_out=8019` — the model emitted EOS token 248046 after 8019
  tokens. EOS stop does not set `stop_reason` (utils.py:104-106), so
  `finish=stop stop=None` is the *normal* EOS path, not a truncation bug.

### Confirmed causal chain (true reproduction, 142080-prompt request)

1. Long decode (~144K computed): `state_idx` crosses a block boundary (50->51 at
   nct≈144494; also 18->19 at 50976, 43*2832 boundary in earlier captures).
2. After the boundary, rejection starts rejecting drafts (`ACCEPT v=1` = only
   bonus accepted) while `num_accepted_tokens.np` still holds 3 (a one-frame lag
   that is design-intended but leaves GDN state migration/rollback consuming
   mismatched counts across the boundary).
3. Model enters a repetition loop: `TOKEN_REPEAT period=8 pattern=[62284, 736,
   4967, 1, 198, 760, 328, 3841]` at computed=144484.
4. EngineCore sees EOS (or repetition) and stops: `finish=stop stop=None`
   (EOS) — this is the "response disappears / premature stop" symptom.
5. If decay is worse the loop degrades to `pattern=[0]` (token-0 storm),
   self-sustaining until the request is aborted or the cursor runs away.

So "premature termination", "tool-call failure", and "token-0 repetition" are
**one defect family**: cross-block GDN/linear-attention state migration in the
custom async PP+MTP path corrupts the recurrent state at long sequence, the model
degrades into repetition, and the symptom depends on how far the decay goes.

### What is still not pinned to a single line (final open item)

The exact cross-block frame where `num_accepted_tokens` (postprocess migration,
uses frame-N real `v`) disagrees with GDN forward rollback (`diff(query_start_loc)`,
gdn_attn.py:558) has not been reduced to a concrete fix yet. The `POST` trace now
records all four postprocess inputs, so a reproduced boundary frame can be
replayed through `postprocess_mamba_fused_kernel` (mamba_utils.py:403-435)
arithmetic to find the exact bad `needs_copy`/`accept_token_bias`/`dest_block_idx`
value. That replay + a targeted fix is the remaining step.

## Root cause pinned: postprocess reverse-copies state across the boundary

POST replay + full code path analysis located the exact defect. This is the
"final open item" now closed.

### Evidence frame (replay of chatcmpl-bb795d9108dc8384, step 4004)

```
4003 STATE accepted=3 state_idx=50 nct=144431
4004 POST state_idx=51 scheduled=3 computed=144431 draft=2   <- boundary frame
4005 ACCEPT v=1 ... nct2=1
```

`num_tokens_running_state = 144431 + 3 - 2 = 144432 = 51*2832` — **exactly block-aligned**.

### Two kernels disagree about which column is `src` vs `dst`

Both the fused **precopy** (in `preprocess_mamba`, V1 align path) and the fused
**postprocess** (`postprocess_mamba_align_gpu` → `run_fused_postprocess`) run in
the SAME step when `with_postprocess_align` is on (spec decode + hybrid, i.e.
this server, gpu_model_runner.py:1254).

1. **`preprocess_mamba`** (mamba_utils.py:1267) advances `mamba_state_idx` to the
   new block and copies 50→51 correctly:
   - `prev_state_idx = 50`, `curr_state_idx = 51` (1334-1335)
   - fused `src_col = prev_state_idx = 50`, `dst_col = curr_state_idx = 51`
     (1343, precopy kernel 578) → **50→51, correct**
   - resets `num_accepted_tokens_cpu = 1` (1357) and re-syncs to GPU (gpu_model_runner:4810-4813)
2. **`postprocess_mamba_fused_kernel`** (mamba_utils.py:395-467) then runs and
   reverse-copies:
   - `src_block_idx = load(mamba_state_idx_gpu)` (413) = **51** (already advanced!)
   - `num_accepted = 1` (preprocess reset) → `new_num_computed = 144432 + 1 - 1 = 144432`
   - `aligned_new_computed = 144432`, `needs_copy = 144432 >= 144432` = **True**
   - `dest_block_idx = 144432//2832 - 1 = 50` (434)
   - `src(51) != dest(50)` and `bias(0) != 0`-skip guard (443) is `src==dst`
     only → does **NOT** skip → executes `_copy_mamba_state_block(51 → 50)`
     **reverse-copying the just-migrated state back into the old block.**

### Trigger condition

The reverse copy only fires when `needs_copy=True`, i.e. when the frame's
`new_num_computed` is **exactly block-aligned** (`aligned_new_computed >=
num_tokens_running_state`). Because preprocess resets `num_accepted` to 1 on
boundary crossing, `new_num_computed = num_tokens_running_state + 1 - 1 =
num_tokens_running_state`; if that lands on a block boundary (`51*2832`), the
postprocess kernel's `dest = aligned//B - 1` computes the OLD block while `src`
is the NEW block → reverse copy. After it, `mamba_state_idx` stays at 51 but
block 51's state was clobbered back to 50's content, so the GDN forward reads a
stale/wrong recurrent state → model degrades → repetition loop → EOS / token-0.

This matches the observed rarity (exact alignment is uncommon) and why it only
appears at long sequences where many boundary crossings accumulate.

## Logical review: the reverse-copy hypothesis has a hole — real defect is the SSM rolling-column misalignment at boundary

A careful review of the reverse-copy root cause found it does NOT actually
corrupt the live state, and identified a stronger candidate.

### Why the reverse copy (51→50) is likely harmless

`_copy_mamba_state_block(src=51 → dst=50)` READS block 51 and WRITES block 50.
GDN reads its initial state from `block[start_indices]` for conv and from
`block[start_indices + num_accepted - 1]` for SSM (fused_sigmoid_gating.py:106-126,
1347). After the boundary, `mamba_state_idx` stays 51 and GDN keeps reading
block 51 (`start_indices + 0..2`). Block 50 is only the *previous* running block;
`preprocess_mamba` never reads it again after it migrated away (next migration is
from prev=51 → curr=52). So the reverse copy writes a dead block → no live-state
corruption. The hypothesis as stated fails its own harm analysis.

### The real candidate: SSM rolling-column misalignment at the boundary

The GDN SSM kernel indexes state by a **column** that is relative to the
CURRENT frame's `start_indices`:

- `ssm_state_indices = block_table_tensor[:, :num_spec+1]` and
  `block_table_tensor = block_table[:, start_indices : start_indices+1+num_spec]`
  (gdn_attn.py:308-310, utils.py:1032-1041), with `num_spec = num_speculative_tokens = 2`.
- Read: `i_t = num_accepted - 1`, `state_idx = ssm_state_indices[i_n, i_t]`
  = `block[start_indices + num_accepted - 1]` (fused_sigmoid_gating.py:106-126).
- Write (inplace final state): each token `i_t` writes
  `ssm_state_indices[i_n, i_t]` = `block[start_indices + i_t]`
  (fused_sigmoid_gating.py:170-176).

`num_accepted = diff(query_start_loc) = num_scheduled_tokens` per frame
(input_batch.py:153), i.e. 3 for MTP2 (bonus + 2 drafts) → read column 2.

**Non-boundary frames roll correctly** (start_indices constant): frame N writes
columns 0..2 = block[S..S+2]; frame N+1 reads column 2 = block[S+2] = frame N's
draft-2 state. Consistent.

**Boundary frames misalign** (start_indices jumps S → S+1): frame N writes
block[S..S+2]; frame N+1 reads column 2 = **block[S+3]**, which frame N NEVER
wrote → GDN initializes from an unwritten/stale block → state drift.

`preprocess_mamba` only migrates the running state column (prev→curr, one
src/dst pair, mamba_utils.py:1339-1344/1346-1356), NOT the rolling columns
1..2. So after a boundary the rolling columns' absolute addresses shift by one
block while their contents are never carried over → mismatch. This matches:

- the observed rarity (only exact-block-aligned boundary frames in the replay
  show the anomaly window; here 4005 ACCEPT v=1 = first rejection right after
  the 4004 boundary);
- the slow drift (intermittent rejections for ~50 frames before full
  repetition → EOS);
- why only long sequences (many boundary crossings) decay.

### Uncertainties to close with a GPU-side check (not yet done)

1. Confirm frame 4004's GDN actually reads block[start_indices+2] = block 53 and
   that block 53 holds stale/zero data (dump the SSM state block before/after).
2. Confirm whether `preprocess_mamba`/fused precopy is *supposed* to migrate the
   rolling columns (check upstream semantics / test suite expectations for
   boundary crossing with num_spec>0).
3. Decide fix: either migrate all `1+num_spec` rolling columns on boundary, or
   index the SSM read column against the PREVIOUS frame's start_indices.

Note: the STOP_DIAG/EOS finding and the overall symptom family (repetition →
EOS or token-0) are unchanged and still valid.

### Confirmed by offline replay (2026-08-20): the misalignment is real, block-address proven

Replayed all 125 POST frames from chatcmpl-bb795d9108dc8384 against the GDN
read/write block arithmetic:

- GDN SSM read block per frame = `start_indices + num_accepted - 1`
  (`num_accepted = diff(query_start_loc) = num_scheduled_tokens = 3` → +2).
- GDN SSM write blocks per frame = `start_indices + {0..scheduled-1}`.
- `start_indices = (num_computed + num_scheduled - 1) // block_size`.

Result: **exactly one frame is misaligned — step 4004, the single boundary
crossing (start_indices 50 → 51).** It reads block 53 while the previous frame
wrote only blocks 50, 51, 52 → it initializes the SSM recurrent state from a
block that was never written by the previous step (fresh / prefill-default or
stale). All other 124 frames read a block that the previous frame wrote.

This is block-address proof, independent of the state contents: the rolling
column shifts by one block at the boundary while its contents are never carried
over (preprocess migrates only column 0). Fix options remain those listed above.

### CPU simulation confirms the fix (2026-08-20)

Replayed all 125 real POST frames through a state-position simulator with the
exact block arithmetic:

- `single` migration (current code): **1/124 misaligned reads** — exactly the
  boundary frame (`computed=144431, S=51`, GDN reads block 53 = `None`, should be
  144431).
- `rolling` migration (proposed fix: on boundary, carry the whole
  `NUM_SPEC+1`-column window `prev_S+c -> S+c`, high-to-low to avoid clobber):
  **0/124 misaligned** — boundary frame now reads the correct carried-over state.

The migration must run *before* the GDN read and after the previous GDN write —
the real preprocess (which runs before forward) already satisfies this ordering.
This is strong evidence that migrating the whole rolling window at the boundary
fixes the defect without touching GDN or postprocess.

### Upstream status

Searched vllm-project/vllm (via GitHub API + local `origin/main`):

- #52873 "Qwen3-Next GDN + MTP crossing ... kills draft acceptance" — same
  symptom family (GDN+MTP acceptance collapses at a long-sequence boundary), but
  closed as checkpoint-specific, not a vLLM defect; no fix to port.
- #52244 "Restore hybrid GDN prefix-cache hits under MTP spec decoding" — GDN
  state *publication position* for prefix cache (prefill side), different code
  path; references #43559 "state poisoning".
- #50021 / #53077 / #52078 — related GDN fixes, none address the decode-phase
  rolling-column boundary copy.
- No existing PR addresses decode-phase cross-block SSM rolling-column
  misalignment. The preprocess/fused-precopy machinery is upstream code
  (this fork only adds pp_trace + pointer bounds guards), so the defect is
  plausibly present upstream too, exposed only by long sequences.

### Fix decision

Migrate the whole `1 + num_spec` rolling window at the boundary in
`preprocess_mamba` (and the fused precopy kernel), high-to-low, for the temporal
state; keep the existing conv running-state migration. Not yet implemented —
needs the conv/temporal split in `collect_mamba_copy_meta`/`run_fused_precopy`
and validation.

## Content-based degradation confirmation (2026-08-20, streaming probes)

The earlier "degradation = early EOS token count" method was WRONG (models are
stochastic; token count alone is not a defect signal). Re-validated with
**content** via a streaming probe (`/tmp/opencode/probe_stream.py`, 4 cases) on
a clean baseline (MTP2 + fused GDN, no uncommitted fixes):

| case | prompt | result | evidence |
|---|---|---|---|
| tok0 | 142K | OK this run | no `!` storm (but other runs show `ax6is` token-loop tail) |
| tool | 100K | **DEGRADED** | **mandatory get_weather tool NOT called** (instruction-following broken) |
| ending | 100K | **DEGRADED** | ends with truncated `and`, not the required ending phrase |
| repeat | 142K | **DEGRADED** | same sentence repeated **62×**, 30-char frag 128×, 6-char frag 271× (token loop) |

And the critical control: **drafter OFF** (`speculative-config` removed) with a
142K prompt produced **fully coherent content** (0 exclamation marks, no loop),
only stopping early at ~1493 tokens. So the content corruption is **specific to
the MTP/GDN spec-decode path**, not a model/long-context/quantization property.

This replaces the dead "cross-block rolling-column" hypothesis (rolling
migration was A/B-neutral: 2354 vs 2354 tokens) and re-focuses root-causing on
MTP/GDN recurrent-state corruption at long sequence. The rolling-migration fix
was reverted; baseline is clean again.

Symptoms family (all seen across runs): token-0 `!` storm, `ax6is`/similar
token-loop tails, sentence repetition 62×, truncated endings, tool-call failure
(no/malformed call). All consistent with GDN/MTP recurrent state corrupting at
long sequence and the model degrading into a loop.
