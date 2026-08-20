# V1 Fixed-Frame Final Fix Report

Date: 2026-08-20
Worktree: `/tmp/opencode/vllm-v1-pp-frame`

## Findings Addressed

- Replaced CRC32 request row identity with an exact fixed-width 64-byte UTF-8
  representation transported as sixteen `int32` words. IDs containing NUL or
  exceeding 64 bytes are rejected, so distinct supported IDs cannot collide or
  silently truncate.
- Expanded frame validation to enforce fixed key/payload shapes, shared device,
  `int32` transport dtype, legal flags, non-negative active cursors, zero
  inactive keys/cursors, and `-1` inactive payloads.
- Inactive rows are excluded during alignment and therefore cannot provide
  sampled tokens, drafts, or cursors to local scheduling.
- Timeout and protocol-failure paths terminate the receive work when supported,
  mark the round terminated, fence its generation, clear publication state, and
  prevent stale round reuse. The protocol remains one fixed-capacity collective
  per scheduling step.

No V2 or attention backend files were modified.

## Tests and Checks

```text
VLLM_TARGET_DEVICE=cpu python -m pytest -q tests/v1/worker/test_pp_spec_broadcast.py
14 passed, 15 warnings

python -m py_compile vllm/v1/worker/pp_spec_broadcast.py \
  vllm/v1/worker/gpu_model_runner.py
PASS

git diff --check
PASS
```

The warnings are the existing missing `vllm._version` warning and TorchScript
deprecation warnings. Full GPU/NCCL and service E2E checks remain unavailable
because this checkout lacks the compiled `vllm._C_stable_libtorch` extension.

## Final Re-review Fix Wave

- `_pp_timed_out_gen` is consulted before publishing and consuming rounds;
  stale generations are rejected and terminated rather than reused.
- Validation failures now terminate the associated round before being surfaced.
- `wait_pp_work` invokes Work termination on both timeout returns and wait
  exceptions.
- Removed the misleading post-rejection inactive-payload mutation.
- Row-key encoding failures from the public alignment helper are normalized to
  `PPProtocolError`.
- Added focused coverage for timeout fencing, round termination, wait exception
  cleanup, timeout cleanup, and row-key error normalization.

Final focused result: `19 passed, 15 warnings`.

## Remaining Fencing Fixes

- Stale receive rejection now terminates the pending round before raising.
- Alignment and post-wait row application run under the round cleanup path;
  `PPProtocolError` terminates the round and advances the generation fence.
- Added focused coverage for pending-round termination on stale receive.

Verification: `20 passed, 15 warnings`; Python compilation and `git diff --check`
passed.

## Complete Application Cleanup

- Added an outer cleanup boundary around the entire post-wait application
  routine, including gather, tensor writes, request-state updates, cursor
  alignment, and backfill bookkeeping.
- Any exception now terminates the active round and advances the generation
  fence before propagating the original error.
- Added an application-stage exception regression test.

Final focused result: `21 passed, 15 warnings`.

## Whole-Branch Review Final Fix Wave

- Terminating or superseding a pending receive round now aborts and waits each
  unique Work handle when supported, while tolerating mocked Work without
  `abort()` or with failing cleanup methods.
- Cursor application now reconciles in both directions to the sender-authoritative
  cursor. Forward gaps are safely filled; sender-behind cursors move back without
  truncating valid request state.
- Frame packing rejects non-`int32` transport tensors instead of silently casting
  sender payloads.
- Added focused lifecycle, cursor-ahead/behind, and dtype-rejection tests.

Verification: `25 passed, 15 warnings`; Python compilation and `git diff --check`
passed. Full GPU/NCCL checks remain unavailable because the compiled extension is
not present in this checkout.

## Scoped Review Follow-up

- Cursor reconciliation now indexes `last_cursor[i]` before checking the row's
  validity, preventing list-versus-integer comparison failures in the actual
  backfill loop.
- Pending-round supersession now uses an unconditional termination path, so a
  pending round's Work is aborted and waited even when its generation is not
  currently fenced.
- Added regressions for indexed per-row cursor handling and unfenced pending Work
  supersession.

Verification: `27 passed, 15 warnings`; Python compilation and `git diff --check`
passed.

## All-Chunked-Prefill Frame Fix

- All-inactive sender frames now retain zero row keys and cursors; sampled and
  draft payloads remain `-1`, satisfying `validate_pp_frame` while preserving
  the single fixed-capacity collective.
- Added coverage with nonzero local row metadata proving the packed inactive
  frame validates and does not align to local requests.

Verification: `28 passed, 15 warnings`; Python compilation and `git diff --check`
passed.
