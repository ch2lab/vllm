# Task 3 Report: Cancellation-Safe Row Application

## Implementation

- Added `align_pp_frame_rows` to map validated fixed-frame rows by stable
  `pp_row_key` rather than scheduler row position.
- Active duplicate frame keys are rejected as a protocol error.
- `_pp_finish_receive_and_backfill` now aligns sampled tokens, draft tokens,
  and cursors after frame validation.
- Cancelled rows, inactive/missing frame rows, and newly-added local rows are
  represented by sentinel rows (`-1` tokens and cursor), so they cannot apply
  stale frame data.
- The local discard mask is consulted only after the collective frame has been
  received and validated; it does not affect collective participation.
- No V2 or attention code was changed.

## Tests

Added focused coverage for:

- Stable-key matching with reordered local rows.
- Cancelled rows, missing local rows, and new local rows.
- Duplicate active frame-key rejection.

`python -m py_compile` passed for all modified Python files.

The focused pytest command could not start because this checkout lacks the
compiled extension `vllm._C_stable_libtorch`:

```text
python -m pytest tests/v1/worker/test_pp_spec_broadcast.py -q
ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'
```

## Commit

Implementation commit: `ff1bdfce27` (`fix: make PP row backfill cancellation-safe`).
