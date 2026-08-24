# V1 PP+MTP Fixed-Frame Broadcast

## Problem

The custom V1 PP+MTP path conditionally skips three broadcasts using each
rank's local discard mask. Request cancellation can make those masks diverge,
leaving one rank in NCCL while the other skips it. The three independent
payloads also depend on local batch shapes and are published independently.

## Design

Each PP step performs exactly one broadcast from the final PP rank. The
broadcast uses a fixed-size contiguous frame sized by `max_num_seqs` and the
configured speculative width. The frame contains a header, per-row stable
request keys, row flags, cursors, sampled tokens, and draft tokens.

The final PP rank is authoritative for row validity and chunked-prefill flags.
The receiving rank never locally decides whether to participate. It matches
rows by stable request key rather than row position and applies placeholders
for unmatched or discarded rows.

The receiver owns one async Work handle per frame. Frames carry a monotonically
increasing generation. Generation, dimensions, and request-key mismatches are
reported as protocol errors. Work waits are bounded so a failed collective
does not leave a worker spinning indefinitely.

## Scope

This change preserves the existing sampled/cursor/draft payload semantics and
targets the custom V1 PP+MTP path. It does not change the V2 runner, attention
backend, KV layout, or FlashInfer kernels.

## Verification

Verify ordinary decode, chunked prefill, request cancellation, batch reshuffle,
concurrent PP2+MTP2 requests, and recovery after cancellation. Confirm that
the service remains responsive and that no rank diverges in frame generation.
