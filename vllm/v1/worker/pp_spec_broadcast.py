# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP sampled-token broadcast helpers for speculative decoding.

Under PP + async scheduling the last rank broadcasts one fixed-capacity frame to
the other ranks (which never run the sampler) so they can advance positions and
build the next input batch. The frame carries sampled tokens, cursors, drafts,
row keys, and active-row flags in one collective.

Kept in a CUDA-free module so the shape/transport logic is unit-testable over a
plain gloo CPU group.
"""

from dataclasses import dataclass
from datetime import timedelta
import zlib

import torch
import torch.distributed as dist


class PPProtocolError(RuntimeError):
    """Raised when PP ranks disagree on a fixed-frame protocol invariant."""


@dataclass
class PPReceiveFrame:
    """Fixed-shape CPU/GPU payload exchanged for one PP scheduling round."""

    generation: int
    row_keys: torch.Tensor
    row_flags: torch.Tensor
    cursors: torch.Tensor
    sampled_tokens: torch.Tensor
    draft_tokens: torch.Tensor

    def __eq__(self, other) -> bool:
        if not isinstance(other, PPReceiveFrame):
            return NotImplemented
        return self.generation == other.generation and all(
            torch.equal(a, b) for a, b in zip(self._tensors(), other._tensors())
        )

    def _tensors(self):
        return (self.row_keys, self.row_flags, self.cursors,
                self.sampled_tokens, self.draft_tokens)


def pp_frame_width(num_spec_tokens: int) -> int:
    return 4 + 2 * num_spec_tokens + 1


def pp_row_key(req_id: str) -> int:
    return zlib.crc32(req_id.encode("utf-8"))


def validate_pp_frame(
    frame: PPReceiveFrame,
    expected_generation: int,
    previous_generation: int,
) -> None:
    if frame.generation != expected_generation:
        raise PPProtocolError(
            f"PP frame generation {frame.generation} does not match "
            f"expected generation {expected_generation}"
        )
    if frame.generation <= previous_generation:
        raise PPProtocolError(
            f"PP frame generation {frame.generation} is not monotonic after "
            f"generation {previous_generation}"
        )
    if not torch.all((frame.row_flags == 0) | (frame.row_flags == 1)):
        raise PPProtocolError("PP frame row_flags must contain only 0 or 1")
    inactive = frame.row_flags == 0
    frame.sampled_tokens[inactive] = -1
    frame.draft_tokens[inactive] = -1


def wait_pp_work(work, generation: int, rank: int,
                 timeout_seconds: float = 30.0) -> None:
    """Wait for a PP collective with actionable timeout context."""
    try:
        completed = work.wait(timeout=timedelta(seconds=timeout_seconds))
    except Exception as exc:
        raise PPProtocolError(
            f"PP frame receive failed at generation {generation} on rank {rank}: "
            f"{exc}"
        ) from exc
    if completed is False:
        raise PPProtocolError(
            f"PP frame receive timed out at generation {generation} on rank {rank} "
            f"after {timeout_seconds}s"
        )


def next_pp_generation(previous: int, generation: int | None = None) -> int:
    """Return the next strictly monotonic frame generation."""
    candidate = previous + 1 if generation is None else generation
    if candidate <= previous:
        raise ValueError("PP frame generation must be monotonic")
    return candidate


def pack_pp_frame(
    frame: PPReceiveFrame,
    max_num_seqs: int,
    num_spec_tokens: int,
    previous_generation: int | None = None,
) -> torch.Tensor:
    """Pack a frame into one fixed ``[max_num_seqs, width]`` int32 tensor."""
    if previous_generation is not None:
        next_pp_generation(previous_generation, frame.generation)
    tensors = frame._tensors()
    device = tensors[0].device
    for name, tensor in zip(
        ("row_keys", "row_flags", "cursors", "sampled_tokens", "draft_tokens"),
        tensors,
    ):
        if tensor.device != device:
            raise ValueError(
                f"{name} must be on device {device}, got {tensor.device}"
            )
    if frame.row_keys.shape != (max_num_seqs,):
        raise ValueError("row_keys must have max_num_seqs entries")
    expected = {
        "row_flags": (max_num_seqs,),
        "cursors": (max_num_seqs,),
        "sampled_tokens": (max_num_seqs, num_spec_tokens + 1),
        "draft_tokens": (max_num_seqs, num_spec_tokens),
    }
    for name, shape in expected.items():
        if getattr(frame, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    generation = torch.full((max_num_seqs, 1), frame.generation,
                            dtype=torch.int32, device=frame.row_keys.device)
    return torch.cat(
        (
            generation,
            *(
                tensor.to(torch.int32).reshape(max_num_seqs, -1)
                for tensor in tensors
            ),
        ),
        dim=1,
    )


def unpack_pp_frame(packed: torch.Tensor, max_num_seqs: int,
                    num_spec_tokens: int) -> PPReceiveFrame:
    """Unpack and validate a fixed PP frame."""
    width = pp_frame_width(num_spec_tokens)
    if packed.shape != (max_num_seqs, width):
        raise ValueError(f"packed frame must have shape {(max_num_seqs, width)}")
    if not torch.equal(packed[:, 0], packed[0, 0].expand(max_num_seqs)):
        raise ValueError("frame generation must be constant")
    columns = iter((packed[:, 1], packed[:, 2], packed[:, 3]))
    sampled = packed[:, 4:5 + num_spec_tokens]
    draft = packed[:, 5 + num_spec_tokens:]
    return PPReceiveFrame(int(packed[0, 0].item()), next(columns), next(columns),
                          next(columns), sampled, draft)


def broadcast_pp_frame(
    frame: PPReceiveFrame,
    max_num_seqs: int,
    num_spec_tokens: int,
    group,
    src: int,
) -> None:
    """Broadcast one fixed-capacity PP frame, including inactive rows."""
    packed = pack_pp_frame(frame, max_num_seqs, num_spec_tokens)
    dist.broadcast(packed, src=src, group=group)


class PPReceiveRound:
    """One round of the async PP sampled/draft broadcast handoff.

    The non-last rank's ``sample_tokens`` (output-proc thread) builds a round --
    the receive buffer plus its in-flight ``Work`` handle -- and publishes it
    atomically; ``execute_model`` (driver thread) consumes it in
    ``_pp_finish_receive_and_backfill``. Publishing the whole round in a single
    attribute assignment (instead of publishing each ``Work`` handle separately)
    removes the partial-publication window where a consumer could wait on a stale
    handle while the new round's buffers are still being filled. Buffers are owned
    exclusively by this round: the consumer only reads them after the round's
    broadcast has been waited.
    """

    __slots__ = ("gen", "recv", "cursor", "draft", "recv_work", "cursor_work",
                 "draft_work")

    def __init__(
        self,
        gen: int,
        recv: torch.Tensor,
        cursor: torch.Tensor | None,
        draft: torch.Tensor | None,
        recv_work,
        cursor_work,
        draft_work,
    ) -> None:
        self.gen = gen
        self.recv = recv
        self.cursor = cursor
        self.draft = draft
        self.recv_work = recv_work
        self.cursor_work = cursor_work
        self.draft_work = draft_work


def count_valid_sampled_tokens_per_req(sampled_token_ids: torch.Tensor) -> torch.Tensor:
    """Per-request count of valid sampled tokens in a ``[num_reqs, width]`` grid.

    Valid tokens are the non-``-1`` entries (rejected/padded positions are ``-1``);
    this mirrors the accepted-count idiom used elsewhere in the runner
    (``(sampled_token_ids != -1).sum(dim=1)``). For the non-spec width-1 case every
    row holds one real token, so the count is 1 per request.
    """
    return (sampled_token_ids != -1).sum(dim=1)


def sanitize_token_zero_col(
    sampled_token_ids: torch.Tensor, width: int
) -> None:
    """Replace token-0 ('!') entries with the row's last valid non-0 token.

    Token-0 storms: a corrupted/garbage logits row makes the sampler emit 0
    (the '!' token); the 0 then enters the next step's input grid and
    reproduces, pinning the request in an all-'!' loop. The committed bonus
    is NOT always the last column: when every draft is rejected the grid is
    ``[bonus, -1, -1]`` (bonus at column 0), so a last-column-only sweep
    misses it. Sweep every column: legitimate text never contains token 0,
    so any 0 is a garbage-logits artifact. Replacing 0 with the row's last
    positive token (or -1 when none) breaks the loop.
    """
    if sampled_token_ids.numel() == 0:
        return
    # The grid width can be narrower than ``width`` on steps with no spec
    # tokens (sampled_token_ids is [num_reqs, 1]); always use the tensor's
    # actual width so the arange/expand below stays in range.
    width = sampled_token_ids.shape[-1]
    if width <= 1:
        return
    row_last_pos = torch.where(
        sampled_token_ids > 0,
        torch.arange(width, device=sampled_token_ids.device).expand_as(sampled_token_ids),
        sampled_token_ids.new_full((), -1),
    ).max(dim=1).values
    fallback = sampled_token_ids.gather(
        1, row_last_pos.clamp(min=0).unsqueeze(1)
    ).squeeze(1)
    fallback = torch.where(row_last_pos >= 0, fallback, fallback.new_full((), -1))
    replace = sampled_token_ids == 0
    if replace.any():
        sampled_token_ids[replace] = fallback.unsqueeze(1).expand(
            -1, width
        )[replace]


def select_latest_sampled_token_per_req(
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-request latest *valid* sampled token from a ``[num_reqs, width]`` grid.

    Rejected/padded positions are ``-1``; a request that advanced by ``v`` valid
    tokens has its latest token in the last valid column ``recv[i, v - 1]`` (reject
    ``v=1`` -> col 0; accept + bonus ``v=2`` -> col 1). This is the real value the
    non-last rank must persist as its next input instead of a ``-1`` placeholder
    (the C4 value-back-write that fixes the ``indexSelectSmallIndex`` break: every
    confirmed request has ``v >= 1`` so the result is never ``-1``). Pure-GPU ops
    only: this runs inside CUDA-graph capture, where CPU syncs deadlock.
    """
    counts = count_valid_sampled_tokens_per_req(sampled_token_ids)
    last_idx = (counts - 1).clamp(min=0)
    return sampled_token_ids.gather(1, last_idx.unsqueeze(1)).squeeze(1)


def gather_valid_sampled_tokens_per_req(
    sampled_token_ids: torch.Tensor,
) -> list[list[int]]:
    """Per-request list of ALL valid sampled tokens ``recv[i, 0:v]``, in order.

    ``select_latest_sampled_token_per_req`` keeps only the single latest token;
    the holistic C4 back-write needs every confirmed token (accepted drafts +
    bonus) so the non-last rank can fill the ``v`` ``token_ids_cpu`` positions the
    next step will read (indexed by ``num_computed_tokens``, which includes the
    spec tokens) — not just one slot. Valid entries are the leading non-``-1``
    columns; a fully padded row yields ``[]`` (advance the cursor by 0).
    """
    counts = count_valid_sampled_tokens_per_req(sampled_token_ids).tolist()
    rows = sampled_token_ids.tolist()
    return [row[:v] for row, v in zip(rows, counts)]


def num_computed_tokens_drift_correction(
    prev_num_draft_len: int, valid_sampled_count: int
) -> int:
    """Amount to subtract from the optimistic ``num_computed_tokens`` on a non-last
    rank to undo async spec-decode drift after a (partial) draft rejection.

    Async spec decode advances ``num_computed_tokens`` optimistically by
    ``1 (bonus) + prev_num_draft_len (drafts assumed accepted)``; the true advance is
    the broadcast ``valid_sampled_count`` (accepted drafts + the bonus). The
    difference is the number of optimistically-counted drafts that were actually
    rejected. On the last rank this correction is applied by the GPU kernel
    ``update_num_computed_tokens_for_batch_change`` from the sampler's valid count;
    the non-last rank never runs the sampler, so it reconstructs the same correction
    from the broadcast valid count instead — keeping rope/KV positions identical on
    every rank (the invariant: advance ``num_computed_tokens`` by the valid count).
    Non-negative (you cannot accept more drafts than were proposed); ``0`` when every
    draft was accepted or none were proposed.
    """
    return (1 + prev_num_draft_len) - valid_sampled_count


def broadcast_sampled_token_ids(
    sampled_token_ids: torch.Tensor, group, src: int
) -> None:
    """Broadcast the (possibly multi-column) sampled token ids from ``src``."""
    assert sampled_token_ids.dim() == 2, (
        f"expected 2-D [num_reqs, width], got {tuple(sampled_token_ids.shape)}"
    )
    dist.broadcast(sampled_token_ids, src=src, group=group)


def receive_sampled_token_ids(
    num_reqs: int,
    width: int,
    group,
    src: int,
    device,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Receive a ``[num_reqs, width]`` sampled-token grid broadcast from ``src``."""
    recv = torch.empty((num_reqs, width), dtype=dtype, device=device)
    dist.broadcast(recv, src=src, group=group)
    return recv
