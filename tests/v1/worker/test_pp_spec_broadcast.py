# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import pytest
import torch

from vllm.v1.worker.pp_spec_broadcast import (
    PPReceiveFrame,
    PPProtocolError,
    align_pp_frame_rows,
    broadcast_pp_frame,
    next_pp_generation,
    pack_pp_frame,
    pp_row_key,
    unpack_pp_frame,
    validate_pp_frame,
    wait_pp_work,
)


def test_chunked_sender_and_non_chunked_receiver_use_one_collective(monkeypatch):
    calls = []

    def broadcast(tensor, src, group, async_op=False):
        calls.append((tuple(tensor.shape), async_op))
        return tensor

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    frame = PPReceiveFrame(
        generation=1,
        row_keys=torch.zeros(4, dtype=torch.int32),
        row_flags=torch.zeros(4, dtype=torch.int32),
        cursors=torch.zeros(4, dtype=torch.int32),
        sampled_tokens=torch.full((4, 3), -1, dtype=torch.int32),
        draft_tokens=torch.full((4, 2), -1, dtype=torch.int32),
    )

    broadcast_pp_frame(frame, max_num_seqs=4, num_spec_tokens=2,
                       group=None, src=1)

    assert calls == [((4, 9), False)]


def test_pp_frame_has_fixed_shape_for_runner_limits():
    frame = PPReceiveFrame(
        generation=3,
        row_keys=torch.arange(4),
        row_flags=torch.tensor([1, 0, 1, 0]),
        cursors=torch.tensor([2, 0, 1, 0]),
        sampled_tokens=torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]]),
        draft_tokens=torch.tensor([[100, 101], [200, 201], [300, 301], [400, 401]]),
    )

    packed = pack_pp_frame(frame, max_num_seqs=4, num_spec_tokens=2)

    assert packed.shape == (4, 9)
    assert unpack_pp_frame(packed, max_num_seqs=4, num_spec_tokens=2) == frame


def test_pp_generation_must_increase():
    assert next_pp_generation(7) == 8
    with pytest.raises(ValueError, match="monotonic"):
        next_pp_generation(7, generation=7)


def test_pack_rejects_non_monotonic_frame_generation():
    frame = PPReceiveFrame(
        generation=7,
        row_keys=torch.zeros(2),
        row_flags=torch.zeros(2),
        cursors=torch.zeros(2),
        sampled_tokens=torch.zeros(2, 2),
        draft_tokens=torch.zeros(2, 1),
    )
    with pytest.raises(ValueError, match="monotonic"):
        pack_pp_frame(frame, max_num_seqs=2, num_spec_tokens=1,
                      previous_generation=7)


def test_pack_rejects_malformed_shape_and_mixed_devices():
    frame = PPReceiveFrame(
        generation=1,
        row_keys=torch.zeros(2),
        row_flags=torch.zeros(2),
        cursors=torch.zeros(2),
        sampled_tokens=torch.zeros(2, 1),
        draft_tokens=torch.zeros(2, 1),
    )
    with pytest.raises(ValueError, match="sampled_tokens.*shape"):
        pack_pp_frame(frame, max_num_seqs=2, num_spec_tokens=1)

    mixed = PPReceiveFrame(
        generation=1,
        row_keys=torch.zeros(2),
        row_flags=torch.zeros(2),
        cursors=torch.zeros(2),
        sampled_tokens=torch.zeros(2, 2),
        draft_tokens=torch.zeros(2, 1, device="meta"),
    )
    with pytest.raises(ValueError, match="draft_tokens.*device"):
        pack_pp_frame(mixed, max_num_seqs=2, num_spec_tokens=1)


def test_receiver_rejects_repeated_or_out_of_order_generation():
    frame = PPReceiveFrame(
        generation=4,
        row_keys=torch.tensor([1, 2]),
        row_flags=torch.tensor([1, 0]),
        cursors=torch.tensor([3, 0]),
        sampled_tokens=torch.tensor([[10, 11], [-1, -1]]),
        draft_tokens=torch.tensor([[20], [-1]]),
    )
    last_received_generation = 3
    validate_pp_frame(frame, expected_generation=4,
                      previous_generation=last_received_generation)
    last_received_generation = frame.generation
    with pytest.raises(PPProtocolError, match="generation"):
        validate_pp_frame(frame, expected_generation=4,
                          previous_generation=last_received_generation)
    with pytest.raises(PPProtocolError, match="monotonic"):
        validate_pp_frame(
            PPReceiveFrame(3, frame.row_keys, frame.row_flags, frame.cursors,
                           frame.sampled_tokens, frame.draft_tokens),
            expected_generation=3, previous_generation=last_received_generation)


def test_inactive_rows_are_cleared_and_flags_are_validated():
    frame = PPReceiveFrame(
        generation=1,
        row_keys=torch.tensor([1, 0]),
        row_flags=torch.tensor([1, 0]),
        cursors=torch.tensor([2, 99]),
        sampled_tokens=torch.tensor([[10, 11], [20, 21]]),
        draft_tokens=torch.tensor([[30], [40]]),
    )
    validate_pp_frame(frame, expected_generation=1, previous_generation=0)
    assert frame.sampled_tokens[1].tolist() == [-1, -1]
    assert frame.draft_tokens[1].tolist() == [-1]
    with pytest.raises(PPProtocolError, match="row_flags"):
        validate_pp_frame(
            PPReceiveFrame(1, torch.tensor([1, 2]), torch.tensor([2, 0]),
                           torch.zeros(2), torch.zeros(2, 2), torch.zeros(2, 1)),
            expected_generation=1, previous_generation=0)


def test_frame_rows_match_stable_keys_when_local_rows_reordered():
    frame = PPReceiveFrame(
        1, torch.tensor([pp_row_key("a"), pp_row_key("b"), 0]),
        torch.tensor([1, 1, 0]), torch.zeros(3), torch.zeros(3, 2),
        torch.zeros(3, 1))
    assert align_pp_frame_rows(frame, ["b", "a"]) == [1, 0]


def test_cancelled_missing_and_new_local_rows_are_unmatched():
    frame = PPReceiveFrame(
        1, torch.tensor([pp_row_key("old"), pp_row_key("cancelled"), 0]),
        torch.tensor([1, 1, 0]), torch.zeros(3), torch.zeros(3, 2),
        torch.zeros(3, 1))
    assert align_pp_frame_rows(
        frame, ["new", "cancelled", "old"], {1}
    ) == [-1, -1, 0]


def test_duplicate_active_frame_keys_are_rejected():
    key = pp_row_key("duplicate")
    frame = PPReceiveFrame(
        1, torch.tensor([key, key]), torch.ones(2), torch.zeros(2),
        torch.zeros(2, 2), torch.zeros(2, 1))
    with pytest.raises(PPProtocolError, match="duplicate"):
        align_pp_frame_rows(frame, ["duplicate"])


def test_pp_work_timeout_has_protocol_context():
    waits = []

    class NeverReady:
        def wait(self, timeout):
            waits.append(timeout)
            return False

    with pytest.raises(
        PPProtocolError,
        match=r"generation 7.*rank 1.*expected shape \(4, 9\).*received shape \(2, 9\)",
    ):
        wait_pp_work(
            NeverReady(), generation=7, rank=1, timeout_seconds=30,
            expected_shape=(4, 9), received_shape=(2, 9),
        )
    assert waits == [timedelta(seconds=30)]
