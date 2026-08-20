# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import pytest
import torch

from vllm.v1.worker.pp_spec_broadcast import (
    PPReceiveFrame,
    PPProtocolError,
    align_pp_frame_rows,
    broadcast_pp_frame,
    ensure_pp_generation_not_fenced,
    next_pp_generation,
    pack_pp_frame,
    pp_row_key,
    unpack_pp_frame,
    validate_pp_frame,
    wait_pp_work,
    terminate_pp_round,
    terminate_fenced_pp_round,
    PPReceiveRound,
)


def keys(*req_ids):
    return torch.tensor([pp_row_key(req_id) if req_id else (0,) * 16
                         for req_id in req_ids], dtype=torch.int32)


def test_chunked_sender_and_non_chunked_receiver_use_one_collective(monkeypatch):
    calls = []

    def broadcast(tensor, src, group, async_op=False):
        calls.append((tuple(tensor.shape), async_op))
        return tensor

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    frame = PPReceiveFrame(
        generation=1,
        row_keys=keys("", "", "", ""),
        row_flags=torch.zeros(4, dtype=torch.int32),
        cursors=torch.zeros(4, dtype=torch.int32),
        sampled_tokens=torch.full((4, 3), -1, dtype=torch.int32),
        draft_tokens=torch.full((4, 2), -1, dtype=torch.int32),
    )

    broadcast_pp_frame(frame, max_num_seqs=4, num_spec_tokens=2,
                       group=None, src=1)

    assert calls == [((4, 24), False)]


def test_pp_frame_has_fixed_shape_for_runner_limits():
    frame = PPReceiveFrame(
        generation=3,
        row_keys=keys("a", "", "b", ""),
        row_flags=torch.tensor([1, 0, 1, 0]),
        cursors=torch.tensor([2, 0, 1, 0]),
        sampled_tokens=torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]]),
        draft_tokens=torch.tensor([[100, 101], [200, 201], [300, 301], [400, 401]]),
    )

    packed = pack_pp_frame(frame, max_num_seqs=4, num_spec_tokens=2)

    assert packed.shape == (4, 24)
    assert unpack_pp_frame(packed, max_num_seqs=4, num_spec_tokens=2) == frame


def test_pp_generation_must_increase():
    assert next_pp_generation(7) == 8
    with pytest.raises(ValueError, match="monotonic"):
        next_pp_generation(7, generation=7)


def test_pack_rejects_non_monotonic_frame_generation():
    frame = PPReceiveFrame(
        generation=7,
        row_keys=keys("", ""),
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
        row_keys=keys("", ""),
        row_flags=torch.zeros(2),
        cursors=torch.zeros(2),
        sampled_tokens=torch.zeros(2, 1),
        draft_tokens=torch.zeros(2, 1),
    )
    with pytest.raises(ValueError, match="sampled_tokens.*shape"):
        pack_pp_frame(frame, max_num_seqs=2, num_spec_tokens=1)

    mixed = PPReceiveFrame(
        generation=1,
        row_keys=keys("", ""),
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
        row_keys=keys("a", ""),
        row_flags=torch.tensor([1, 0], dtype=torch.int32),
        cursors=torch.tensor([3, 0], dtype=torch.int32),
        sampled_tokens=torch.tensor([[10, 11], [-1, -1]], dtype=torch.int32),
        draft_tokens=torch.tensor([[20], [-1]], dtype=torch.int32),
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
        row_keys=keys("a", ""),
        row_flags=torch.tensor([1, 0], dtype=torch.int32),
        cursors=torch.tensor([2, 99], dtype=torch.int32),
        sampled_tokens=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        draft_tokens=torch.tensor([[30], [40]], dtype=torch.int32),
    )
    with pytest.raises(PPProtocolError, match="inactive"):
        validate_pp_frame(frame, expected_generation=1, previous_generation=0)
    with pytest.raises(PPProtocolError, match="row_flags"):
        validate_pp_frame(
            PPReceiveFrame(1, keys("a", "b"), torch.tensor([2, 0], dtype=torch.int32),
                           torch.zeros(2, dtype=torch.int32), torch.zeros(2, 2, dtype=torch.int32), torch.zeros(2, 1, dtype=torch.int32)),
            expected_generation=1, previous_generation=0)


def test_frame_rows_match_stable_keys_when_local_rows_reordered():
    frame = PPReceiveFrame(
        1, keys("a", "b", ""),
        torch.tensor([1, 1, 0], dtype=torch.int32), torch.zeros(3, dtype=torch.int32),
        torch.zeros(3, 2, dtype=torch.int32), torch.zeros(3, 1, dtype=torch.int32))
    assert align_pp_frame_rows(frame, ["b", "a"]) == [1, 0]


def test_cancelled_missing_and_new_local_rows_are_unmatched():
    frame = PPReceiveFrame(
        1, keys("old", "cancelled", ""),
        torch.tensor([1, 1, 0], dtype=torch.int32), torch.zeros(3, dtype=torch.int32),
        torch.zeros(3, 2, dtype=torch.int32), torch.zeros(3, 1, dtype=torch.int32))
    assert align_pp_frame_rows(
        frame, ["new", "cancelled", "old"], {1}
    ) == [-1, -1, 0]


def test_duplicate_active_frame_keys_are_rejected():
    key = pp_row_key("duplicate")
    frame = PPReceiveFrame(
        1, torch.tensor([key, key], dtype=torch.int32), torch.ones(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, 2, dtype=torch.int32), torch.zeros(2, 1, dtype=torch.int32))
    with pytest.raises(PPProtocolError, match="duplicate"):
        align_pp_frame_rows(frame, ["duplicate"])


def test_row_keys_are_fixed_width_and_collision_safe():
    first = pp_row_key("a")
    second = pp_row_key("b")
    assert first != second
    assert len(first) == 16
    with pytest.raises(ValueError, match="too long"):
        pp_row_key("x" * 65)


def test_frame_validation_rejects_invalid_shapes_devices_and_values():
    frame = PPReceiveFrame(
        1, torch.zeros(2, 16, dtype=torch.int32), torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([0, -1], dtype=torch.int32), torch.tensor([[2, -1], [3, -1]], dtype=torch.int32),
        torch.tensor([[4], [5]], dtype=torch.int32),
    )
    with pytest.raises(PPProtocolError, match="cursors"):
        validate_pp_frame(frame, 1, 0, max_num_seqs=2, num_spec_tokens=1)

    frame.cursors[1] = 0
    frame.sampled_tokens[1, 1] = 7
    with pytest.raises(PPProtocolError, match="inactive"):
        validate_pp_frame(frame, 1, 0, max_num_seqs=2, num_spec_tokens=1)


def test_inactive_frame_rows_cannot_align_or_supply_cursor():
    key = pp_row_key("inactive")
    frame = PPReceiveFrame(
        1, torch.zeros(2, 16, dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32), torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([[-1, -1], [-1, -1]], dtype=torch.int32), torch.tensor([[-1], [-1]], dtype=torch.int32),
    )
    validate_pp_frame(frame, 1, 0, max_num_seqs=2, num_spec_tokens=1)
    assert align_pp_frame_rows(frame, ["inactive"]) == [-1]


def test_row_alignment_normalizes_key_encoding_errors():
    frame = PPReceiveFrame(
        1, torch.zeros(1, 16, dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32), torch.tensor([0], dtype=torch.int32),
        torch.full((1, 2), -1, dtype=torch.int32),
        torch.full((1, 1), -1, dtype=torch.int32),
    )
    with pytest.raises(PPProtocolError, match="row key"):
        align_pp_frame_rows(frame, ["x" * 65])


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


def test_pp_work_timeout_aborts_work():
    class Work:
        def __init__(self):
            self.aborted = False

        def wait(self, timeout):
            return False

        def abort(self):
            self.aborted = True

    work = Work()
    with pytest.raises(PPProtocolError):
        wait_pp_work(work, generation=1, rank=0, timeout_seconds=0)
    assert work.aborted


def test_pp_work_exception_aborts_work():
    class Work:
        def __init__(self):
            self.aborted = False

        def wait(self, timeout):
            raise RuntimeError("transport failed")

        def abort(self):
            self.aborted = True

    work = Work()
    with pytest.raises(PPProtocolError, match="transport failed"):
        wait_pp_work(work, generation=1, rank=0, timeout_seconds=0)
    assert work.aborted


def test_timeout_fences_stale_generation():
    with pytest.raises(PPProtocolError, match="fenced"):
        ensure_pp_generation_not_fenced(7, 7)
    ensure_pp_generation_not_fenced(8, 7)


def test_validation_failure_terminates_round():
    round = PPReceiveRound(1, torch.empty(1, 1), None, None, None, None, None)
    terminate_pp_round(round)
    assert round.terminated


def test_fenced_receive_terminates_pending_round():
    round = PPReceiveRound(7, torch.empty(1, 1), None, None, None, None, None)
    with pytest.raises(PPProtocolError, match="fenced"):
        terminate_fenced_pp_round(round, 7, 7)
    assert round.terminated
