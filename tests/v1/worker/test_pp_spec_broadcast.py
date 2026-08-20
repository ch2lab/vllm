# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.pp_spec_broadcast import (
    PPReceiveFrame,
    next_pp_generation,
    pack_pp_frame,
    unpack_pp_frame,
)


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
