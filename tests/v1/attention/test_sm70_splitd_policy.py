# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Admission-predicate table for the SM70 splitd paged prefill route."""

import torch

from vllm.envs import VLLM_SM70_SPLITD_PREFILL
from vllm.v1.attention.ops.sm70_splitd_policy import sm70_splitd_admit


def _admit(q_len, kv_len, causal=True, dtype=torch.float16, n=1):
    return sm70_splitd_admit(n, q_len, kv_len, causal, dtype)


def test_admit_long_context():
    assert _admit(4096, 53792)
    assert _admit(4096, 262144)
    assert _admit(64, 3136)
    assert _admit(8192, 262144)
    # q_len == kv_len (no cached prefix yet) is eligible
    assert _admit(1024, 1024)


def test_reject_multi_request():
    assert not sm70_splitd_admit(2, 4096, 53792, True, torch.float16)


def test_reject_q_below_min_or_unaligned():
    assert not _admit(32, 3136)
    assert not _admit(100, 3136)
    assert not _admit(1568, 3136)  # 1568 % 64 != 0


def test_reject_q_above_max():
    assert not _admit(8256, 131072)


def test_reject_kv_out_of_range():
    assert not _admit(4096, 0)
    assert not _admit(4096, 262145)
    assert not _admit(4096, 131071)  # not a multiple of 32
    assert not _admit(1664, 4849)


def test_reject_q_longer_than_kv():
    assert not _admit(2048, 1024)


def test_reject_non_causal_and_bf16():
    assert not _admit(4096, 53792, causal=False)
    assert not _admit(4096, 53792, dtype=torch.bfloat16)


def test_gate_defaults_off():
    assert VLLM_SM70_SPLITD_PREFILL is False
