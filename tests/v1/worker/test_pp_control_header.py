# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.worker.pp_spec_broadcast import (
    control_header_enabled,
    make_pp_control_header,
)


def test_control_header_defaults_to_payload_present(monkeypatch):
    monkeypatch.delenv("PP_USE_CONTROL_HEADER", raising=False)
    assert control_header_enabled()
    assert make_pp_control_header(True).tolist() == [[1]]


def test_control_header_can_disable_new_protocol(monkeypatch):
    monkeypatch.setenv("PP_USE_CONTROL_HEADER", "0")
    assert not control_header_enabled()
    assert make_pp_control_header(False).dtype == torch.int32
