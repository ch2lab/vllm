# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash draft must inherit the target's KV-cache dtype.

Regression: load_dflash_model used to clobber the worker's single
cache_config.cache_dtype with the draft's configured kv_cache_dtype (e.g.
nvfp4) so the draft's attention impl would bind it. But the FlashInfer
metadata builders resolve their op dtype from that same global cache_config
AFTER model load, so a target running fp8 KV cache with an nvfp4 draft got its
fp8 layers mis-planned as uint8 and crashed at cudagraph capture. A DFlash
draft reuses the target's KV cache (cross-attention reads the target's block
table), so it has no KV cache of its own and its dtype must match the target's.
"""

from types import SimpleNamespace

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode import dflash as dflash_module
from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model

_MODEL = "/data/models/Qwen3.5-0.8B"
_DRAFT_MODEL = "/data/models/Qwen3.8-27B-DFlash2-NVFP4"


def _dflash_speculative_config(vllm_config, kv_cache_dtype: str) -> SpeculativeConfig:
    """A DFlash SpeculativeConfig: the draft must be a model whose config.json
    constructs a plain ModelConfig (heterogeneous qwen3_5 configs cannot)."""
    return SpeculativeConfig(
        method="dflash",
        model=_DRAFT_MODEL,
        num_speculative_tokens=5,
        kv_cache_dtype=kv_cache_dtype,
        target_model_config=vllm_config.model_config,
        target_parallel_config=vllm_config.parallel_config,
    )


def _make_dummy_model():
    """A minimal target model the loader can walk without real weights."""
    return SimpleNamespace()


def _patch_dependencies(monkeypatch):
    """Stub out model loading; keep the loader's config handling real."""
    monkeypatch.setattr(
        dflash_module.utils,
        "get_model",
        lambda vllm_config, model_config: SimpleNamespace(model=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
        lambda config: False,
    )
    monkeypatch.setattr(
        dflash_module.utils,
        "maybe_share_target_embed",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        dflash_module.utils,
        "get_target_lm_head",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        dflash_module.utils,
        "_should_share",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        dflash_module.utils,
        "get_pp_safe_draft_load_config",
        lambda load_config: load_config,
    )


def test_load_dflash_model_keeps_target_cache_dtype(monkeypatch):
    """Loading a DFlash draft with a different configured kv_cache_dtype must
    not change the worker's global cache_config.cache_dtype."""
    _patch_dependencies(monkeypatch)
    vllm_config = create_vllm_config(model_name=_MODEL, max_model_len=1024)
    vllm_config.cache_config.cache_dtype = "fp8"
    vllm_config.speculative_config = _dflash_speculative_config(vllm_config, "nvfp4")

    load_dflash_model(_make_dummy_model(), vllm_config)

    assert vllm_config.cache_config.cache_dtype == "fp8"
