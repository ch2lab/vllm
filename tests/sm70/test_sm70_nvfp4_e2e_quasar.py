# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone end-to-end test: load the QUASAR NVFP4 Qwen3.5 model on a V100
(SM70) through the new SM70 NVFP4 GEMM path and verify it generates coherent
output.

This is the e2e validation for the SM70 (V100) NVFP4 GEMM rebuild:
  * Task 4 wired the SM70 NVFP4 linear kernel into dispatch;
  * Task 5 lowered the W4A4-NVFP4 capability gate so SM70 can use the scheme;
  * the kernel magnitude fix (global_scale * 2^22) makes output correctly scaled.

The target is a real ``nvfp4-pack-quantized`` W4A4 checkpoint
(``Qwen3_5ForConditionalGeneration``, dense LLM + GDN hybrid attention + a
vision tower). Without ``VLLM_SM70_NVFP4=1`` the W4A4 NVFP4 scheme's
``get_min_capability()`` is 75, so exact SM70 (7.0) would reject it and the
upstream NVFP4 kernels (SM90+/SM100) would not dispatch. With the env switch,
``Sm70MarlinNvFp4LinearKernel`` becomes the first W4A4 kernel to pass
``is_supported()`` and is selected for every NVFP4 Linear layer.

The script is standalone (no pytest). It:
  1. Prints whether the SM70 NVFP4 gate is open (``is_enabled`` /
     ``is_supported`` / env / registered op) -- the in-process "is the SM70
     path active" signal.
  2. Loads the model (GPU0, TP1 by default, modest gpu_memory_utilization).
  3. Greedy-generates from a simple prompt and prints the text + finish reason.
     Coherence of the text is judged by a human / agent, not here.
  4. Best-effort greps the captured log for the two kernel-selection markers
     (the worker subprocess may not have flushed yet at read time; the
     authoritative grep is done after the process exits).

Run (long task -- model load takes 1-3 min):
    cd /data/src/vllm && CUDA_VISIBLE_DEVICES=0 VLLM_SM70_NVFP4=1 \
        nohup python3 tests/sm70/test_sm70_nvfp4_e2e_quasar.py \
        > /tmp/opencode/sm70_nvfp4_quasar.log 2>&1 &
    tail -f /tmp/opencode/sm70_nvfp4_quasar.log
"""

import os
import sys

MODEL = os.environ.get("SM70_NVFP4_MODEL", "/data/models/Qwen3.8-27B-QUASAR-NVFP4")
LOG_PATH = os.environ.get("SM70_NVFP4_LOG", "/tmp/opencode/sm70_nvfp4_quasar.log")
PROMPT = os.environ.get(
    "SM70_NVFP4_PROMPT",
    "Explain in one sentence why the sky appears blue.",
)
MAX_TOKENS = int(os.environ.get("SM70_NVFP4_MAX_TOKENS", "96"))
TP = int(os.environ.get("SM70_NVFP4_TP", "1"))
MAX_MODEL_LEN = int(os.environ.get("SM70_NVFP4_MAX_MODEL_LEN", "4096"))
GPU_MEM_UTIL = float(os.environ.get("SM70_NVFP4_GPU_MEM_UTIL", "0.90"))
# SM70 (V100) full-attention prefill is f16-WMMA-only (no bf16 tensor-core mma);
# 1Cat's production SM70 route is likewise fp16. Default to fp16 so the full-
# attention path is usable on V100; override with SM70_NVFP4_DTYPE.
DTYPE = os.environ.get("SM70_NVFP4_DTYPE", "float16")
# Eager keeps this a focused correctness smoke test (no CUDA-graph capture
# variable). The NVFP4 GEMM / attention kernels run the same code either way.
ENFORCE_EAGER = os.environ.get("SM70_NVFP4_EAGER", "1") == "1"


def main() -> int:
    print("=" * 76)
    print("SM70 NVFP4 e2e (QUASAR Qwen3.5) -- load + generate on V100 (SM70)")
    print("=" * 76)
    print(f"model          : {MODEL}")
    print(f"VLLM_SM70_NVFP4 : {os.environ.get('VLLM_SM70_NVFP4', '<unset>')}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(
        f"TP={TP} max_tokens={MAX_TOKENS} max_model_len={MAX_MODEL_LEN} "
        f"gpu_mem_util={GPU_MEM_UTIL} enforce_eager={ENFORCE_EAGER}"
    )

    import torch

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device available.")
        return 2
    cc = tuple(torch.cuda.get_device_capability())
    print(f"device0        : {torch.cuda.get_device_name(0)} "
          f"(sm{cc[0]}.{cc[1]})")

    # Load the stable C extension in THIS process so the gate check below
    # reflects real hardware + registered op (it registers
    # torch.ops._C.sm70_nvfp4_marlin_gemm). The workers load it too; this just
    # lets the main process report the gate state deterministically.
    try:
        import vllm._C_stable_libtorch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"note: could not import vllm._C_stable_libtorch in main proc: {exc}")

    from vllm.sm70_nvfp4_ops import is_enabled

    sm70_supported = False
    try:
        from vllm.model_executor.kernels.linear.nvfp4.sm70 import (
            Sm70MarlinNvFp4LinearKernel,
        )
        sm70_supported, _reason = Sm70MarlinNvFp4LinearKernel.is_supported()
        print(
            f"SM70 NVFP4 gate  : is_enabled()={is_enabled()} "
            f"is_supported()={sm70_supported}"
        )
        if not sm70_supported:
            print(
                "WARNING: SM70 NVFP4 kernel is NOT the selected kernel here "
                "(gate closed). The model will not use the SM70 GEMM path."
            )
    except Exception as exc:
        print(f"note: kernel is_supported() check failed: {exc}")

    from vllm import LLM, SamplingParams

    print("-" * 76)
    print("Loading model ... [model load takes 1-3 min; be patient]")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,
        enforce_eager=ENFORCE_EAGER,
        seed=0,
    )
    print("Model loaded OK.")

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS)
    outs = llm.generate([PROMPT], sp)
    out = outs[0].outputs[0]
    text = out.text
    finish_reason = out.finish_reason
    n_tokens = len(out.token_ids)

    print("-" * 76)
    print(f"finish_reason = {finish_reason}   n_tokens = {n_tokens}")
    print("PROMPT: " + PROMPT)
    print("OUTPUT:")
    print(text)
    print("-" * 76)

    # Best-effort: did the captured log already record the SM70 kernel selection?
    # (The selection / weight-processing lines are emitted by the worker
    # subprocess, whose buffers may not have flushed to the file yet at this
    # point. The authoritative grep is done after the process exits.)
    sys.stdout.flush()
    sys.stderr.flush()
    print("Kernel-selection evidence (best-effort grep of captured log):")
    markers = {
        "selected": "Using Sm70MarlinNvFp4LinearKernel for NVFP4 GEMM",
        "weights": "Using SM70 NVFP4 GEMM",
    }
    try:
        with open(LOG_PATH, "r", errors="ignore") as f:
            blob = f.read()
        print(f"  log file : {LOG_PATH} (read {len(blob)} chars)")
        for key, marker in markers.items():
            hit = marker in blob
            print(f"  [{key:9s}] found={hit}  << {marker} >>")
    except Exception as exc:
        print(f"  (could not grep the log yet: {exc})")

    print("-" * 76)
    active = bool(is_enabled() and sm70_supported)
    print(f"SM70 NVFP4 path active (in-process gate): {active}")
    print(
        f"RESULT: loaded=yes generated=yes finish_reason={finish_reason} "
        f"n_tokens={n_tokens}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
