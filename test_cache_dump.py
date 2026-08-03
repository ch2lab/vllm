import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from vllm import LLM, SamplingParams

PROMPT = (
    "The history of computing spans many decades, from early mechanical "
    "calculators to modern neural networks. "
) * 40


def main():
    import sys

    kv_dtype = sys.argv[1] if len(sys.argv) > 1 else "fp8"
    llm = LLM(
        model="/data/models/Qwen3.5-0.8B",
        tensor_parallel_size=1,
        kv_cache_dtype=kv_dtype,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        language_model_only=True,
        enforce_eager=True,
    )
    out = llm.generate([PROMPT], SamplingParams(max_tokens=16, temperature=0))
    print("OUTPUT:", repr(out[0].outputs[0].text[:120]))
    runner = (
        llm.llm_engine.engine_core.model_executor.driver_worker.model_runner
    )
    caches = runner.kv_caches
    print(f"== {kv_dtype}: {len(caches)} kv caches ==")
    for i, c in enumerate(caches):
        if c is None:
            print(i, "None")
        elif c.numel() == 0:
            print(i, "empty")
        else:
            print(i, tuple(c.shape), c.dtype)
    torch.save(caches, f"/data/src/vllm/cache_dump_{kv_dtype}.pt")
    print("saved")


if __name__ == "__main__":
    main()
