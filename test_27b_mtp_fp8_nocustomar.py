import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/data/models/Qwen3.6-27B-AWQ",
        tensor_parallel_size=2,
        kv_cache_dtype="fp8",
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        language_model_only=True,
        disable_custom_all_reduce=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 4,
        },
    )
    llm.generate(["warmup"], SamplingParams(max_tokens=16, temperature=0))
    prompt = (
        "Explain the theory of general relativity in detail, covering "
        "spacetime curvature and the experimental evidence for it."
    )
    t0 = time.time()
    out = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0))
    dt = time.time() - t0
    n = len(out[0].outputs[0].token_ids)
    print(f"MTP4_FP8_NOCUSTOMAR: {n} tokens in {dt:.2f}s = {n / dt:.1f} tok/s")
    print("OUTPUT:", repr(out[0].outputs[0].text[:300]))


if __name__ == "__main__":
    main()
