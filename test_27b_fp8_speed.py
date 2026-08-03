import os
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/data/models/Qwen3.6-27B-AWQ",
        tensor_parallel_size=2,
        kv_cache_dtype="fp8",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        language_model_only=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    )
    # warmup
    llm.generate(["Hello"], SamplingParams(max_tokens=8, temperature=0))
    t0 = time.time()
    out = llm.generate(
        ["Write a very long story about a robot."],
        SamplingParams(max_tokens=512, temperature=0),
    )
    dt = time.time() - t0
    n_out = len(out[0].outputs[0].token_ids)
    print(f"DECODE_TOK={n_out} TIME={dt:.2f}s SPEED={n_out / dt:.2f} tok/s")


if __name__ == "__main__":
    main()
