import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/data/models/Qwen3.5-0.8B",
        tensor_parallel_size=1,
        kv_cache_dtype="fp8",
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        language_model_only=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    )
    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=32, temperature=0),
    )
    print("OUTPUT:", repr(out[0].outputs[0].text))


if __name__ == "__main__":
    main()
