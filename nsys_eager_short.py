import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/data/models/Qwen3.6-27B-AWQ",
        tensor_parallel_size=2,
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        language_model_only=True,
        enforce_eager=True,
    )
    llm.generate(["warmup warmup warmup"], SamplingParams(max_tokens=8, temperature=0))
    llm.generate(
        ["Explain the theory of general relativity."],
        SamplingParams(max_tokens=24, temperature=0),
    )
    print("DONE")


if __name__ == "__main__":
    main()
