import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/data/models/Qwen3.5-0.8B",
        tensor_parallel_size=1,
        kv_cache_dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        language_model_only=True,
        max_num_batched_tokens=1024,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    )
    prompt = (
        "The history of computing spans many decades, from early mechanical "
        "calculators to modern neural networks. "
    ) * 40
    out = llm.generate(
        [prompt],
        SamplingParams(max_tokens=128, temperature=0),
    )
    n_in = len(out[0].prompt_token_ids)
    n_out = len(out[0].outputs[0].token_ids)
    print(f"LONG_FP16 INPUT_TOKENS={n_in} OUTPUT_TOKENS={n_out}")
    print("OUTPUT:", repr(out[0].outputs[0].text[:300]))


if __name__ == "__main__":
    main()
