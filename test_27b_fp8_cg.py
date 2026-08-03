import os

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
    prompt = (
        "Explain the theory of relativity in detail, covering special "
        "relativity, general relativity, spacetime curvature, gravitational "
        "waves, and the experimental evidence for each. "
    ) * 4
    out = llm.generate(
        [prompt],
        SamplingParams(max_tokens=256, temperature=0),
    )
    text = out[0].outputs[0].text
    n_in = len(out[0].prompt_token_ids)
    n_out = len(out[0].outputs[0].token_ids)
    print(f"INPUT_TOKENS={n_in} OUTPUT_TOKENS={n_out}")
    print("OUTPUT:", repr(text[:400]))


if __name__ == "__main__":
    main()
