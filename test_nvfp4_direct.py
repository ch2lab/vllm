"""Test NVFP4 KV cache correctness with Qwen3.5-4B using LLM API directly."""
import os, sys, logging

os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"


def main():
    logging.basicConfig(level=logging.WARNING)
    from vllm import LLM, SamplingParams

    print("=== Loading model with NVFP4 KV cache ===", flush=True)
    llm = LLM(
        model='/data/models/Qwen3.5-4B',
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="nvfp4",
        enforce_eager=True,
    )

    print("\n=== Generating ===", flush=True)
    out = llm.generate(
        ['The capital of France is'],
        SamplingParams(max_tokens=32, temperature=0),
    )
    text = out[0].outputs[0].text
    print(f"\nOUTPUT: {text!r}")
    print(f"TOKENS: {len(out[0].outputs[0].token_ids)}")

    expected_words = ["paris", "Paris"]
    if any(w in text for w in expected_words):
        print("STATUS: CORRECT (mentions Paris)")
    else:
        print("STATUS: GARBAGE (does not mention Paris)")


if __name__ == '__main__':
    main()
