"""Test INT4 KV-cache on Qwen3.5-4B."""
import os, gc, time
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"


def main():
    import torch
    from vllm import LLM, SamplingParams

    model = "/data/models/Qwen3.5-4B"
    prompt = "The capital of France is"

    print(f"Testing INT4 KV-cache on {model}", flush=True)
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="int4_per_token_head",
        enforce_eager=True,
    )
    out = llm.generate([prompt], SamplingParams(max_tokens=32, temperature=0))
    text = out[0].outputs[0].text
    print(f"OUTPUT: {text!r}")
    ok = "paris" in text.lower()
    print(f"STATUS: {'PASS' if ok else 'FAIL'}")
    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
