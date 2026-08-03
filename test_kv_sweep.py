"""AWQ correctness sweep across KV-cache dtypes (turbomind vs wmma).

Usage: VLLM_SM70_AWQ_BACKEND=<turbomind|wmma> KVDT=<dtype> python3 test_kv_sweep.py
Prints greedy output (coherence) + tok/s for one KV-cache dtype.
"""
import os, time

os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
KVDT = os.environ.get("KVDT", "nvfp4")
PROMPT = "Explain the theory of general relativity in detail."

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="/data/models/Qwen3.6-27B-AWQ",
        tensor_parallel_size=2,
        max_model_len=2048,
        gpu_memory_utilization=0.92,
        kv_cache_dtype=KVDT,
        speculative_config={"method": "mtp", "num_speculative_tokens": 4},
    )
    for _ in range(3):
        llm.generate(["warmup"], SamplingParams(max_tokens=16, temperature=0))
    start = time.time()
    out = llm.generate([PROMPT], SamplingParams(max_tokens=96, temperature=0))
    elapsed = time.time() - start
    n = len(out[0].outputs[0].token_ids)
    text = out[0].outputs[0].text
    snippet = " ".join(text.split())[:300]
    print(f"KVDT={KVDT} BACKEND={os.environ.get('VLLM_SM70_AWQ_BACKEND','wmma')}")
    print(f"PERF: {n} tokens in {elapsed:.2f}s = {n/elapsed:.1f} tok/s")
    print(f"OUTPUT>>> {snippet} <<<END")
