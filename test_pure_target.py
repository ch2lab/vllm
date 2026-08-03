"""Pure target decode (no MTP) with turbomind, to isolate target cost."""
import os, time

os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="/data/models/Qwen3.6-27B-AWQ",
        tensor_parallel_size=2,
        max_model_len=2048,
        gpu_memory_utilization=0.92,
    )
    for _ in range(5):
        llm.generate(["warmup"], SamplingParams(max_tokens=16, temperature=0))
    start = time.time()
    out = llm.generate(
        ["Explain the theory of general relativity in detail."],
        SamplingParams(max_tokens=128, temperature=0),
    )
    elapsed = time.time() - start
    n = len(out[0].outputs[0].token_ids)
    print(f"PURE_TARGET: {n} tokens in {elapsed:.2f}s = {n/elapsed:.1f} tok/s "
          f"({elapsed*1000/n:.1f} ms/tok)")
