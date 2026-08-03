"""Benchmark NVFP4 weights model with MTP4."""
import os, time
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"


def main():
    from vllm import LLM, SamplingParams

    model = "/data/models/Qwen3.6-27B-Text-NVFP4-MTP"
    print(f"Benchmarking: {model}", flush=True)
    llm = LLM(
        model=model,
        tensor_parallel_size=2,
        max_model_len=2048,
        gpu_memory_utilization=0.92,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 4,
        },
    )
    # Warmup
    out = llm.generate(['Hello'], SamplingParams(max_tokens=32, temperature=0))
    print(f"WARMUP: {out[0].outputs[0].text[:60]}")
    for _ in range(5):
        llm.generate(['warmup'], SamplingParams(max_tokens=16, temperature=0))

    # Measure
    start = time.time()
    outs = llm.generate(
        ['Explain the theory of general relativity in detail.'],
        SamplingParams(max_tokens=128, temperature=0),
    )
    elapsed = time.time() - start
    n = len(outs[0].outputs[0].token_ids)
    print(f"NVFP4_WEIGHTS_MTP4: {n} tokens in {elapsed:.2f}s = {n/elapsed:.1f} tok/s")
    print(f"MS_PER_TOKEN: {elapsed*1000/n:.1f} ms")
    print(f"OUTPUT: {outs[0].outputs[0].text[:100]}")


if __name__ == "__main__":
    main()
