"""E2E: Qwen3.6-27B-AWQ TP2 with nvfp4 KV cache on SM70, CUDA Graph + MTP."""
import os, time
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model='/data/models/Qwen3.6-27B-AWQ',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.88,
        kv_cache_dtype='nvfp4',
        disable_log_stats=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 4,
        },
    )

    sp = SamplingParams(max_tokens=256, temperature=0)
    for _ in range(3):
        llm.generate(['warmup'], SamplingParams(max_tokens=16, temperature=0))

    prompt = ("Explain the theory of general relativity in detail, "
              "covering spacetime curvature and gravitational time dilation.")
    t0 = time.time()
    out = llm.generate([prompt], sp)
    elapsed = time.time() - t0
    text = out[0].outputs[0].text
    n = len(out[0].outputs[0].token_ids)
    print(f"DECODE: {n} tokens in {elapsed:.2f}s = {n/elapsed:.1f} tok/s",
          flush=True)
    print("OUTPUT:", text[:400], flush=True)

if __name__ == '__main__':
    main()
