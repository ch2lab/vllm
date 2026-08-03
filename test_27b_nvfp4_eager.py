"""E2E: Qwen3.6-27B-AWQ TP2 with nvfp4 KV cache on SM70 (eager first)."""
import os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model='/data/models/Qwen3.6-27B-AWQ',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.88,
        kv_cache_dtype='nvfp4',
        disable_log_stats=True,
        enforce_eager=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 4,
        },
    )

    sp = SamplingParams(max_tokens=128, temperature=0)
    prompt = ("Explain the theory of general relativity in detail, "
              "covering spacetime curvature and gravitational time dilation.")
    out = llm.generate([prompt], sp)
    text = out[0].outputs[0].text
    print("OUTPUT:", text[:500], flush=True)
    print("TOKENS:", len(out[0].outputs[0].token_ids), flush=True)

if __name__ == '__main__':
    main()
