"""SM70 performance benchmark: Qwen3.6-27B-AWQ, MTP4, TP2, FP8 KV cache."""
import os, time
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

LOG = "/data/src/vllm/bench_result.log"

def main():
    from vllm import LLM, SamplingParams

    with open(LOG, "w") as log:
        llm = LLM(
            model='/data/models/Qwen3.6-27B-AWQ',
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            kv_cache_dtype='fp8',
            disable_log_stats=False,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 4,
            },
        )

        sp = SamplingParams(max_tokens=256, temperature=0)

        # Warmup
        for _ in range(5):
            llm.generate(['warmup'], SamplingParams(max_tokens=32, temperature=0))

        prompts = [
            "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, gravitational time dilation, and the experimental evidence that supports each of these concepts.",
            "Write a comprehensive guide to Python decorators, including function decorators, class decorators, decorator factories, functools.wraps, and practical use cases in web frameworks.",
            "Describe the history of artificial intelligence from the 1950s Dartmouth conference through the AI winters to the modern deep learning revolution, covering key milestones and figures.",
        ]

        # Decode benchmark
        t0 = time.time()
        total_tokens = 0
        for p in prompts:
            out = llm.generate([p], sp)
            n = len(out[0].outputs[0].token_ids)
            total_tokens += n
            log.write(f"GEN: {n} tokens\n")
        elapsed = time.time() - t0
        log.write(f"\nDECODE: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.1f} tok/s\n")

        # Metrics
        metrics = llm.get_metrics()
        drafts = None
        for m in metrics:
            if m.name == 'vllm:spec_decode_num_drafts':
                drafts = m.value
        for m in metrics:
            if 'spec_decode' in m.name:
                if hasattr(m, 'values'):
                    vals = list(m.values)
                    log.write(f"METRIC: {m.name} = {vals}\n")
                    if 'per_pos' in m.name and drafts:
                        rates = [f"{v/drafts*100:.1f}%" for v in vals]
                        log.write(f"ACCEPTANCE: {rates}\n")
                else:
                    log.write(f"METRIC: {m.name} = {m.value}\n")

        log.write("STATUS: SUCCESS\n")

    print(f"Done. See {LOG}")

if __name__ == '__main__':
    main()
