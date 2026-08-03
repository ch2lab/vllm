"""A/B: no-MTP with custom all-reduce disabled (NCCL instead), to test
whether cross-rank custom-allreduce spin coupling paces the ~67ms step."""
import os, time
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

LOG = "/data/src/vllm/bench_result_nccl.log"

def main():
    from vllm import LLM, SamplingParams

    with open(LOG, "w") as log:
        llm = LLM(
            model='/data/models/Qwen3.6-27B-AWQ',
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            kv_cache_dtype='fp8',
            disable_log_stats=True,
            disable_custom_all_reduce=True,
        )

        sp = SamplingParams(max_tokens=256, temperature=0)
        for _ in range(5):
            llm.generate(['warmup'], SamplingParams(max_tokens=32, temperature=0))

        prompts = [
            "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, gravitational time dilation, and the experimental evidence that supports each of these concepts.",
            "Write a comprehensive guide to Python decorators, including function decorators, class decorators, decorator factories, functools.wraps, and practical use cases in web frameworks.",
            "Describe the history of artificial intelligence from the 1950s Dartmouth conference through the AI winters to the modern deep learning revolution, covering key milestones and figures.",
        ]

        t0 = time.time()
        total_tokens = 0
        for p in prompts:
            out = llm.generate([p], sp)
            n = len(out[0].outputs[0].token_ids)
            total_tokens += n
            log.write(f"GEN: {n} tokens\n")
        elapsed = time.time() - t0
        log.write(f"\nDECODE: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.1f} tok/s\n")
        log.write("STATUS: SUCCESS\n")

    print(f"Done. See {LOG}")

if __name__ == '__main__':
    main()
