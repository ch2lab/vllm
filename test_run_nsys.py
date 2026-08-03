"""nsys decode timeline: same as test_run.py but gpu_memory_utilization=0.88
to leave headroom for nsys overhead. Keep the CUDA-graph memory profiler
ENABLED — disabling it lets KV allocation eat the ~2 GiB graph pool and OOMs
capture."""
import os, time

LOG = "/data/src/vllm/bench_result_nsys.log"

def main():
    from vllm import LLM, SamplingParams

    with open(LOG, "w") as log:
        llm = LLM(
            model='/data/models/Qwen3.6-27B-AWQ',
            tensor_parallel_size=2,
            gpu_memory_utilization=0.88,
            kv_cache_dtype='fp8',
            disable_log_stats=True,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 4,
            },
        )

        sp = SamplingParams(max_tokens=128, temperature=0)
        for _ in range(3):
            llm.generate(['warmup'], SamplingParams(max_tokens=16, temperature=0))

        prompt = ("Explain the theory of general relativity in detail, "
                  "covering spacetime curvature and gravitational time dilation.")
        t0 = time.time()
        out = llm.generate([prompt], sp)
        elapsed = time.time() - t0
        n = len(out[0].outputs[0].token_ids)
        log.write(f"DECODE: {n} tokens in {elapsed:.2f}s = {n/elapsed:.1f} tok/s\n")
        log.write("STATUS: SUCCESS\n")
    print(f"Done. See {LOG}")

if __name__ == '__main__':
    main()
