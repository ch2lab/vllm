"""cProfile the in-process engine loop (no-MTP decode) to find the ~70ms/step
host cost of the hybrid GDN model."""
import os, time, cProfile, pstats, io
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

LOG = "/data/src/vllm/bench_result_prof.log"

def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model='/data/models/Qwen3.6-27B-AWQ',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        kv_cache_dtype='fp8',
        disable_log_stats=True,
    )
    for _ in range(3):
        llm.generate(['warmup'], SamplingParams(max_tokens=16, temperature=0))

    prompt = ("Explain the theory of general relativity in detail, covering "
              "spacetime curvature and gravitational time dilation.")
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()
    out = llm.generate([prompt], SamplingParams(max_tokens=64, temperature=0))
    dt = time.time() - t0
    pr.disable()
    n = len(out[0].outputs[0].token_ids)

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(35)
    with open(LOG, "w") as log:
        log.write(f"DECODE: {n} tokens in {dt:.2f}s = {n/dt:.1f} tok/s\n")
        log.write(s.getvalue())
    print(f"Done. See {LOG}")

if __name__ == '__main__':
    main()
