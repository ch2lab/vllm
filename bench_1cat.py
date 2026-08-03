"""1Cat benchmark: Qwen3.6-27B-AWQ TP2 MTP4, matching our test_run.py config."""
import sys, os
# Remove script dir from path to avoid importing our local vllm
sys.path = [p for p in sys.path if not p.startswith('/data/src')]
os.chdir('/data/models')

def main():
    import time
    from vllm import LLM, SamplingParams

    MODEL = "/data/models/Qwen3.6-27B-AWQ"

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        max_model_len=2048,
        gpu_memory_utilization=0.92,
        speculative_config={"method": "mtp", "num_speculative_tokens": 4},
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=128)

    # Warmup
    out = llm.generate(["Hello"], sampling)
    print(f"WARMUP: {out[0].outputs[0].text[:50]}")

    # Benchmark
    prompt = "Write a Python script that prints hello world"
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        out = llm.generate([prompt], sampling)
        t1 = time.perf_counter()
        n_tokens = len(out[0].outputs[0].token_ids)
        elapsed = t1 - t0
        tps = n_tokens / elapsed
        times.append((elapsed, n_tokens, tps))
        print(f"RUN {i}: {n_tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")

    best = max(times, key=lambda x: x[2])
    avg_tps = sum(t[2] for t in times) / len(times)
    print(f"\nBEST: {best[1]} tokens in {best[0]:.2f}s = {best[2]:.1f} tok/s")
    print(f"AVG: {avg_tps:.1f} tok/s")
    print(f"MS_PER_TOKEN: {1000/best[2]:.1f} ms")
    print(f"OUTPUT: {out[0].outputs[0].text[:100]}")
    print("STATUS: SUCCESS")

if __name__ == '__main__':
    main()
