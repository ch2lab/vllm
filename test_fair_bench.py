"""Fair benchmark: FULL_AND_PIECEWISE vs FULL_DECODE_ONLY with thorough warmup."""
import torch, sys, time, traceback, os

LOG = "/data/src/vllm/test_run.log"
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

if __name__ == '__main__':
    with open(LOG, "w") as log:
        try:
            from vllm import LLM, SamplingParams
            llm = LLM(
                model='/data/models/Qwen3.6-27B-AWQ',
                tensor_parallel_size=2,
                max_model_len=2048,
                gpu_memory_utilization=0.92,
                speculative_config={
                    "method": "mtp",
                    "num_speculative_tokens": 4,
                },
            )
            sp128 = SamplingParams(max_tokens=128, temperature=0)
            prompt = 'Explain the theory of general relativity in detail.'

            # Thorough warmup: 20 requests at full length to trigger all JIT
            log.write("WARMUP: 20 requests x 128 tokens...\n")
            for i in range(20):
                out = llm.generate([prompt], sp128)
                n = len(out[0].outputs[0].token_ids)
                log.write(f"  warmup {i}: {n} tokens\n")
            log.write("WARMUP DONE\n")

            # Measure: 5 rounds
            times = []
            for i in range(5):
                start = time.time()
                outs = llm.generate([prompt], sp128)
                elapsed = time.time() - start
                n = len(outs[0].outputs[0].token_ids)
                tps = n / elapsed
                times.append((elapsed, n, tps))
                log.write(f"RUN {i}: {n} tokens in {elapsed:.2f}s = {tps:.1f} tok/s\n")

            best = max(times, key=lambda x: x[2])
            avg = sum(t[2] for t in times) / len(times)
            log.write(f"\nBEST: {best[1]} tokens in {best[0]:.2f}s = {best[2]:.1f} tok/s\n")
            log.write(f"AVG: {avg:.1f} tok/s\n")
            log.write(f"MS_PER_TOKEN: {1000/best[2]:.1f} ms\n")
            log.write(f"OUTPUT: {outs[0].outputs[0].text[:80]}\n")
            log.write("STATUS: SUCCESS\n")
        except Exception as e:
            log.write(f"STATUS: FAILED\n")
            log.write(f"ERROR: {type(e).__name__}: {e}\n")
            log.write(traceback.format_exc())

    print(f"Done. See {LOG}")
