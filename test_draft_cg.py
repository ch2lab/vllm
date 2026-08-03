"""Test draft model CUDA Graph (FULL mode)."""
import os, sys, time

os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

LOG = "/data/src/vllm/test_draft_cg.log"

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
            # Warmup
            out = llm.generate(['Hello'], SamplingParams(max_tokens=32, temperature=0))
            log.write(f"WARMUP: {out[0].outputs[0].text[:60]}\n")
            for _ in range(5):
                llm.generate(['warmup'], SamplingParams(max_tokens=16, temperature=0))

            # Measure
            prompt = 'Explain the theory of general relativity in detail.'
            results = []
            for i in range(5):
                start = time.time()
                out = llm.generate([prompt], SamplingParams(max_tokens=128, temperature=0))
                elapsed = time.time() - start
                n = len(out[0].outputs[0].token_ids)
                tps = n / elapsed
                results.append(tps)
                log.write(f"Round {i}: {n} tokens in {elapsed:.2f}s = {tps:.1f} tok/s\n")

            log.write(f"AVG: {sum(results)/len(results):.1f} tok/s\n")
            log.write(f"OUTPUT: {out[0].outputs[0].text[:80]}\n")
            log.write("STATUS: SUCCESS\n")
        except Exception as e:
            import traceback
            log.write(f"STATUS: FAILED\n")
            log.write(f"ERROR: {type(e).__name__}: {e}\n")
            log.write(traceback.format_exc())

    print(f"Done. See {LOG}")
