"""Measure MTP acceptance rate with spec decode logging."""
import os, sys, time

os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

LOG = "/data/src/vllm/measure_acceptance.log"

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
                disable_log_stats=False,
            )
            # Warmup
            for _ in range(5):
                llm.generate(['warmup'], SamplingParams(max_tokens=16, temperature=0))

            # Generate enough tokens to get stable stats
            prompts = [
                'Explain the theory of general relativity in detail.',
                'Write a comprehensive overview of machine learning algorithms.',
                'Describe the history of computing from the 1940s to today.',
            ]
            for p in prompts:
                out = llm.generate([p], SamplingParams(max_tokens=256, temperature=0))
                n = len(out[0].outputs[0].token_ids)
                log.write(f"Generated {n} tokens\n")

            log.write("STATUS: SUCCESS\n")
        except Exception as e:
            import traceback
            log.write(f"STATUS: FAILED\n")
            log.write(f"ERROR: {type(e).__name__}: {e}\n")
            log.write(traceback.format_exc())

    print(f"Done. See {LOG}")
