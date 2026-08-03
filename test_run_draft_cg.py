"""Test 27B MTP4 with SM70 draft-model breakable CUDA graph enabled."""
import os

os.environ["VLLM_SM70_DRAFT_CUDAGRAPH"] = "1"
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

import time, traceback

LOG = "/data/src/vllm/test_run_draft_cg.log"

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
            out = llm.generate(['Hello'], SamplingParams(max_tokens=32, temperature=0))
            log.write(f"WARMUP: {out[0].outputs[0].text[:60]}\n")
            for _ in range(5):
                llm.generate(['warmup'], SamplingParams(max_tokens=16, temperature=0))

            start = time.time()
            outs = llm.generate(['Explain the theory of general relativity in detail.'],
                               SamplingParams(max_tokens=128, temperature=0))
            elapsed = time.time() - start
            n = len(outs[0].outputs[0].token_ids)
            log.write(f"DRAFT_CG: {n} tokens in {elapsed:.2f}s = {n/elapsed:.1f} tok/s\n")
            log.write(f"MS_PER_TOKEN: {elapsed*1000/n:.1f} ms\n")
            log.write(f"OUTPUT: {outs[0].outputs[0].text[:80]}\n")
            log.write("STATUS: SUCCESS\n")
        except Exception as e:
            log.write(f"STATUS: FAILED\n")
            log.write(f"ERROR: {type(e).__name__}: {e}\n")
            log.write(traceback.format_exc())

    print(f"Done. See {LOG}")
