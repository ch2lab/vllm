"""Measure pure target decode (no speculation) for breakdown analysis."""
import torch, sys, time, traceback, os

LOG = "/data/src/vllm/test_nospec.log"
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
            )
            out = llm.generate(['Hello'], SamplingParams(max_tokens=32, temperature=0))
            log.write(f"WARMUP: {out[0].outputs[0].text[:40]}\n")
            for _ in range(5):
                llm.generate(['warmup'], SamplingParams(max_tokens=16, temperature=0))
            start = time.time()
            outs = llm.generate(['Explain the theory of general relativity in detail.'],
                               SamplingParams(max_tokens=128, temperature=0))
            elapsed = time.time() - start
            n = len(outs[0].outputs[0].token_ids)
            log.write(f"NOSPEC: {n} tokens in {elapsed:.2f}s = {n/elapsed:.1f} tok/s\n")
            log.write(f"MS_PER_TOKEN: {elapsed*1000/n:.1f} ms\n")
            log.write("STATUS: SUCCESS\n")
        except Exception as e:
            log.write(f"STATUS: FAILED\nERROR: {type(e).__name__}: {e}\n")
            log.write(traceback.format_exc())
    print(f"Done. See {LOG}")
