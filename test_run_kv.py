"""Test 27B MTP4 with different KV-cache types."""
import os, sys, time, gc
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

KV_DTYPE = sys.argv[1] if len(sys.argv) > 1 else "auto"
LOG = f"/data/src/vllm/test_run_{KV_DTYPE}.log"


def main():
    import torch
    from vllm import LLM, SamplingParams

    with open(LOG, "w") as log:
        try:
            kwargs = dict(
                model='/data/models/Qwen3.6-27B-AWQ',
                tensor_parallel_size=2,
                max_model_len=2048,
                gpu_memory_utilization=0.92,
                speculative_config={
                    "method": "mtp",
                    "num_speculative_tokens": 4,
                },
            )
            if KV_DTYPE != "auto":
                kwargs["kv_cache_dtype"] = KV_DTYPE

            llm = LLM(**kwargs)
            # Warmup
            out = llm.generate(['Hello'], SamplingParams(max_tokens=32, temperature=0))
            log.write(f"WARMUP: {out[0].outputs[0].text[:60]}\n")
            for _ in range(5):
                llm.generate(['warmup'], SamplingParams(max_tokens=16, temperature=0))

            # Measure
            start = time.time()
            outs = llm.generate(
                ['Explain the theory of general relativity in detail.'],
                SamplingParams(max_tokens=128, temperature=0),
            )
            elapsed = time.time() - start
            n = len(outs[0].outputs[0].token_ids)
            text = outs[0].outputs[0].text
            log.write(f"KV_DTYPE: {KV_DTYPE}\n")
            log.write(f"MTP_GRAPH: {n} tokens in {elapsed:.2f}s = {n/elapsed:.1f} tok/s\n")
            log.write(f"MS_PER_TOKEN: {elapsed*1000/n:.1f} ms\n")
            log.write(f"OUTPUT: {text[:120]}\n")
            # Correctness check
            ok = len(text.strip()) > 20 and not text.startswith("not the only")
            log.write(f"CORRECT: {ok}\n")
            log.write("STATUS: SUCCESS\n")
        except Exception as e:
            import traceback
            log.write(f"STATUS: FAILED\n")
            log.write(f"ERROR: {type(e).__name__}: {e}\n")
            log.write(traceback.format_exc())

    print(f"Done. See {LOG}")


if __name__ == '__main__':
    main()
