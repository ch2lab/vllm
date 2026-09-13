"""E2E: fused fla_verify_sm70 in the real server (short prompts T<=8).

Scenarios:
  1. correctness: tiny prompts (fused fires) + longer prompts (chain) outputs
  2. perf: wave of tiny requests, sequential issue (each = 1 small-prefill
     step + short decode) - fused vs chain wall time
Run twice: VLLM_SM70_FLA_FUSED_VERIFY=1 (default) and =0.
"""
import os
import time

LOG = "/data/src/vllm/bench_fused_e2e.log"


def log_write(f, s):
    f.write(s + "\n")
    f.flush()


def main():
    from vllm import LLM, SamplingParams

    fused = os.environ.get("VLLM_SM70_FLA_FUSED_VERIFY", "1")
    with open(LOG, "w") as log:
        log_write(log, f"== fused e2e FUSED={fused} {time.strftime('%F %T')} ==")
        t0 = time.time()
        llm = LLM(
            model='/data/models/Qwen3.8-27B-AWQ-MTP',
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            kv_cache_dtype='nvfp4',
            disable_log_stats=False,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 2,
            },
        )
        log_write(log, f"LOAD: {time.time()-t0:.1f}s")

        # Warmup (short prompts -> fused fires during warmup too)
        llm.generate(["Hi."], SamplingParams(max_tokens=8, temperature=0))

        # 1. Correctness: tiny + medium prompts, greedy
        t0 = time.time()
        prompts = [
            "Hi.", "Ok.", "Yes.", "No.", "What is 2+2? Answer with a number.",
            "Capital of France? One word.", "Hello.",
            "Explain quantum computing in one sentence.",
        ]
        out = llm.generate(prompts, SamplingParams(max_tokens=24, temperature=0))
        log_write(log, f"CORRECTNESS {len(prompts)} prompts: {time.time()-t0:.1f}s")
        for i, o in enumerate(out):
            txt = o.outputs[0].text.replace("\n", " ")[:100]
            log_write(log, f"OUT{i}[{prompts[i][:20]}]: {txt}")

        # 2. Perf: wave of tiny requests, one at a time (each small prefill
        #    rides a separate step -> fused path per step)
        nmax = int(os.environ.get("N_TINY", "12"))
        t0 = time.time()
        for i in range(nmax):
            llm.generate([f"Hi {i}."], SamplingParams(max_tokens=8, temperature=0))
        dt = time.time() - t0
        log_write(log, f"WAVE {nmax} tiny: {dt:.2f}s = {nmax/dt:.1f} req/s")

        # 3. Medium decode sanity (32+ token prompt -> chain prefill path)
        t0 = time.time()
        out = llm.generate(["Explain quantum computing in detail."],
                           SamplingParams(max_tokens=64, temperature=0))
        n_out = len(out[0].outputs[0].token_ids)
        log_write(log, f"DECODE {n_out}t: {time.time()-t0:.1f}s = "
                       f"{n_out/(time.time()-t0):.1f} tok/s")
        log_write(log, "STATUS: SUCCESS")
    print(f"Done. See {LOG}")


if __name__ == '__main__':
    main()