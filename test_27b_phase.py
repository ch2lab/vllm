import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams


def main():
    kv_dtype = sys.argv[1] if len(sys.argv) > 1 else "float16"
    gpu_mem = float(sys.argv[2]) if len(sys.argv) > 2 else 0.92
    llm = LLM(
        model="/data/models/Qwen3.6-27B-AWQ",
        tensor_parallel_size=2,
        kv_cache_dtype=kv_dtype,
        max_model_len=8192,
        gpu_memory_utilization=gpu_mem,
        language_model_only=True,
        max_num_batched_tokens=2048,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 4,
        },
    )
    llm.generate(["warmup"], SamplingParams(max_tokens=16, temperature=0))
    prompt = (
        "Discuss the development of quantum mechanics in the early 20th "
        "century, including Planck's radiation law, the photoelectric "
        "effect, Bohr's atom, de Broglie waves, Heisenberg's matrix "
        "mechanics, and Schrodinger's wave equation. "
    ) * 12

    t0 = time.time()
    out1 = llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
    t1 = time.time() - t0
    n_in = len(out1[0].prompt_token_ids)
    first = out1[0].outputs[0].token_ids
    print(f"RUN_A: in={n_in} max_tokens=1 total={t1:.3f}s "
          f"first_tok={first[0] if first else None}")

    prompt_b = prompt + "Summarize the key ideas. "
    t0 = time.time()
    out2 = llm.generate([prompt_b], SamplingParams(max_tokens=512, temperature=0))
    t2 = time.time() - t0
    n_out = len(out2[0].outputs[0].token_ids)
    print(f"RUN_B: in={n_in + 4} out={n_out} total={t2:.3f}s")

    step = (t2 - t1) / max(n_out - 1, 1)
    prefill_time = t1 - step
    print(f"step_avg={step * 1000:.1f}ms  prefill_time={prefill_time:.3f}s")
    print(f"TTFT ~= {t1:.3f}s (prefill + 1 decode step)")
    print(f"PREFILL: {n_in / prefill_time:.1f} tok/s ({n_in} tokens)")
    print(f"DECODE:  {n_out / (t2 - t1):.1f} tok/s pure decode (excl prefill)")
    print(f"OVERALL: {n_out / t2:.1f} tok/s (incl prefill, old口径)")


if __name__ == "__main__":
    main()
