import os
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/data/models/Qwen3.6-27B-AWQ",
        tensor_parallel_size=2,
        kv_cache_dtype="float16",
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        language_model_only=True,
        max_num_batched_tokens=2048,
    )
    llm.generate(["warmup"], SamplingParams(max_tokens=64, temperature=0))
    prompt = (
        "Discuss the development of quantum mechanics in the early 20th "
        "century, including Planck's radiation law, the photoelectric "
        "effect, Bohr's atom, de Broglie waves, Heisenberg's matrix "
        "mechanics, and Schrodinger's wave equation. "
    ) * 12

    t0 = time.time()
    llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
    t1 = time.time() - t0

    t0 = time.time()
    out = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0))
    t2 = time.time() - t0
    n_out = len(out[0].outputs[0].token_ids)

    step = (t2 - t1) / max(n_out - 1, 1)
    print(f"NOMTP: in=625 out={n_out} prefill={t1:.3f}s "
          f"decode={(n_out - 1) / (t2 - t1):.1f} tok/s step={step * 1000:.1f}ms")


if __name__ == "__main__":
    main()
