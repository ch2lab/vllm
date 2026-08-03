import os
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/data/models/Qwen3.6-27B-AWQ",
        tensor_parallel_size=2,
        kv_cache_dtype="nvfp4",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
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
    out = llm.generate([prompt], SamplingParams(max_tokens=512, temperature=0))
    dt = time.time() - t0
    n_in = len(out[0].prompt_token_ids)
    n_out = len(out[0].outputs[0].token_ids)
    print(f"LONG_MTP_NVFP4: in={n_in} out={n_out} total={dt:.1f}s tok/s={n_out / dt:.1f}")
    print("OUTPUT_HEAD:", repr(out[0].outputs[0].text[:300]))
    print("OUTPUT_TAIL:", repr(out[0].outputs[0].text[-300:]))


if __name__ == "__main__":
    main()
