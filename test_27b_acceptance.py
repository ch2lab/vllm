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
        disable_log_stats=False,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 4,
        },
    )
    warm = (
        "Discuss the development of quantum mechanics in the early 20th "
        "century, including Planck's radiation law, the photoelectric "
        "effect, Bohr's atom, de Broglie waves, Heisenberg's matrix "
        "mechanics, and Schrodinger's wave equation. "
    ) * 12
    llm.generate([warm], SamplingParams(max_tokens=64, temperature=0))
    prompt = (
        "Discuss the development of quantum mechanics in the early 20th "
        "century, including Planck's radiation law, the photoelectric "
        "effect, Bohr's atom, de Broglie waves, Heisenberg's matrix "
        "mechanics, and Schrodinger's wave equation. "
    ) * 12
    t0 = time.time()
    out = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0))
    dt = time.time() - t0
    n_out = len(out[0].outputs[0].token_ids)
    print(f"ACC_PROBE: in=625 out={n_out} total={dt:.2f}s tok/s={n_out / dt:.1f}")

    for m in llm.get_metrics():
        name = m.name
        if "spec" in name or "draft" in name or "accept" in name:
            print(f"METRIC {name} {m.labels} = {m.value if hasattr(m, 'value') else m.values}")


if __name__ == "__main__":
    main()
