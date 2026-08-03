"""Long-context MTP4 benchmark: kv_dtype, max_len, prompt_tokens param."""
import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams

SENT = (
    "The development of quantum mechanics in the early 20th century, "
    "covering Planck's radiation law, the photoelectric effect, Bohr's atom, "
    "de Broglie waves, Heisenberg's matrix mechanics, and Schrodinger's wave "
    "equation, transformed physics. "
)


def main():
    kv_dtype = sys.argv[1] if len(sys.argv) > 1 else "fp8"
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
    prompt_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 64000
    gpu_mem = float(sys.argv[4]) if len(sys.argv) > 4 else 0.92

    llm = LLM(
        model="/data/models/Qwen3.6-27B-AWQ",
        tensor_parallel_size=2,
        kv_cache_dtype=kv_dtype,
        max_model_len=max_len,
        gpu_memory_utilization=gpu_mem,
        language_model_only=True,
        max_num_batched_tokens=2048,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 4,
        },
    )
    llm.generate(["warmup"], SamplingParams(max_tokens=16, temperature=0))

    tok = llm.get_tokenizer()
    # Build a long prompt to the requested token length.
    raw = SENT * 4000
    prompt = tok.decode(tok.encode(raw)[:prompt_tokens])
    n_in = len(tok.encode(prompt))

    t0 = time.time()
    out = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0))
    dt = time.time() - t0
    n_out = len(out[0].outputs[0].token_ids)
    print(f"LONGCTX kv={kv_dtype} max_len={max_len} in={n_in} out={n_out} "
          f"decode={n_out / dt:.1f} tok/s total={dt:.1f}s")
    print("OUTPUT_HEAD:", repr(out[0].outputs[0].text[:120]))
    print("OUTPUT_TAIL:", repr(out[0].outputs[0].text[-120:]))


if __name__ == "__main__":
    main()