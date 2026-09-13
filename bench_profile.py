"""Profile decode-step kernel costs: warm 51k prefill, then N decode steps."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")
import time

from vllm import LLM, SamplingParams

CTX = int(os.environ.get("CTX", "51000"))
GEN = int(os.environ.get("GEN", "32"))


def make_prompt(n, seed=7):
    import random

    rng = random.Random(seed)
    return [rng.randint(1000, 140000) for _ in range(n)]


def main():
    kw = dict(
        model=os.environ.get(
            "MODEL", "/data/models/Qwen3.8-27B-FP8"),
        tensor_parallel_size=2,
        max_model_len=65536,
        gpu_memory_utilization=0.90,
        kv_cache_dtype="nvfp4",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        disable_log_stats=True,
    )
    if os.environ.get("ENFORCE_EAGER") == "1":
        kw["enforce_eager"] = True
    if os.environ.get("DISABLE_CUSTOM_AR") == "1":
        kw["disable_custom_all_reduce"] = True
    llm = LLM(**kw)
    for _ in range(2):
        llm.generate(["warmup"], SamplingParams(max_tokens=8, temperature=0))
    p = make_prompt(CTX)
    t0 = time.time()
    llm.generate([p], SamplingParams(max_tokens=1, temperature=0))
    print(f"WARM PREFILL {CTX}t: {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    out = llm.generate([p], SamplingParams(max_tokens=GEN, temperature=0))
    dt = time.time() - t0
    n_out = len(out[0].outputs[0].token_ids)
    print(f"DECODE {n_out}t @{CTX}: {dt:.2f}s = {n_out/dt:.2f} tok/s", flush=True)
    print("PROFILE MARKER END", flush=True)


if __name__ == "__main__":
    main()
