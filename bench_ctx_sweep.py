"""Context-dependence sweep: pure-decode tok/s vs context length.
Goal: confirm decode throughput stays FLAT (no attenuation) as context grows.
Uses model-default max_model_len (tokenizer model_max_length=262144). TP2, nvfp4.
Run: python3 bench_ctx_sweep.py   env CTXS=..., GEN=..., MAX_MODEL_LEN=..."""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")
MODEL = os.environ.get("MODEL", "/data/models/Qwen3.8-27B-FP8")


def build(n, seed):
    import random
    rng = random.Random(seed)
    return [rng.randint(1000, 140000) for _ in range(n)]


def main():
    from vllm import LLM, SamplingParams
    kw = dict(model=MODEL, tensor_parallel_size=2,
              gpu_memory_utilization=0.90, kv_cache_dtype="nvfp4",
              disable_log_stats=False, enable_prefix_caching=True,
              enable_chunked_prefill=True, max_num_batched_tokens=8192)
    ml = os.environ.get("MAX_MODEL_LEN")
    if ml:
        kw["max_model_len"] = int(ml)
    llm = LLM(**kw)
    CTXS = [int(x) for x in os.environ.get(
        "CTXS", "2048,4096,8192,16384,32768,51000,100000").split(",")]
    GEN = int(os.environ.get("GEN", "64"))
    print(f"{'ctx':>7} {'tok/s':>8}  vs 2k")
    base = None
    for c in CTXS:
        p = build(c, seed=c // 1000 + 1)
        llm.generate([p], SamplingParams(max_tokens=1, temperature=0))
        t0 = time.time()
        llm.generate([p], SamplingParams(max_tokens=GEN, temperature=0))
        t0 = time.time()
        o = llm.generate([p], SamplingParams(max_tokens=GEN, temperature=0))
        t2 = time.time() - t0
        n = len(o[0].outputs[0].token_ids)
        rate = n / t2
        tag = "" if base is None else f"{rate / base * 100:.0f}%"
        if base is not None and rate < base * 0.92:
            tag += "  <-- ATTENUATION"
        base = base if base is not None else rate
        print(f"{c:>7} {rate:>8.1f}  {tag}", flush=True)


if __name__ == "__main__":
    main()
