"""Long-context decode bench: warm prefix cache, then measure pure decode.
Env: MODEL, KV, MTP, CTX (prompt tokens), GEN (tokens to generate)."""
import os
import time

os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

MODEL = os.environ["MODEL"]
KV = os.environ.get("KV", "nvfp4")
MTP = int(os.environ.get("MTP", "2"))
CTX = int(os.environ.get("CTX", "10000"))
GEN = int(os.environ.get("GEN", "256"))


def build_long_prompt(n_tokens):
    unit = ("The importance of long context reasoning in large language models "
            "cannot be overstated. Modern architectures combine linear attention "
            "for efficiency with sparse full attention for recall. This hybrid "
            "design enables scaling to hundreds of thousands of tokens while "
            "keeping inference cost linear. ")
    return unit * (n_tokens // 30 + 1)


def main():
    from vllm import LLM, SamplingParams

    kw = dict(
        model=MODEL,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        kv_cache_dtype=KV,
        disable_log_stats=False,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
    )
    if MTP > 0:
        kw["speculative_config"] = {
            "method": "mtp", "num_speculative_tokens": MTP}
    t0 = time.time()
    llm = LLM(**kw)
    print(f"LOAD {time.time()-t0:.0f}s", flush=True)

    tok = llm.get_tokenizer()
    prompt = build_long_prompt(CTX)
    n_prompt = len(tok.encode(prompt))
    print(f"PROMPT {n_prompt} tokens", flush=True)

    # Warm: prefill into prefix cache
    t0 = time.time()
    llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
    tp = time.time() - t0
    print(f"WARM PREFILL {n_prompt}t: {tp:.1f}s = {n_prompt/tp:.0f} tok/s",
          flush=True)

    # Hot re-prefill + decode
    t0 = time.time()
    out = llm.generate(
        [prompt], SamplingParams(max_tokens=GEN, temperature=0))
    tt = time.time() - t0
    n_out = len(out[0].outputs[0].token_ids)
    print(f"HOT TOTAL {n_out}t in {tt:.1f}s", flush=True)

    # Second identical gen: prefix fully cached -> pure decode + tiny prefill
    t0 = time.time()
    out = llm.generate(
        [prompt], SamplingParams(max_tokens=GEN, temperature=0))
    t2 = time.time() - t0
    n_out = len(out[0].outputs[0].token_ids)
    print(f"PURE DECODE {n_out}t @{n_prompt}: {t2:.2f}s = {n_out/t2:.1f} tok/s",
          flush=True)
    print(f"OUT: {out[0].outputs[0].text.replace(chr(10), ' ')[:100]}",
          flush=True)
    print("STATUS: DONE", flush=True)


if __name__ == "__main__":
    main()
