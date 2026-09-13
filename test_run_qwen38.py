"""SM70 smoke + benchmark: Qwen3.8-27B-AWQ-MTP, TP2, MTP2, NVFP4 KV cache."""
import os, time
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

LOG = "/data/src/vllm/bench_qwen38.log"
EAGER = os.environ.get("EAGER", "0") == "1"


def log_write(f, s):
    f.write(s + "\n")
    f.flush()


def build_long_prompt(n_tokens):
    """Repeatable pseudo-text prompt of approximately n_tokens."""
    unit = ("The importance of long context reasoning in large language models "
            "cannot be overstated. Modern architectures combine linear attention "
            "for efficiency with sparse full attention for recall. This hybrid "
            "design enables scaling to hundreds of thousands of tokens while "
            "keeping inference cost linear. ")
    n_unit = n_tokens // 30 + 1
    return unit * n_unit


def main():
    from vllm import LLM, SamplingParams

    with open(LOG, "w") as log:
        log_write(log, f"== Qwen3.8-27B-AWQ-MTP smoke (TP2, MTP2, nvfp4 kv) {time.strftime('%F %T')} ==")
        t0 = time.time()

        llm = LLM(
            model='/data/models/Qwen3.8-27B-AWQ-MTP',
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            kv_cache_dtype='nvfp4',
            disable_log_stats=False,
            enforce_eager=EAGER,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 2,
            },
        )
        log_write(log, f"LOAD: {time.time()-t0:.1f}s")

        # Warmup
        for _ in range(3):
            llm.generate(['warmup'], SamplingParams(max_tokens=16, temperature=0))

        # Correctness smoke: greedy short generations
        prompts = [
            "Explain the theory of general relativity in a few sentences.",
            "What is the capital of France? Answer with one word.",
        ]
        out = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0))
        for i, o in enumerate(out):
            txt = o.outputs[0].text.replace("\n", " ")[:120]
            log_write(log, f"SMOKE{i}: {txt}")

        # Long-context prefill benchmark (~8k tokens)
        n_ctx = int(os.environ.get("CTX_TOKENS", "8000"))
        long_prompt = build_long_prompt(n_ctx)
        n_prompt = len(llm.get_tokenizer().encode(long_prompt))
        log_write(log, f"LONG PROMPT: {n_prompt} tokens")

        t0 = time.time()
        out = llm.generate([long_prompt], SamplingParams(max_tokens=1, temperature=0))
        t_prefill = time.time() - t0
        log_write(log, f"PREFILL {n_prompt}t: {t_prefill:.1f}s = {n_prompt/t_prefill:.0f} tok/s")

        # Short-prompt decode benchmark
        t0 = time.time()
        out = llm.generate(['Explain quantum computing in detail.'], SamplingParams(max_tokens=256, temperature=0))
        t_decode = time.time() - t0
        n_out = len(out[0].outputs[0].token_ids)
        log_write(log, f"DECODE {n_out}t: {t_decode:.1f}s = {n_out/t_decode:.1f} tok/s")

        # Long-context decode benchmark (after 8k prefill, same KV active)
        t0 = time.time()
        out = llm.generate([long_prompt], SamplingParams(max_tokens=128, temperature=0))
        t_dec_long = time.time() - t0
        n_out = len(out[0].outputs[0].token_ids)
        log_write(log, f"LONGCTX DECODE {n_out}t (prefill {n_prompt}+{n_out}): {t_dec_long:.1f}s = {n_out/t_dec_long:.1f} tok/s")

        # Metrics
        for m in llm.get_metrics():
            if 'spec' in m.name or 'kv' in m.name.lower() or 'prefill' in m.name or 'decode' in m.name:
                if 'per_pos' in m.name and hasattr(m, 'values'):
                    log_write(log, f"METRIC: {m.name} = {list(m.values)}")
                elif hasattr(m, 'value'):
                    log_write(log, f"METRIC: {m.name} = {m.value}")

        log_write(log, "STATUS: SUCCESS")

    print(f"Done. See {LOG}")


if __name__ == '__main__':
    main()