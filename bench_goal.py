"""Goal baseline: Qwen3.8-27B-FP8 + nvfp4 KV, TP2 + MTP2.

Targets: 10k prefill >= 2000 tok/s; 50k+ pure decode >= 50 tok/s.
Env: CTX (decode context tokens, default 51000), GEN (default 128).
"""
import os
import time

# SM70 box: custom AR requires a NVLink P2P pair. GPU0<->1 is PCIe-host-bridge
# (PHB, no NVLink) and deadlocks the cross-device flag protocol; GPU1<->2 is
# NV6. Pin to 1,2 unless the caller overrides.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")

MODEL = os.environ.get("MODEL", "/data/models/Qwen3.8-27B-FP8")
KV = os.environ.get("KV", "nvfp4")
MTP = int(os.environ.get("MTP", "0"))  # goal config: no MTP / spec decode
CTX = int(os.environ.get("CTX", "51000"))
GEN = int(os.environ.get("GEN", "128"))
PREFILL_CTX = int(os.environ.get("PREFILL_CTX", "10000"))
LOG = os.environ.get("LOG", "/data/src/vllm/bench_goal.log")


def build_long_prompt(n_tokens, seed=1234):
    # token-exact prompt: n_tokens random token ids (deterministic, no BPE
    # surprises, exact length for the 51k > max_model_len edge).
    import random
    rng = random.Random(seed)
    return [rng.randint(1000, 140000) for _ in range(n_tokens)]


def main():
    from vllm import LLM, SamplingParams

    with open(LOG, "w") as log:
        def w(s):
            log.write(s + "\n")
            log.flush()
            print(s, flush=True)

        w(f"== goal bench: {MODEL} KV={KV} MTP={MTP} "
          f"{time.strftime('%F %T')} ==")
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
        # max_model_len: leave unset -> vLLM uses the model/tokenizer default
        # (qwen3_5 tokenizer model_max_length = 262144). Only override if the
        # caller explicitly sets MAX_MODEL_LEN.
        _ml = os.environ.get("MAX_MODEL_LEN")
        if _ml:
            kw["max_model_len"] = int(_ml)
        if MTP > 0:
            kw["speculative_config"] = {
                "method": "mtp", "num_speculative_tokens": MTP}
        if os.environ.get("DISABLE_CUSTOM_AR") == "1":
            kw["disable_custom_all_reduce"] = True
        if os.environ.get("ENFORCE_EAGER") == "1":
            kw["enforce_eager"] = True
        t0 = time.time()
        llm = LLM(**kw)
        w(f"LOAD: {time.time()-t0:.0f}s")
        tok = llm.get_tokenizer()

        for _ in range(3):
            llm.generate(['warmup'], SamplingParams(max_tokens=16,
                                                    temperature=0))
        out = llm.generate(
            ["Explain general relativity in two sentences."],
            SamplingParams(max_tokens=64, temperature=0))
        w(f"SMOKE: {out[0].outputs[0].text.replace(chr(10), ' ')[:120]}")

        # --- Target 1: prefill @ ~10k (cold) ---
        p10k = build_long_prompt(PREFILL_CTX, seed=1)
        n10k = len(p10k)
        t0 = time.time()
        llm.generate([p10k], SamplingParams(max_tokens=1, temperature=0))
        tp10 = time.time() - t0
        w(f"PREFILL {n10k}t (cold): {tp10:.1f}s = {n10k/tp10:.0f} tok/s "
          f"[target 2000]")

        # --- Target 2: pure decode @ CTX (warm prefix cache) ---
        pctx = build_long_prompt(CTX, seed=2)
        nctx = len(pctx)
        t0 = time.time()
        llm.generate([pctx], SamplingParams(max_tokens=1, temperature=0))
        w(f"WARM PREFILL {nctx}t: {time.time()-t0:.1f}s")

        t0 = time.time()
        out = llm.generate([pctx], SamplingParams(max_tokens=GEN,
                                                  temperature=0))
        t_hot = time.time() - t0
        n_out = len(out[0].outputs[0].token_ids)
        w(f"HOT TOTAL {n_out}t @{nctx}: {t_hot:.1f}s")

        t0 = time.time()
        out = llm.generate([pctx], SamplingParams(max_tokens=GEN,
                                                  temperature=0))
        t2 = time.time() - t0
        n_out = len(out[0].outputs[0].token_ids)
        w(f"PURE DECODE {n_out}t @{nctx}: {t2:.2f}s = {n_out/t2:.1f} tok/s "
          f"[target 50]")
        w(f"OUT: {out[0].outputs[0].text.replace(chr(10), ' ')[:100]}")

        for m in llm.get_metrics():
            if 'spec_decode' in m.name:
                w(f"METRIC: {m.name}")
        w("STATUS: DONE")


if __name__ == "__main__":
    main()
