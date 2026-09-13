"""Parametrized smoke + bench: weight format x kv-cache dtype matrix.

Env: MODEL (path), KV (auto|fp8_e4m3|fp8_e5m2|nvfp4), TAG (label),
     MTP (num speculative tokens, 0 = off), CTX_TOKENS (default 4000).
"""
import os, time

os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

MODEL = os.environ["MODEL"]
KV = os.environ.get("KV", "auto")
TAG = os.environ.get("TAG", "run")
MTP = int(os.environ.get("MTP", "2"))
CTX = int(os.environ.get("CTX_TOKENS", "4000"))
OUT = f"/data/src/vllm/matrix_{TAG}.log"


def build_long_prompt(n_tokens):
    unit = ("The importance of long context reasoning in large language models "
            "cannot be overstated. Modern architectures combine linear attention "
            "for efficiency with sparse full attention for recall. This hybrid "
            "design enables scaling to hundreds of thousands of tokens while "
            "keeping inference cost linear. ")
    return unit * (n_tokens // 30 + 1)


def main():
    from vllm import LLM, SamplingParams
    with open(OUT, "w") as log:
        def w(s):
            log.write(s + "\n"); log.flush(); print(s, flush=True)

        w(f"== {TAG} model={MODEL} kv={KV} mtp={MTP} {time.strftime('%F %T')} ==")
        t0 = time.time()
        kw = dict(
            model=MODEL,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            kv_cache_dtype=KV,
            disable_log_stats=False,
            trust_remote_code=True,
        )
        if os.environ.get("EAGER", "0") == "1":
            kw["enforce_eager"] = True
        if MTP > 0:
            kw["speculative_config"] = {
                "method": "mtp", "num_speculative_tokens": MTP}
        llm = LLM(**kw)
        w(f"LOAD: {time.time()-t0:.1f}s")

        SP = SamplingParams(max_tokens=16, temperature=0)
        for _ in range(3):
            llm.generate(["warmup"], SP)

        prompts = [
            "Explain the theory of general relativity in a few sentences.",
            "What is the capital of France? Answer with one word.",
        ]
        out = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0))
        for i, o in enumerate(out):
            txt = o.outputs[0].text.replace("\n", " ")[:120]
            w(f"SMOKE{i}: {txt}")

        tok = llm.get_tokenizer()
        lp = build_long_prompt(CTX)
        n_prompt = len(tok.encode(lp))
        w(f"LONG PROMPT: {n_prompt} tokens")

        t0 = time.time()
        llm.generate([lp], SamplingParams(max_tokens=1, temperature=0))
        tp = time.time() - t0
        w(f"PREFILL {n_prompt}t: {tp:.1f}s = {n_prompt/tp:.0f} tok/s")

        t0 = time.time()
        out = llm.generate(["Explain quantum computing in detail."],
                           SamplingParams(max_tokens=256, temperature=0))
        td = time.time() - t0
        n_out = len(out[0].outputs[0].token_ids)
        w(f"DECODE {n_out}t: {td:.1f}s = {n_out/td:.1f} tok/s")
        w(f"DECODE OUT: {out[0].outputs[0].text.replace(chr(10),' ')[:100]}")

        t0 = time.time()
        out = llm.generate([lp], SamplingParams(max_tokens=128, temperature=0))
        tl = time.time() - t0
        n_out = len(out[0].outputs[0].token_ids)
        w(f"LONGCTX {n_out}t @{n_prompt}: {tl:.1f}s = {n_out/tl:.1f} tok/s")
        w("STATUS: SUCCESS")


if __name__ == "__main__":
    main()
