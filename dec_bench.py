"""Quick decode benchmark (single load). Env: MODEL, KV, MTP, TAG."""
import os, time

os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
from vllm import LLM, SamplingParams

MODEL = os.environ["MODEL"]
KV = os.environ.get("KV", "nvfp4")
MTP = int(os.environ.get("MTP", "2"))
TAG = os.environ.get("TAG", "dec")

kw = dict(model=MODEL, tensor_parallel_size=2, gpu_memory_utilization=0.90,
          kv_cache_dtype=KV, disable_log_stats=False)
CG = os.environ.get("CGMODE")
if CG:
    kw["compilation_config"] = {"cudagraph_mode": CG}
if MTP > 0:
    kw["speculative_config"] = {"method": "mtp",
                                "num_speculative_tokens": MTP}
def main():
    t0 = time.time()
    llm = LLM(**kw)
    print(f"{TAG} LOAD {time.time()-t0:.0f}s", flush=True)

    SP = SamplingParams(max_tokens=8, temperature=0)
    for _ in range(3):
        llm.generate(["warmup"], SP)

    out = llm.generate(["Explain quantum computing in detail."],
                       SamplingParams(max_tokens=256, temperature=0))
    n = len(out[0].outputs[0].token_ids)
    txt = out[0].outputs[0].text.replace("\n", " ")
    print(f"{TAG} DECODE {n}t: {txt[:80]}", flush=True)

    # second run (graphs hot)
    t0 = time.time()
    out = llm.generate(["Describe the history of the Roman Empire briefly."],
                       SamplingParams(max_tokens=256, temperature=0))
    dt = time.time() - t0
    n = len(out[0].outputs[0].token_ids)
    print(f"{TAG} DECODE2 {n}t {dt:.2f}s = {n/dt:.1f} tok/s", flush=True)
    print(f"{TAG} OUT2: {out[0].outputs[0].text.replace(chr(10),' ')[:80]}",
          flush=True)


if __name__ == "__main__":
    main()
