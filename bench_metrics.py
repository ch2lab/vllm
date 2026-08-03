"""Measure prefill speed (~5000 tok), decode speed (256 tok), and TTFT.

Config via env: KVDT (kv-cache dtype), VLLM_SM70_AWQ_BACKEND (turbomind|wmma),
MODEL, MTP (1/0). Warms up the long-prefill + decode path first so CUDA graphs
are captured before timing (avoids cold-start inflation).
"""
import os, time

os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
KVDT = os.environ.get("KVDT", "auto")
BACKEND = os.environ.get("VLLM_SM70_AWQ_BACKEND", "wmma")
MODEL = os.environ.get("MODEL", "/data/models/Qwen3.6-27B-AWQ")
MTP = os.environ.get("MTP", "1") == "1"

_PASSAGE = (
    "The history of computing spans centuries, from early mechanical calculators "
    "to modern GPUs. Ada Lovelace wrote the first algorithm for Charles Babbage's "
    "Analytical Engine. Alan Turing formalized the notion of computation with his "
    "universal machine. Transistors replaced vacuum tubes in the late 1940s. "
    "Integrated circuits then enabled microprocessors and personal computers. "
    "Parallel computing and graphics processors accelerated machine learning. "
)
PROMPT = "Reference material for the task:\n" + _PASSAGE * 60 + \
    "\n\nQuestion: Summarize the key themes above and discuss their implications."

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    spec = {"method": "mtp", "num_speculative_tokens": 4} if MTP else None
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        kv_cache_dtype=KVDT,
        speculative_config=spec,
    )
    sp1 = SamplingParams(max_tokens=1, temperature=0)
    sp256 = SamplingParams(max_tokens=256, temperature=0)

    # Warmup: capture CUDA graphs for both the long-prefill and decode paths.
    for _ in range(2):
        llm.generate(["warmup short"], SamplingParams(max_tokens=16, temperature=0))
    llm.generate([PROMPT], sp256)  # untimed: captures prefill+decode graphs
    llm.generate([PROMPT], sp1)

    # TTFT + prefill: max_tokens=1 on the long prompt.
    t0 = time.time()
    o1 = llm.generate([PROMPT], sp1)
    ttft = time.time() - t0
    prompt_tokens = len(o1[0].prompt_token_ids)
    prefill_speed = prompt_tokens / ttft

    # Full generation (prefill + 256 decode), timed.
    t0 = time.time()
    o2 = llm.generate([PROMPT], sp256)
    full = time.time() - t0
    comp_tokens = len(o2[0].outputs[0].token_ids)
    decode_time = full - ttft
    decode_speed = max(comp_tokens - 1, 1) / decode_time if decode_time > 0 else 0

    snippet = " ".join(o2[0].outputs[0].text.split())[:160]
    print(f"CONFIG model={os.path.basename(MODEL)} KV={KVDT} backend={BACKEND} mtp={MTP}")
    print(f"PROMPT_TOKENS={prompt_tokens} COMPLETION_TOKENS={comp_tokens}")
    print(f"PREFILL: {prefill_speed:.1f} tok/s")
    print(f"TTFT: {ttft*1000:.1f} ms")
    print(f"DECODE: {decode_speed:.1f} tok/s ({decode_time:.2f}s for {comp_tokens-1} tok)")
    print(f"OUTPUT>>> {snippet} <<<END")
