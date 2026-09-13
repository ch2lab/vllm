"""AWQ load + smoke test: confirm SM70 AWQ GEMM + nvfp4 KV + GDN hybrid works."""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")
MODEL = os.environ.get("MODEL", "/data/models/Qwen3.8-27B-AWQ-MTP")
KV = os.environ.get("KV", "nvfp4")
def main():
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=2, max_model_len=65536,
              gpu_memory_utilization=0.90, kv_cache_dtype=KV,
              enable_prefix_caching=True, enable_chunked_prefill=True,
              max_num_batched_tokens=8192, disable_log_stats=False)
    print(f"LOAD: {time.time()-t0:.0f}s", flush=True)
    for _ in range(2):
        llm.generate(['warmup'], SamplingParams(max_tokens=8, temperature=0))
    out = llm.generate(["Explain general relativity in two sentences."],
                       SamplingParams(max_tokens=48, temperature=0))
    print(f"SMOKE: {out[0].outputs[0].text.replace(chr(10), ' ')[:150]}",
          flush=True)
    print("AWQ SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
