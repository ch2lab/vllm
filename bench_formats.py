"""Benchmark: AWQ vs NVFP4 weights × FP8/NVFP4/INT4 KV-cache × TP1/TP2/PP2."""
import subprocess, json, os, sys

RESULTS = "/data/src/vllm/bench_formats_results.jsonl"

CONFIGS = [
    # (name, model, tp, pp, kv_cache_dtype)
    ("AWQ_TP2_KVfp8", "/data/models/Qwen3.6-27B-AWQ", 2, 1, "fp8"),
    ("AWQ_TP2_KVnvfp4", "/data/models/Qwen3.6-27B-AWQ", 2, 1, "nvfp4"),
    ("AWQ_TP2_KVint4", "/data/models/Qwen3.6-27B-AWQ", 2, 1, "int4_per_token_head"),
    ("NVFP4_TP2_KVfp8", "/data/models/Qwen3.6-27B-Text-NVFP4-MTP", 2, 1, "fp8"),
    ("NVFP4_TP2_KVnvfp4", "/data/models/Qwen3.6-27B-Text-NVFP4-MTP", 2, 1, "nvfp4"),
    ("NVFP4_TP2_KVint4", "/data/models/Qwen3.6-27B-Text-NVFP4-MTP", 2, 1, "int4_per_token_head"),
    ("AWQ_TP1_KVfp8", "/data/models/Qwen3.6-27B-AWQ", 1, 1, "fp8"),
    ("AWQ_PP2_KVfp8", "/data/models/Qwen3.6-27B-AWQ", 1, 2, "fp8"),
]

CHILD = '''
import torch, time, traceback, os, json, sys
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
cfg = json.loads(sys.argv[1])
r = {"name": cfg["name"], "status": "FAILED"}
try:
    from vllm import LLM, SamplingParams
    kw = dict(model=cfg["model"], tensor_parallel_size=cfg["tp"],
              pipeline_parallel_size=cfg["pp"], max_model_len=4096,
              gpu_memory_utilization=0.92, kv_cache_dtype=cfg["kv"],
              enforce_eager=False)
    if "AWQ" in cfg["name"]:
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 4}
    llm = LLM(**kw)
    for _ in range(3):
        llm.generate(["Hi"], SamplingParams(max_tokens=8, temperature=0))
    # Decode
    t0 = time.time()
    o = llm.generate(["Explain quantum computing briefly."],
                     SamplingParams(max_tokens=256, temperature=0))
    dt = time.time() - t0
    nt = len(o[0].outputs[0].token_ids)
    r["decode_tok_s"] = round(nt/dt, 1)
    r["decode_ms_tok"] = round(dt*1000/nt, 1)
    # Prefill
    lp = "The theory of relativity " * 200
    t0 = time.time()
    o = llm.generate([lp], SamplingParams(max_tokens=1, temperature=0))
    pt = time.time() - t0
    np_ = len(o[0].prompt_token_ids)
    r["prefill_tokens"] = np_
    r["prefill_ms"] = round(pt*1000, 1)
    r["prefill_tok_s"] = round(np_/pt, 1)
    r["status"] = "OK"
except Exception as e:
    r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
print("RESULT_JSON:" + json.dumps(r))
'''

if __name__ == "__main__":
    if os.path.exists(RESULTS):
        os.remove(RESULTS)
    for name, model, tp, pp, kv in CONFIGS:
        cfg = {"name": name, "model": model, "tp": tp, "pp": pp, "kv": kv}
        print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
        try:
            p = subprocess.run(["/data/vllm-dev/bin/python3", "-c", CHILD, json.dumps(cfg)],
                               capture_output=True, text=True, timeout=600)
            for line in p.stdout.split('\n'):
                if line.startswith("RESULT_JSON:"):
                    r = json.loads(line[len("RESULT_JSON:"):])
                    with open(RESULTS, "a") as f:
                        f.write(json.dumps(r) + "\n")
                    if r["status"] == "OK":
                        print(f"  Decode: {r['decode_tok_s']} tok/s | Prefill: {r['prefill_tok_s']} tok/s ({r['prefill_ms']}ms/{r['prefill_tokens']}tok)")
                    else:
                        print(f"  FAILED: {r.get('error','?')[:150]}")
                    break
            else:
                tail = (p.stderr or p.stdout)[-200:]
                print(f"  NO RESULT. tail: {tail}")
                with open(RESULTS, "a") as f:
                    f.write(json.dumps({"name": name, "status": "FAILED", "error": tail[:200]}) + "\n")
        except subprocess.TimeoutExpired:
            print("  TIMEOUT")
            with open(RESULTS, "a") as f:
                f.write(json.dumps({"name": name, "status": "TIMEOUT"}) + "\n")

    print(f"\n{'='*60}\nDONE. Results in {RESULTS}")
