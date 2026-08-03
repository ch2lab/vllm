"""Comprehensive correctness test: all KV-cache types on Qwen3.5-4B."""
import os, sys, gc, time
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

PROMPT = "The capital of France is"
EXPECTED = ["paris", "Paris"]
MAX_TOKENS = 32


def test_kv_cache_dtype(model, kv_dtype, extra_kwargs=None):
    import torch
    from vllm import LLM, SamplingParams

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    kwargs = dict(
        model=model,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    if kv_dtype != "auto":
        kwargs["kv_cache_dtype"] = kv_dtype
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    try:
        llm = LLM(**kwargs)
        out = llm.generate([PROMPT], SamplingParams(max_tokens=MAX_TOKENS, temperature=0))
        text = out[0].outputs[0].text
        ok = any(w in text for w in EXPECTED)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        return ok, text[:80]
    except Exception as e:
        return None, f"ERROR: {type(e).__name__}: {e}"


def main():
    model = "/data/models/Qwen3.5-4B"
    configs = [
        ("fp16 (baseline)", "auto"),
        ("fp8_e5m2", "fp8_e5m2"),
        ("nvfp4", "nvfp4"),
    ]

    print(f"Model: {model}")
    print(f"Prompt: {PROMPT!r}")
    print("=" * 70)

    results = []
    for name, dtype in configs:
        print(f"\n--- Testing KV-cache: {name} ---", flush=True)
        ok, text = test_kv_cache_dtype(model, dtype)
        status = "PASS" if ok else ("FAIL" if ok is False else "ERROR")
        results.append((name, status, text))
        print(f"  [{status}] {text}")

    print("\n" + "=" * 70)
    print("SUMMARY:")
    all_pass = True
    for name, status, text in results:
        print(f"  {name:20s} {status}")
        if status != "PASS":
            all_pass = False
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
