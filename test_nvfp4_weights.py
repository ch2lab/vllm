"""Test NVFP4 model weights on Qwen3.6-27B-Text-NVFP4-MTP."""
import os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"


def main():
    from vllm import LLM, SamplingParams

    model = "/data/models/Qwen3.6-27B-Text-NVFP4-MTP"
    print(f"Testing NVFP4 weights: {model}", flush=True)
    llm = LLM(
        model=model,
        tensor_parallel_size=2,
        max_model_len=2048,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 4,
        },
    )
    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=32, temperature=0),
    )
    text = out[0].outputs[0].text
    print(f"OUTPUT: {text!r}")
    ok = "paris" in text.lower()
    print(f"STATUS: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
