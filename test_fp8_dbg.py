import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["SM70_DBG_KV"] = "1"

from vllm import LLM, SamplingParams
from vllm.v1.attention.backends import sm70_wmma_attn


def main():
    llm = LLM(
        model="/data/models/Qwen3.5-0.8B",
        tensor_parallel_size=1,
        kv_cache_dtype="fp8",
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        language_model_only=True,
        enforce_eager=True,
    )
    prompt = (
        "The history of computing spans many decades, from early mechanical "
        "calculators to modern neural networks. "
    ) * 40
    out = llm.generate([prompt], SamplingParams(max_tokens=16, temperature=0))
    print("OUTPUT:", repr(out[0].outputs[0].text[:120]))
    sm70_wmma_attn.dbg_verify_kv_snapshots()


if __name__ == "__main__":
    main()
