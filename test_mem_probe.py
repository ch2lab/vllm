"""Probe: where does VRAM go after long-context request? Runs engine in-process."""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
import time
import torch
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model='/data/models/Qwen3.6-27B-AWQ',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.93,
        kv_cache_dtype='nvfp4',
        disable_custom_all_reduce=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
    )
    for _ in range(3):
        llm.generate(["warmup"], SamplingParams(max_tokens=16, temperature=0))

    unit = ("The importance of long context reasoning in large language models "
            "cannot be overstated. Modern architectures combine linear attention "
            "for efficiency with sparse full attention for recall. ")
    prompt = unit * 270
    n = len(llm.get_tokenizer().encode(prompt))
    print(f"PROBE PROMPT={n}", flush=True)
    t0 = time.time()
    llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
    print(f"PROBE PREFILL {n}t: {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    llm.generate([prompt], SamplingParams(max_tokens=64, temperature=0))
    print(f"PROBE LONGCTX 64t: {time.time()-t0:.1f}s", flush=True)

    for dev in range(torch.cuda.device_count()):
        torch.cuda.set_device(dev)
        mem = torch.cuda.memory_stats()
        print(f"PROBE GPU{dev} current_allocated_MB: {mem['allocated_bytes.all.current']/1e6:.0f}")
        print(f"PROBE GPU{dev} reserved_MB: {mem['reserved_bytes.all.current']/1e6:.0f}")
        print(f"PROBE GPU{dev} max_reserved_MB: {mem['reserved_bytes.all.peak']/1e6:.0f}")
        print(f"PROBE GPU{dev} num_alloc_retries: {mem.get('num_alloc_retries', 0)}")
    print("PROBE STATUS: SUCCESS", flush=True)


if __name__ == "__main__":
    main()