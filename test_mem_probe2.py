"""Probe v2: dump engine cache config + torch memory split (in-process engine)."""
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
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
    )
    cc = llm.llm_engine.vllm_config.cache_config
    print(f"PROBE2 cache_config: block_size={cc.block_size} "
          f"num_gpu_blocks={getattr(cc, 'num_gpu_blocks', None)} "
          f"cache_dtype={cc.cache_dtype} "
          f"max_seq_len={llm.llm_engine.vllm_config.model_config.max_model_len}",
          flush=True)
    for _ in range(3):
        llm.generate(["warmup"], SamplingParams(max_tokens=16, temperature=0))

    unit = ("The importance of long context reasoning in large language models "
            "cannot be overstated. Modern architectures combine linear attention "
            "for efficiency with sparse full attention for recall. ")
    prompt = unit * 270
    n = len(llm.get_tokenizer().encode(prompt))
    print(f"PROBE2 PROMPT={n}", flush=True)
    t0 = time.time()
    llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
    print(f"PROBE2 PREFILL {n}t: {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    llm.generate([prompt], SamplingParams(max_tokens=64, temperature=0))
    print(f"PROBE2 LONGCTX 64t: {time.time()-t0:.1f}s", flush=True)

    cc = llm.llm_engine.vllm_config.cache_config
    print(f"PROBE2 after-longctx num_gpu_blocks={getattr(cc, 'num_gpu_blocks', None)}",
          flush=True)
    for dev in range(torch.cuda.device_count()):
        torch.cuda.set_device(dev)
        mem = torch.cuda.memory_stats()
        print(f"PROBE2 GPU{dev} allocated_MB={mem['allocated_bytes.all.current']/1e6:.0f} "
              f"reserved_MB={mem['reserved_bytes.all.current']/1e6:.0f} "
              f"max_reserved_MB={mem['reserved_bytes.all.peak']/1e6:.0f}", flush=True)
    print("PROBE2 STATUS: SUCCESS", flush=True)


if __name__ == "__main__":
    main()