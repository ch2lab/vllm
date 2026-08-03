import glob
import gzip
import json
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
PROF_DIR = "/data/src/vllm/prof_out"
os.makedirs(PROF_DIR, exist_ok=True)
os.environ["VLLM_TORCH_PROFILER_DIR"] = PROF_DIR

from vllm import LLM, SamplingParams
from vllm.config.profiler import ProfilerConfig


def main():
    pcfg = ProfilerConfig()
    pcfg.profiler = "torch"
    pcfg.torch_profiler_dir = PROF_DIR
    pcfg.torch_profiler_with_stack = False
    llm = LLM(
        model="/data/models/Qwen3.6-27B-AWQ",
        tensor_parallel_size=2,
        kv_cache_dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        language_model_only=True,
        profiler_config=pcfg,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    )
    llm.generate(["Hello"], SamplingParams(max_tokens=8, temperature=0))
    llm.llm_engine.start_profile()
    llm.generate(
        ["Write a story."],
        SamplingParams(max_tokens=64, temperature=0),
    )
    llm.llm_engine.stop_profile()
    print("PROFILING DONE, parsing traces...")
    agg: dict[str, list[float]] = {}
    for path in glob.glob(f"{PROF_DIR}/**/*.trace.json*", recursive=True):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            trace = json.load(f)
        for ev in trace.get("traceEvents", []):
            if ev.get("cat") == "kernel":
                agg.setdefault(ev["name"], []).append(ev["dur"])
    rows = [(sum(v), len(v), k) for k, v in agg.items()]
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    print(f"TOTAL KERNEL TIME {total/1e6:.3f} s (rank traces in {PROF_DIR})")
    for t, n, name in rows[:25]:
        print(f"{t/1e3:10.2f} ms {100*t/total:5.1f}% n={n:6d} {name[:95]}")


if __name__ == "__main__":
    main()
