"""Diagnostic: isolate prefill hang with enforce_eager and step logging."""
import time, os, sys
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
LOG = "/data/src/vllm/diag_eager.log"

def log(msg):
    line = f"[{time.time():.1f}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

if __name__ == '__main__':
    with open(LOG, "w") as f:
        f.write(f"START {time.time():.1f}\n")
    try:
        log("importing vllm")
        from vllm import LLM, SamplingParams
        log("creating LLM with enforce_eager=True")
        llm = LLM(
            model='/data/models/Qwen3.6-27B-AWQ',
            tensor_parallel_size=2,
            max_model_len=4096,
            gpu_memory_utilization=0.92,
            enforce_eager=True,
        )
        log("LLM created")
        tok = llm.get_tokenizer()

        log("warmup: Hello")
        llm.generate(['Hello'], SamplingParams(max_tokens=8, temperature=0))
        log("warmup done")

        base = ("The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. "
                "The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. "
                "This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. "
                "This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. "
                "The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. "
                "Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. "
                "Eventually, it became obvious that commercial developers and researchers had grossly underestimated the difficulty of the project. "
                "In 1974, in response to the criticism of Sir James Lighthill and ongoing pressure from the US Congress, both the U.S. and British governments cut off exploratory research in AI. "
                "The next few years would later be called an AI winter, a period when obtaining funding for AI projects was difficult. ")
        prompt = base * 12
        ptokens = len(tok.encode(prompt))
        log(f"prompt tokens: {ptokens}")

        log("prefill warmup 1")
        llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
        log("prefill warmup 1 done")

        log("prefill measure 1")
        t0 = time.time()
        llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
        t1 = time.time()
        log(f"PREFILL: {ptokens} tokens in {(t1-t0)*1000:.1f} ms -> {ptokens/(t1-t0):.0f} tok/s")
        log("STATUS: SUCCESS")
    except Exception as e:
        log(f"STATUS: FAILED")
        log(f"ERROR: {e}")
        import traceback
        log(traceback.format_exc())
