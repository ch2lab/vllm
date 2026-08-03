"""Benchmark prefill speed with various prompt lengths."""
import time, traceback, os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
LOG = "/data/src/vllm/test_prefill.log"

if __name__ == '__main__':
    with open(LOG, "w") as log:
        try:
            from vllm import LLM, SamplingParams
            llm = LLM(
                model='/data/models/Qwen3.6-27B-AWQ',
                tensor_parallel_size=2,
                max_model_len=4096,
                gpu_memory_utilization=0.92,
            )
            tok = llm.get_tokenizer()
            # Warmup
            llm.generate(['Hello'], SamplingParams(max_tokens=8, temperature=0))

            base = ("The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. "
                    "The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. "
                    "This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. "
                    "This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. "
                    "The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. "
                    "Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. "
                    "Eventually, it became obvious that commercial developers and researchers had grossly underestimated the difficulty of the project. "
                    "In 1974, in response to the criticism of Sir James Lighthill and ongoing pressure from the US Congress, both the U.S. and British governments cut off exploratory research in AI. "
                    "The next few years would later be called an AI winter, a period when obtaining funding for AI projects was difficult. ")
            repeat = 12
            prompt = base * repeat
            prompt_tokens = len(tok.encode(prompt))
            log.write(f"PROMPT_TOKENS: {prompt_tokens}\n")

            # Warmup JIT with this prompt
            llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
            llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))

            # Measure prefill (max_tokens=1 to isolate prefill time)
            times = []
            for _ in range(5):
                start = time.time()
                llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
                times.append(time.time() - start)
            avg = sum(times) / len(times)
            mn = min(times)
            log.write(f"PREFILL_AVG: {prompt_tokens} tokens in {avg*1000:.1f} ms\n")
            log.write(f"PREFILL_MIN: {prompt_tokens} tokens in {mn*1000:.1f} ms\n")
            log.write(f"PREFILL_TOK_S_AVG: {prompt_tokens/avg:.0f} tok/s\n")
            log.write(f"PREFILL_TOK_S_MIN: {prompt_tokens/mn:.0f} tok/s\n")
            log.write("STATUS: SUCCESS\n")
        except Exception as e:
            log.write(f"STATUS: FAILED\nERROR: {e}\n")
            log.write(traceback.format_exc())
    print(f"Done. See {LOG}")
