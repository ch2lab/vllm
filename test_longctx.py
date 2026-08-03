"""Measure acceptance with longer context (closer to 1Cat's benchmark)."""
import os, sys, time

os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
os.environ["VLLM_SM70_FUSED_AR_RMSNORM"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

LOG = "/data/src/vllm/test_longctx.log"

if __name__ == '__main__':
    with open(LOG, "w") as log:
        try:
            from vllm import LLM, SamplingParams
            llm = LLM(
                model='/data/models/Qwen3.6-27B-AWQ',
                tensor_parallel_size=2,
                max_model_len=8192,
                gpu_memory_utilization=0.92,
                speculative_config={
                    "method": "mtp",
                    "num_speculative_tokens": 4,
                },
                disable_log_stats=False,
            )
            # Warmup
            for _ in range(3):
                llm.generate(['warmup'], SamplingParams(max_tokens=16, temperature=0))

            # Long prompt (~2000 tokens) to simulate longer context
            long_prompt = """The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation. They were given millions of dollars to make this vision come true. Eventually, it became obvious that commercial developers and researchers had grossly underestimated the difficulty of the project. In 1974, in response to the criticism of Sir James Lighthill and ongoing pressure from congress, both the U.S. and British governments cut off exploratory research in AI. Seven years later, a visionary initiative by the Japanese Government inspired governments and industry to provide AI with billions of dollars, but by the late 1980s the investors became disillusioned by the absence of the needed computer power (hardware) and withdrew funding again. Investment and interest in AI boomed in the first decades of the 21st century when machine learning was successfully applied to many problems in academia and industry due to the presence of powerful computer hardware and the collection of immense data sets. """ * 4

            # Generate long output (timed: long-context decode speed)
            import time as _t
            _start = _t.time()
            out = llm.generate([long_prompt], SamplingParams(max_tokens=512, temperature=0))
            _elapsed = _t.time() - _start
            n = len(out[0].outputs[0].token_ids)
            log.write(f"Generated {n} tokens from long prompt\n")
            log.write(f"LONGCTX: {n} tokens in {_elapsed:.2f}s = {n/_elapsed:.1f} tok/s "
                      f"({_elapsed*1000/n:.1f} ms/tok)\n")

            # Second generation (warmed up KV cache path)
            out2 = llm.generate(['Continue the discussion about AI safety and alignment research.'],
                               SamplingParams(max_tokens=512, temperature=0))
            n2 = len(out2[0].outputs[0].token_ids)
            log.write(f"Generated {n2} tokens (second gen)\n")

            log.write("STATUS: SUCCESS\n")
        except Exception as e:
            import traceback
            log.write(f"STATUS: FAILED\n")
            log.write(f"ERROR: {type(e).__name__}: {e}\n")
            log.write(traceback.format_exc())

    print(f"Done. See {LOG}")
