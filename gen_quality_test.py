"""Decisive test: does cudagraph-mode decode produce real text or garbage?

Runs one generation with max_tokens=30 and prints the output text verbatim.
Usage: python3 gen_quality_test.py [eager|cudagraph]
"""
import os
import sys
import time

mode = sys.argv[1] if len(sys.argv) > 1 else "cudagraph"
LOG = "/data/src/vllm/gen_quality_%s.log" % mode
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"


def main():
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model='/data/models/Qwen3.6-27B-AWQ',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        kv_cache_dtype='fp8',
        max_model_len=32768,
        disable_log_stats=True,
    )
    if mode == "eager":
        kwargs["enforce_eager"] = True

    with open(LOG, "w") as log:
        llm = LLM(**kwargs)
        sp = SamplingParams(max_tokens=30, temperature=0)

        for _ in range(3):
            llm.generate(['warmup'], SamplingParams(max_tokens=8, temperature=0))

        t0 = time.time()
        out = llm.generate(
            ['Explain the theory of general relativity in one paragraph.'],
            sp,
        )
        elapsed = time.time() - t0
        text = out[0].outputs[0].text
        ids = out[0].outputs[0].token_ids
        log.write("ELAPSED: %.2fs for %d tokens\n" % (elapsed, len(ids)))
        log.write("TOKEN_IDS: %s\n" % str(ids))
        log.write("OUTPUT_TEXT_BEGIN\n%s\nOUTPUT_TEXT_END\n" % text)
        log.write("STATUS: SUCCESS\n")
        print("Done. See %s" % LOG)


if __name__ == '__main__':
    main()
