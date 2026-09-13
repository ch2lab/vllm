"""Profile prefill to identify time breakdown by kernel/operation."""
import time, traceback, os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
LOG = "/data/src/vllm/test_profile.log"

if __name__ == '__main__':
    with open(LOG, "w") as log:
        try:
            import torch
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

            # Warmup JIT
            llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
            llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))

            # Profile with torch.cuda.profiler
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                for _ in range(3):
                    llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
                    torch.cuda.synchronize()

            # Write full key_averages table
            kernel_table = prof.key_averages()
            kernel_table_sorted = sorted(kernel_table, key=lambda x: -x.cuda_time_total)

            log.write("\n=== TOP 50 CUDA KERNELS BY TOTAL TIME ===\n")
            log.write(f"{'Kernel':<80} {'Count':>6} {'CUDA_total(ms)':>14} {'CUDA_avg(ms)':>13} {'CPU_total(ms)':>13}\n")
            log.write("-" * 130 + "\n")
            for evt in kernel_table_sorted[:50]:
                name = evt.key[:78]
                log.write(f"{name:<80} {evt.count:>6} {evt.cuda_time_total/1000:>14.2f} {evt.cuda_time_average/1000:>13.3f} {evt.cpu_time_total/1000:>13.2f}\n")

            # Categorize kernels
            categories = {
                'awq_gemm': 0,
                'flash_attn_sm70': 0,
                'gdn_chunk': 0,
                'conv1d': 0,
                'all_reduce': 0,
                'rms_norm': 0,
                'silu': 0,
                'rope': 0,
                'kv_cache': 0,
                'other': 0,
            }
            for evt in kernel_table_sorted:
                name = evt.key.lower()
                t = evt.cuda_time_total / 1000  # ms
                if 'awq' in name or 'gemm_sm70' in name or 'sm70tm' in name or 'gemv_sm70' in name:
                    categories['awq_gemm'] += t
                elif 'flash_attn_sm70' in name:
                    categories['flash_attn_sm70'] += t
                elif 'chunk' in name or 'delta_rule' in name or 'gdn' in name or 'fla_' in name:
                    categories['gdn_chunk'] += t
                elif 'conv1d' in name or 'conv' in name:
                    categories['conv1d'] += t
                elif 'all_reduce' in name or 'custom_all_reduce' in name:
                    categories['all_reduce'] += t
                elif 'rms_norm' in name or 'rmsnorm' in name or 'layer_norm' in name:
                    categories['rms_norm'] += t
                elif 'silu' in name or 'activation' in name:
                    categories['silu'] += t
                elif 'rope' in name or 'rotary' in name:
                    categories['rope'] += t
                elif 'kv' in name or 'cache' in name:
                    categories['kv_cache'] += t
                else:
                    categories['other'] += t

            total_cuda = sum(categories.values())
            log.write(f"\n=== KERNEL CATEGORIES (3 iterations total) ===\n")
            log.write(f"{'Category':<25} {'Time(ms)':>10} {'Percent':>8}\n")
            log.write("-" * 45 + "\n")
            for cat, t in sorted(categories.items(), key=lambda x: -x[1]):
                pct = t / total_cuda * 100 if total_cuda > 0 else 0
                log.write(f"{cat:<25} {t:>10.1f} {pct:>7.1f}%\n")
            log.write(f"{'TOTAL':<25} {total_cuda:>10.1f} {'100.0%':>8}\n")
            log.write(f"\nPer-iteration estimate: {total_cuda/3:.1f} ms\n")

            log.write("\nSTATUS: SUCCESS\n")
        except Exception as e:
            log.write(f"STATUS: FAILED\nERROR: {e}\n")
            log.write(traceback.format_exc())
    print(f"Done. See {LOG}")
