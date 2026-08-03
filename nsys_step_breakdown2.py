"""Per-decode-step breakdown using nccl AllGather kernels as step anchors.

In the eager TP2 non-MTP run, each decode step contains exactly one nccl
AllGather. Consecutive AllGathers are ~187ms apart in wall time; big gaps
(>500ms) separate the warmup generations (8+8+8 steps) from the final
30-token generation. Each step window = [AG_i.start, AG_{i+1}.start]; all
kernels of the worker process whose start falls in the window are summed
per class, giving the true GPU-side forward+sampling cost per step.
"""
import sqlite3
import sys
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else "nsys_eager.sqlite"

CLASS_RULES = [
    ("cross_device_reduce", "custom allreduce"),
    ("awq_gemv", "AWQ GEMV(M<=2)"),
    ("awq_gemm", "AWQ GEMM"),
    ("split_k_reduce", "AWQ split-K reduce"),
    ("gemv2T", "gemv2T (f16?)"),
    ("fused_recurrent_gated_delta_rule", "GDN recurrent"),
    ("causal_conv1d", "causal conv1d"),
    ("flash_attn_sm70", "SM70 attn"),
    ("rms_norm", "rms_norm"),
    ("nccl", "nccl"),
    ("ConvertToFloat8", "FP8 KV quant"),
    ("silu", "silu"),
    ("chunk", "mamba chunk"),
    ("cutlass", "cutlass"),
    ("direct_copy", "direct_copy"),
    ("reduce_kernel", "reduce_kernel"),
    ("rsqrt", "rsqrt"),
    ("vectorized_elementwise", "vec elementwise"),
    ("elementwise_kernel", "elementwise"),
]


def classify(name: str) -> str:
    for key, label in CLASS_RULES:
        if key in name:
            return label
    return "other"


def main() -> None:
    con = sqlite3.connect(DB)
    cur = con.cursor()
    name_map = dict(cur.execute("SELECT id, value FROM StringIds").fetchall())

    bypid: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for gpid, start, end, name_id in cur.execute(
            "SELECT globalPid, start, end, demangledName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        bypid[gpid].append((start, end, name_map.get(name_id, "?")))
    gpid = max(bypid, key=lambda g: sum(ed - st for st, ed, _ in bypid[g]))
    kerns = sorted(bypid[gpid])

    ag = [(s, e) for s, e, n in kerns if "AllGather" in n]
    # split into clusters by >500ms gaps; keep the last cluster (30-token gen)
    clusters: list[list] = [[ag[0]]]
    for a in ag[1:]:
        if a[0] - clusters[-1][-1][1] > 500_000_000:
            clusters.append([a])
        else:
            clusters[-1].append(a)
    cluster = clusters[-1]
    if len(sys.argv) > 2:
        cluster = cluster[-int(sys.argv[2]):]
    print(f"AllGather clusters: {[len(c) for c in clusters]}  "
          f"using last ({len(cluster)} steps)")

    windows: list[tuple[int, int]] = []
    for i in range(len(cluster) - 1):
        windows.append((cluster[i][0], cluster[i + 1][0]))

    class_totals: dict[str, float] = defaultdict(float)
    per_kernel: dict[str, tuple[int, float]] = {}
    for wstart, wend in windows:
        for s, e, nm in kerns:
            if s < wstart:
                continue
            if s >= wend:
                break
            c = classify(nm)
            class_totals[c] += (e - s) / 1e6
            cnt, tot = per_kernel.get(nm, (0, 0.0))
            per_kernel[nm] = (cnt + 1, tot + (e - s) / 1e6)

    n = len(windows)
    total = sum(class_totals.values())
    print(f"\n=== per-step average over {n} steps (GPU busy {total / n:.2f}ms/step) ===")
    for c, t in sorted(class_totals.items(), key=lambda x: -x[1]):
        if t / n > 0.1:
            print(f"{c:22s} {t / n:8.2f}ms ({100 * t / total:5.1f}%)")

    print("\n=== top 15 kernels ===")
    ranked = sorted(per_kernel.items(), key=lambda x: -x[1][1])[:15]
    for nm, (cnt, tot) in ranked:
        print(f"{tot / n:8.3f}ms/step  {cnt / n:6.1f}/step  {nm[:95]}")


if __name__ == "__main__":
    main()
