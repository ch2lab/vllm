"""Per-decode-step kernel breakdown from an eager-mode nsys sqlite trace.

Eager kernels are fully traced (no CUDA-graph lossy sampling), so summing
kernel durations per contiguous GPU burst gives the true GPU-side forward
time per step, plus a class-level breakdown of where the ~57ms goes.
"""
import sqlite3
import sys
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else "nsys_eager.sqlite"
NBURSTS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
GAP_NS = 10_000_000  # 10ms between kernels = step boundary

CLASS_RULES = [
    ("awq_gemv", "AWQ GEMV(M<=2)"),
    ("awq_gemm", "AWQ GEMM"),
    ("fused_recurrent_gated_delta_rule", "GDN recurrent"),
    ("causal_conv1d", "causal conv1d"),
    ("flash_attn_sm70", "SM70 attn"),
    ("rms_norm", "rms_norm"),
    ("nccl", "nccl"),
    ("ConvertToFloat8", "FP8 KV quant"),
    ("silu", "silu"),
    ("chunk", "mamba chunk"),
    ("cutlass", "cutlass"),
    ("argmax", "sampling"),
    ("slot_mapping", "kv prep"),
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
    procs = {g: (pid, nm) for g, pid, nm in
             cur.execute("SELECT globalPid, pid, name FROM PROCESSES").fetchall()}

    bypid: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for gpid, start, end, name_id in cur.execute(
            "SELECT globalPid, start, end, demangledName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        bypid[gpid].append((start, end, name_map.get(name_id, "?")))

    gpid = max(bypid, key=lambda g: sum(ed - st for st, ed, _ in bypid[g]))
    kerns = sorted(bypid[gpid])
    print(f"process: {procs.get(gpid)}  kernels={len(kerns)}")

    bursts = [[kerns[0]]]
    for k in kerns[1:]:
        if k[0] - bursts[-1][-1][1] > GAP_NS:
            bursts.append([k])
        else:
            bursts[-1].append(k)

    print(f"bursts: {len(bursts)}  (last {NBURSTS} = decode steps)\n")

    target = bursts[-NBURSTS:]
    class_totals: dict[str, float] = defaultdict(float)
    top_kernels: dict[str, list] = defaultdict(list)  # name -> [count, total_ns]

    for bi, b in enumerate(target):
        wall = (b[-1][1] - b[0][0]) / 1e6
        busy = sum(ed - st for st, ed, _ in b) / 1e6
        per_class: dict[str, float] = defaultdict(float)
        for s, e, nm in b:
            c = classify(nm)
            per_class[c] += (e - s) / 1e6
            class_totals[c] += (e - s) / 1e6
            tk = top_kernels.setdefault(nm, [0, 0])
            tk[0] += 1
            tk[1] += e - s
        print(f"burst {bi - NBURSTS + 1}: wall={wall:7.2f}ms busy={busy:7.2f}ms "
              f"({100 * busy / wall:5.1f}% gpu) kernels={len(b)}")
        for c, t in sorted(per_class.items(), key=lambda x: -x[1]):
            if t > 0.3:
                print(f"    {c:18s} {t:7.2f}ms ({100 * t / busy:5.1f}%)")

    n = len(target)
    print(f"\n=== per-step average over {n} bursts ===")
    for c, t in sorted(class_totals.items(), key=lambda x: -x[1]):
        if t / n > 0.2:
            print(f"{c:18s} {t / n:7.2f}ms ({100 * t / sum(class_totals.values()):5.1f}%)")

    print("\n=== top 20 kernels by total time ===")
    ranked = sorted(top_kernels.items(), key=lambda x: -x[1][1])[:20]
    for nm, (cnt, tot) in ranked:
        print(f"{tot / 1e6 / n:8.3f}ms/step  {cnt / n:7.1f}/step  {nm[:100]}")


if __name__ == "__main__":
    main()
