"""Analyze nsys_decode.sqlite: isolate the decode window and report
GPU busy/idle breakdown and per-kernel shares."""
import sqlite3
from collections import defaultdict

db = sqlite3.connect("/data/src/vllm/nsys_decode.sqlite")
cur = db.cursor()

# kernel name table
names = {}
for sid, val in cur.execute("SELECT id, value FROM StringIds"):
    names[sid] = val

rows = list(cur.execute(
    "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
    "WHERE deviceId = 0"))
rows.sort()
print(f"total kernel records: {len(rows)}")
t_end = rows[-1][1]

# Decode window: fixed trailing 4.6s wall clock (the timed generate call).
w0 = t_end - int(4.6e9)
win = [(s, e, n) for s, e, n in rows if s >= w0]
w1 = win[-1][1]
span = (w1 - w0) / 1e9
print(f"decode window: {len(win)} kernels over {span:.2f}s")

# busy time (merge overlaps)
iv = sorted((s, e) for s, e, _ in win)
busy = 0
cs, ce = iv[0]
for s, e in iv[1:]:
    if s <= ce:
        ce = max(ce, e)
    else:
        busy += ce - cs
        cs, ce = s, e
busy += ce - cs
print(f"GPU busy: {busy / 1e9:.2f}s = {busy / (w1 - w0) * 100:.1f}% of window")

# top gaps
gaps = []
for i in range(1, len(iv)):
    g = iv[i][0] - iv[i - 1][1]
    if g > 0:
        gaps.append((g, iv[i - 1][1]))
gaps.sort(reverse=True)
tot_gap = sum(g for g, _ in gaps)
print(f"total idle: {tot_gap / 1e9:.2f}s; top gaps (ms): "
      + ", ".join(f"{g / 1e6:.1f}" for g, _ in gaps[:10]))

# per-kernel shares in window
acc = defaultdict(lambda: [0, 0])
for s, e, n in win:
    nm = names.get(n, str(n))
    # shorten template noise
    short = nm.split("(")[0]
    if len(short) > 70:
        short = short[:70]
    acc[short][0] += e - s
    acc[short][1] += 1
top = sorted(acc.items(), key=lambda kv: -kv[1][0])[:20]
print("\nper-kernel in decode window:")
for nm, (t, c) in top:
    print(f"  {t / 1e6:9.1f} ms  {t / busy * 100:5.1f}%  x{c:5d}  {nm}")
