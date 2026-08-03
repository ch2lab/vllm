"""Analyze decode-window kernel time from nsys sqlite."""
import sqlite3
import sys

db = sqlite3.connect('/tmp/decode16k.sqlite')
c = db.cursor()
names = {r[0]: r[1] for r in c.execute("SELECT id, value FROM StringIds")}

# Find the decode window: last N seconds of the trace (decode dominates tail).
# MTP decode runs 256 tokens; total ~38s. Prefill ~16s. Decode window ~ last 10s.
rows = c.execute("SELECT MIN(start), MAX(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
lo, hi = rows
print(f"trace range: {lo/1e9:.1f}s - {hi/1e9:.1f}s")
# decode window = last 12s
win_start = hi - 12e9

# Aggregate kernel time in decode window, by kernel name
agg = {}
for r in c.execute("""
    SELECT demangledName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE start >= ? AND deviceId = 0
""", (win_start,)):
    nid, s, e = r
    nm = names.get(nid, str(nid))
    if nm not in agg:
        agg[nm] = 0.0
    agg[nm] += (e - s) / 1e6  # ms

total = sum(agg.values())
print(f"decode window total: {total:.1f} ms, kernels: {len(agg)}")
print(f"{'kernel':<60} {'ms':>10} {'%':>6}")
for nm, ms in sorted(agg.items(), key=lambda x: -x[1])[:25]:
    print(f"{nm:<60} {ms:>10.1f} {100*ms/total:>5.1f}%")