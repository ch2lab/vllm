"""Histogram graph exec durations and per-exec kernel counts."""
import sqlite3

db = sqlite3.connect('/data/src/vllm/nsys_nospec.sqlite')
c = db.cursor()

rows = c.execute("""
    SELECT start, end, globalPid>>24 FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
    WHERE graphId=109 ORDER BY start
""").fetchall()
print("Total execs:", len(rows))
durs = {}
for s, e, p in rows:
    d = (e - s) / 1e6
    b = int(d / 10) * 10
    durs[b] = durs.get(b, 0) + 1
for b in sorted(durs):
    print("  dur %3d-%3d ms: %d execs" % (b, b + 10, durs[b]))

# Per-exec kernel counts: kernels between consecutive exec starts
# group stream-17 kernels by which exec window they fall in (by start time)
krows = c.execute("""
    SELECT start, end, globalPid>>24 FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE streamId=17 AND start BETWEEN 184900000000 AND 248400000000
    ORDER BY start
""").fetchall()
execs = rows
import bisect
starts = [s for s, e, p in execs]
counts = {}
for s, e, p in krows:
    i = bisect.bisect_right(starts, s) - 1
    if i >= 0:
        counts[i] = counts.get(i, 0) + 1
# exclude first/last partial windows
from collections import Counter
hist = Counter(counts.get(i, 0) for i in range(len(execs)))
print("Per-exec kernel count histogram (both processes combined):")
for n in sorted(hist):
    print("  %4d kernels: %d execs" % (n, hist[n]))
