"""Sync events in the decode window (last ~70s) of worker 2280529."""
import sqlite3

con = sqlite3.connect('/data/src/vllm/nsys_nospec.sqlite')
cur = con.cursor()
PID = 319735904337920
T0 = 178000e6  # decode window start guess (ns)

cur.execute("""
    SELECT start/1e6, (end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
    WHERE globalPid = ? AND start > ? AND start > 0
    ORDER BY start
""", (PID, T0))
rows = cur.fetchall()
print(f"syncs in window: {len(rows)}")

from collections import Counter
hist = Counter()
for t, d in rows:
    bucket = int(d // 5) * 5
    hist[f"{bucket}-{bucket+5}ms"] += 1
for k in sorted(hist):
    print(f"  {k}: {hist[k]}")
