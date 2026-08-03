"""Analyze sync event types and durations in worker 2280529 over time."""
import sqlite3

con = sqlite3.connect('/data/src/vllm/nsys_nospec.sqlite')
cur = con.cursor()
PID = 319735904337920  # worker 2280529

cur.execute("""
    SELECT e.start, e.end, e.end - e.start AS dur, s.value
    FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION e
    JOIN StringIds s ON e.syncType = s.id
    WHERE e.globalPid = ? AND e.start > 0
    ORDER BY e.start
""", (PID,))
rows = cur.fetchall()
print(f"total sync: {len(rows)}")

from collections import Counter
types = Counter(r[3] for r in rows)
print("types:", dict(types))

# Distribution of durations (ms)
durs = [r[2] / 1e6 for r in rows]
big = [(r[0] / 1e6, r[2] / 1e6, r[3]) for r in rows if r[2] > 1e7]
print(f"syncs > 10ms: {len(big)}")

# Show first 40 big syncs
for t, d, api in big[:40]:
    print(f"  t={t:.3f}ms dur={d:.2f}ms {api}")
