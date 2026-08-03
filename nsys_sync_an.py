"""Locate cudaEventSynchronize / cudaStreamSynchronize calls in the worker
process of the nospec capture, relative to step boundaries."""
import sqlite3

con = sqlite3.connect('/data/src/vllm/nsys_nospec.sqlite')
cur = con.cursor()

cur.execute("SELECT globalPid, pid, name FROM PROCESSES")
procs = {p[0]: (p[1], p[2]) for p in cur.fetchall()}

cur.execute("""
    SELECT e.start, e.end, e.end - e.start AS dur, e.globalPid, s.value
    FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION e
    JOIN StringIds s ON e.syncType = s.id
    WHERE e.start > 0
    ORDER BY e.start
""")
rows = cur.fetchall()
print(f"SYNC EVENTS: {len(rows)}")
from collections import Counter
bypid = Counter(r[3] for r in rows)
for pid, n in bypid.most_common():
    print(f"  pid={pid} ({procs.get(pid)}) n={n}")
