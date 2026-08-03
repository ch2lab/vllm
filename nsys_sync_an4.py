"""RUNTIME sync API calls in decode window, per thread, for worker 2280529."""
import sqlite3

con = sqlite3.connect('/data/src/vllm/nsys_nospec.sqlite')
cur = con.cursor()
PID = 319735904337920
T0 = 178000e6

cur.execute("""
    SELECT r.start/1e6, (r.end-r.start)/1e6, r.globalTid, s.value
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN StringIds s ON r.nameId = s.id
    WHERE r.globalTid IN (SELECT globalTid FROM ThreadNames
                          WHERE globalTid IN (SELECT globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME)
                          AND globalTid < 1000000000000000)
      AND r.start > ?
    ORDER BY r.start
""", (T0,))
rows = cur.fetchall()
print(f"runtime calls in window: {len(rows)}")

from collections import Counter
names = Counter(r[3] for r in rows)
for k, v in names.most_common(12):
    print(f"  {k}: {v}")
