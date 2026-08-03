"""Check launchType distribution in decode window, per stream."""
import sqlite3

db = sqlite3.connect('/data/src/vllm/nsys_nospec.sqlite')
c = db.cursor()
names = {r[0]: r[1] for r in c.execute("SELECT id, value FROM StringIds")}

lo, hi = 184_900_000_000, 248_400_000_000
rows = c.execute("""
    SELECT streamId, launchType, COUNT(*)
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE start BETWEEN ? AND ?
    GROUP BY streamId, launchType ORDER BY 1, 2
""", (lo, hi)).fetchall()
for s, lt, n in rows:
    print("stream %d launchType %d: %d" % (s, lt, n))

# Any kernels at all on streams 25/29 in decode?
rows = c.execute("""
    SELECT streamId, COUNT(*), MIN(start)/1e9, MAX(start)/1e9
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE start BETWEEN ? AND ? AND streamId IN (25, 29)
    GROUP BY streamId
""", (lo, hi)).fetchall()
print()
for s, n, t0, t1 in rows:
    print("stream %d: %d kernels, %.3f-%.3f s" % (s, n, t0, t1))

# Also: how many kernels with lt=0 in the FULL trace, and where
rows = c.execute("""
    SELECT streamId, COUNT(*), MIN(start)/1e9, MAX(start)/1e9
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE launchType = 0
    GROUP BY streamId ORDER BY 2 DESC
""").fetchall()
print()
for s, n, t0, t1 in rows:
    print("lt0 stream %d: %d kernels, %.3f-%.3f s" % (s, n, t0, t1))
