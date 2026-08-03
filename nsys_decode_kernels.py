"""List ALL distinct kernels on stream 17 during the decode window."""
import sqlite3

db = sqlite3.connect('/data/src/vllm/nsys_nospec.sqlite')
c = db.cursor()
names = {r[0]: r[1] for r in c.execute("SELECT id, value FROM StringIds")}

lo, hi = 184_900_000_000, 248_400_000_000
rows = c.execute("""
SELECT demangledName, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL
WHERE start BETWEEN ? AND ? AND streamId=17 GROUP BY demangledName ORDER BY 2 DESC
""", (lo, hi)).fetchall()
print("Distinct kernel names in decode window: %d" % len(rows))
for nid, cnt in rows:
    nm = names.get(nid, str(nid))
    print("  %6d  %s" % (cnt, nm))
