"""Dump stream-29 kernels in a warmup-phase slice vs decode-phase slice."""
import sqlite3

db = sqlite3.connect('/data/src/vllm/nsys_nospec.sqlite')
c = db.cursor()
names = {r[0]: r[1] for r in c.execute("SELECT id, value FROM StringIds")}


def dump(lo_ms, hi_ms, stream, label, limit=60):
    lo, hi = lo_ms * 1_000_000, hi_ms * 1_000_000
    rows = c.execute("""
        SELECT start, end, demangledName, globalPid>>24
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start BETWEEN ? AND ? AND streamId=?
        ORDER BY start
    """, (lo, hi, stream)).fetchall()
    print("=== %s: %d kernels on stream %d ===" % (label, len(rows), stream))
    t0 = rows[0][0] if rows else 0
    for s, e, nid, pid in rows[:limit]:
        nm = names.get(nid, str(nid))
        if len(nm) > 110:
            nm = nm[:110]
        print("  %+9.3fms %8.3fms pid%d %s" % ((s - t0) / 1e6, (e - s) / 1e6, pid, nm))


dump(160_000, 160_100, 29, "warmup slice stream29")
dump(160_000, 160_100, 17, "warmup slice stream17")
dump(200_000, 200_100, 29, "decode slice stream29")
