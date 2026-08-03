"""Full per-step window: sync + kernels + memcpys + runtime calls.
Anchor: sync at t=184928.8ms (60.1ms). Window 184900-184990ms."""
import sqlite3

con = sqlite3.connect('/data/src/vllm/nsys_nospec.sqlite')
cur = con.cursor()
PID = 319735904337920
W0, W1 = 184900e6, 184990e6
SYNC_TID = 319735906619051
LAUNCH_TID = 319735923395666

print("=== MEMCPY in window ===")
cur.execute("""
    SELECT start/1e6, (end-start)/1e6, deviceId, streamId, globalPid
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    WHERE start >= ? AND start <= ? AND globalPid = ?
    ORDER BY start
""", (W0, W1, PID))
for r in cur.fetchall():
    print(f"  memcpy t={r[0]:.3f} dur={r[1]:.3f}ms dev={r[2]} strm={r[3]}")

print("=== KERNELS in window ===")
cur.execute("""
    SELECT start/1e6, (end-start)/1e6, streamId, shortName
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE start >= ? AND start <= ? AND globalPid = ?
    ORDER BY start
""", (W0, W1, PID))
for r in cur.fetchall():
    print(f"  kern t={r[0]:.3f} dur={r[1]:.3f}ms strm={r[2]} {r[3][:60]}")

print("=== RUNTIME on sync thread ===")
cur.execute("""
    SELECT start/1e6, (end-start)/1e6, s.value
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id
    WHERE r.globalTid = ? AND r.start >= ? AND r.start <= ?
    ORDER BY r.start
""", (SYNC_TID, W0, W1))
for r in cur.fetchall():
    print(f"  {r[0]:.3f} +{r[1]:.3f}ms {r[2]}")

print("=== RUNTIME on launch thread ===")
cur.execute("""
    SELECT start/1e6, (end-start)/1e6, s.value
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id
    WHERE r.globalTid = ? AND r.start >= ? AND r.start <= ?
    ORDER BY r.start
""", (LAUNCH_TID, W0, W1))
for r in cur.fetchall():
    print(f"  {r[0]:.3f} +{r[1]:.3f}ms {r[2]}")
