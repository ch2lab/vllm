import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_nospec.sqlite")
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))

rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(10e9)
iv = [(s, e) for s, e in rows if s >= w0]
gaps = [(iv[i - 1][1], iv[i][0]) for i in range(1, len(iv))
        if iv[i][0] - iv[i - 1][1] > 20_000_000]
gs, ge = gaps[len(gaps)//2]

# worker thread = the one launching graphs (find via joined string)
row = cur.execute(
    "SELECT r.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
    "JOIN StringIds s ON r.nameId = s.id "
    "WHERE s.value LIKE 'cudaGraphLaunch%' AND r.start >= ? AND r.start <= ? "
    "LIMIT 1", (gs, ge)).fetchone()
worker_tid = row[0]

print(f"worker tid {worker_tid}; full API timeline in gap:")
for st, en, nid in cur.execute(
        "SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "WHERE globalTid = ? AND start >= ? AND start <= ? ORDER BY start",
        (worker_tid, gs, ge)):
    d = (en - st) / 1e6
    if d > 0.1:
        print(f"  @ {(st-gs)/1e6:7.2f} dur {d:7.2f}ms {names.get(nid, '?')}")
# count tiny calls total
n = cur.execute(
    "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
    "WHERE globalTid = ? AND start >= ? AND start <= ?",
    (worker_tid, gs, ge)).fetchone()[0]
print(f"(total API calls in gap: {n})")

# graph launch durations across window
durs = [d for (d,) in cur.execute(
    "SELECT r.end-r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
    "JOIN StringIds s ON r.nameId = s.id "
    "WHERE s.value LIKE 'cudaGraphLaunch%' AND r.start >= ?", (w0,))]
durs.sort()
print(f"\ncudaGraphLaunch in last 10s: n={len(durs)}, "
      f"median {durs[len(durs)//2]/1e6:.2f}ms, "
      f"p90 {durs[int(len(durs)*0.9)]/1e6:.2f}ms, "
      f"max {durs[-1]/1e6:.2f}ms")
