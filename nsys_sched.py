import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_decode2.sqlite")
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))

rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(4.6e9)
iv = [(s, e) for s, e in rows if s >= w0]
gaps = [(iv[i - 1][1], iv[i][0]) for i in range(1, len(iv))
        if iv[i][0] - iv[i - 1][1] > 50_000_000]

states = dict(cur.execute("SELECT id, name FROM ENUM_GPU_CTX_SWITCH"))
blocks = dict(cur.execute("SELECT id, name FROM ENUM_SCHEDULING_THREAD_BLOCK"))

sync_tid = 319412307656953
gs, ge = gaps[len(gaps) // 2]

print(f"gap {(ge-gs)/1e6:.1f}ms; sync thread states:")
agg = {}
for st, cpu, isi, tid, tstate, tblock in cur.execute(
        "SELECT start, cpu, isSchedIn, globalTid, threadState, threadBlock "
        "FROM SCHED_EVENTS WHERE globalTid = ? AND start >= ? AND start <= ? "
        "ORDER BY start", (sync_tid, gs - 5_000_000, ge + 5_000_000)):
    agg.setdefault((tstate, tblock), [0, 0])
    agg[(tstate, tblock)][0] += 1
print({f"{states.get(s, s)}/{blocks.get(b, b)}": v[0]
       for (s, b), v in agg.items()})

# durations: pair sched-in/out events
evs = list(cur.execute(
    "SELECT start, isSchedIn, threadState, threadBlock FROM SCHED_EVENTS "
    "WHERE globalTid = ? AND start >= ? AND start <= ? ORDER BY start",
    (sync_tid, gs - 2_000_000, ge + 2_000_000)))
prev = None
print("\nthread-state timeline (dur ms, state/block):")
for st, isi, tstate, tblock in evs:
    if prev is not None:
        d = (st - prev[0]) / 1e6
        if d > 0.3:
            print(f"  {d:8.2f}ms  {states.get(prev[1], prev[1])}/"
                  f"{blocks.get(prev[2], prev[2])}")
    prev = (st, tstate, tblock)
