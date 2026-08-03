import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_nospec.sqlite")
cur = db.cursor()
states = dict(cur.execute("SELECT id, name FROM ENUM_GPU_CTX_SWITCH"))
blocks = dict(cur.execute("SELECT id, name FROM ENUM_SCHEDULING_THREAD_BLOCK"))

rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(10e9)
iv = [(s, e) for s, e in rows if s >= w0]
gaps = [(iv[i - 1][1], iv[i][0]) for i in range(1, len(iv))
        if iv[i][0] - iv[i - 1][1] > 20_000_000]
gs, ge = gaps[len(gaps)//2]
worker_tid = 319735923395666

evs = list(cur.execute(
    "SELECT start, isSchedIn, threadState, threadBlock FROM SCHED_EVENTS "
    "WHERE globalTid = ? AND start >= ? AND start <= ? ORDER BY start",
    (worker_tid, gs, ge)))
print(f"sched events for worker in gap: {len(evs)}")
agg = {}
prev = None
for st, isi, tstate, tblock in evs:
    if prev is not None:
        d = st - prev[0]
        k = (prev[1], prev[2])
        agg[k] = agg.get(k, 0) + d
    prev = (st, tstate, tblock)
for (tstate, tblock), d in sorted(agg.items(), key=lambda kv: -kv[1]):
    print(f"  {states.get(tstate, tstate):14s}/"
          f"{blocks.get(tblock, tblock):12s} {d/1e6:8.2f}ms")
