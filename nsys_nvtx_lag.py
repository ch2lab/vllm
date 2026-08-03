import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_nvtx.sqlite")
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))

rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(30e9)

sched_tid = 319912939812233
worker_tid = 319916681131624
engine_tid = 319906161816565

# worker preprocess starts
pre = [nm for nm in ()]
pres = sorted(st for st, en, txtid in cur.execute(
    "SELECT start, end, textId FROM NVTX_EVENTS WHERE globalTid = ? "
    "AND start >= ? AND textId = (SELECT id FROM StringIds WHERE "
    "value='gpu_model_runner: preprocess')", (worker_tid, w0)))

# scheduler allocate_slots (proxy for schedule completion)
sched = sorted(st for st, in cur.execute(
    "SELECT start FROM NVTX_EVENTS WHERE globalTid = ? AND start >= ? "
    "AND textId = (SELECT id FROM StringIds WHERE "
    "value='schedule: allocate_slots')", (sched_tid, w0))) if False else \
    sorted(st for (st,) in cur.execute(
    "SELECT start FROM NVTX_EVENTS WHERE globalTid = ? AND start >= ? "
    "AND textId = (SELECT id FROM StringIds WHERE "
    "value='schedule: allocate_slots')", (sched_tid, w0)))

# engine get_output range ENDs (moment output is received)
go_ends = sorted(en for (en,) in cur.execute(
    "SELECT end FROM NVTX_EVENTS WHERE globalTid = ? AND start >= ? "
    "AND end IS NOT NULL AND textId = (SELECT id FROM StringIds WHERE "
    "value='llm_engine step: get_output')", (engine_tid, w0)))

import bisect
print("idx  sched->pre_delay  getout_end->pre_delay")
for i in range(len(pres) // 2, len(pres) // 2 + 12):
    p = pres[i]
    js = bisect.bisect_right(sched, p) - 1
    jg = bisect.bisect_right(go_ends, p) - 1
    ds = (p - sched[js]) / 1e6 if js >= 0 else -1
    dg = (p - go_ends[jg]) / 1e6 if jg >= 0 else -1
    print(f"{i:4d}   {ds:10.2f}ms   {dg:10.2f}ms")
