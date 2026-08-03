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
worker_tid = 319735923395666

print("worker thread OSRT calls in gap (>0.2ms):")
for st, en, nid in cur.execute(
        "SELECT start, end, nameId FROM OSRT_API "
        "WHERE globalTid = ? AND start >= ? AND start <= ? "
        "AND end-start > 200000 ORDER BY start", (worker_tid, gs, ge)):
    print(f"  @ {(st-gs)/1e6:7.2f} dur {(en-st)/1e6:7.2f}ms "
          f"{names.get(nid, '?')}")

n_all = cur.execute(
    "SELECT COUNT(*) FROM OSRT_API WHERE globalTid = ? "
    "AND start >= ? AND start <= ?", (worker_tid, gs, ge)).fetchone()[0]
print(f"(total OSRT calls: {n_all})")
