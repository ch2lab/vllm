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
gs, ge = gaps[len(gaps) // 2]

sync_tid = None
for st, en, nid, tid in cur.execute(
        "SELECT start, end, nameId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "WHERE start >= ? AND start <= ? AND end-start > 20000000", (gs, ge)):
    sync_tid = tid
print(f"sync tid={sync_tid}")

print("API sequence on sync thread, gap_start-2ms .. gap_end+2ms:")
for st, en, nid in cur.execute(
        "SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "WHERE globalTid = ? AND start >= ? AND start <= ? ORDER BY start",
        (sync_tid, gs - 2_000_000, ge + 2_000_000)):
    nm = names.get(nid, "?")
    print(f"  @ {(st-gs)/1e6:8.2f}ms  dur {(en-st)/1e6:8.2f}ms  {nm}")
