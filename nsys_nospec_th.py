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

# find the thread doing the big event sync
sync_tid = cur.execute(
    "SELECT globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME "
    "WHERE start >= ? AND start <= ? AND end-start > 50000000",
    (gs, ge)).fetchone()[0]
print(f"sync thread: {sync_tid}")

# all API calls from ALL threads in the gap, sorted by start
print("\nALL threads' API calls in gap (dur>0.1ms):")
for st, en, nid, tid in cur.execute(
        "SELECT start, end, nameId, globalTid "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "WHERE start >= ? AND start <= ? AND end-start > 100000 "
        "ORDER BY start", (gs, ge)):
    tag = "SYNC_T" if tid == sync_tid else "other "
    print(f"  {tag} @ {(st-gs)/1e6:7.2f} dur {(en-st)/1e6:7.2f}ms "
          f"{names.get(nid, '?')}")

# and what did the sync thread do in the 10ms before the gap
print("\nsync thread API just before gap (-15ms..0):")
for st, en, nid in cur.execute(
        "SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "WHERE globalTid = ? AND start >= ? AND start <= ? ORDER BY start",
        (sync_tid, gs - 15_000_000, gs)):
    print(f"  @ {(st-gs)/1e6:7.2f} dur {(en-st)/1e6:7.2f}ms {names.get(nid, '?')}")
