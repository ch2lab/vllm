import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_decode2.sqlite")
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))

rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(4.6e9)

for pat in ("topk", "rejection", "eagle", "flash_attn_sm70", "cross_device"):
    tot, cnt, mx = 0, 0, 0
    for s, e, n in cur.execute(
            "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE deviceId = 0 AND start >= ?", (w0,)):
        nm = names.get(n, "")
        if pat in nm:
            tot += e - s
            cnt += 1
            mx = max(mx, e - s)
    print(f"{pat:18s} n={cnt:5d} total={tot/1e6:8.1f}ms "
          f"avg={tot/max(cnt,1)/1e3:7.1f}us max={mx/1e6:6.1f}ms")
