import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_decode.sqlite")
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))

rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(4.6e9)
iv = [(s, e) for s, e in rows if s >= w0]

# pick three representative gaps mid-run
gaps = []
for i in range(1, len(iv)):
    g = iv[i][0] - iv[i - 1][1]
    if g > 50_000_000:
        gaps.append((iv[i - 1][1], iv[i][0]))

tables = [t for (t,) in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")]
print("tables:", [t for t in tables if "RUNTIME" in t or "OSRT" in t])

for gs, ge in gaps[10:13]:
    print(f"\n=== gap {(ge-gs)/1e6:.1f}ms [{gs}, {ge}] ===")
    for tbl in ("CUPTI_ACTIVITY_KIND_RUNTIME",):
        for st, en, nid in cur.execute(
                f"SELECT start, end, nameId FROM {tbl} "
                f"WHERE start >= ? AND start <= ? AND end > ? "
                f"ORDER BY (end-start) DESC LIMIT 8",
                (gs, ge, gs + 1_000_000)):
            nm = names.get(nid, "?")
            print(f"  {tbl.split('_')[-1]:8s} {(en-st)/1e6:8.1f}ms  {nm}")
