import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_nvtx.sqlite")
cur = db.cursor()
tables = [t for (t,) in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%NVTX%'")]
print("nvtx tables:", tables)
cols = [c[1] for c in cur.execute("PRAGMA table_info(NVTX_EVENTS)")]
print("cols:", cols)

# aggregate ranges by text, steady-state window only
rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(30e9)

names = dict(cur.execute("SELECT id, value FROM StringIds"))
acc = {}
for txtid, st, en in cur.execute(
        "SELECT textId, start, end FROM NVTX_EVENTS "
        "WHERE start >= ? AND end IS NOT NULL", (w0,)):
    nm = names.get(txtid, str(txtid))
    a = acc.setdefault(nm, [0, 0])
    a[0] += en - st
    a[1] += 1
print("\nNVTX phase totals (last 30s):")
for nm, (t, c) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
    print(f"  {t/1e6:10.1f} ms  x{c:5d}  {nm}")
