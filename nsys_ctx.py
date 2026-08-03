import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_decode2.sqlite")
cur = db.cursor()
tables = [t for (t,) in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")]
cand = [t for t in tables if "SWITCH" in t or "SAMPLE" in t or "SCHED" in t]
print("tables:", cand)
for t in cand:
    cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})")]
    n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"{t}: {n} rows, cols={cols}")
