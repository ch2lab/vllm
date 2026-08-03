import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_decode2.sqlite")
cur = db.cursor()
tables = [t for (t,) in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")]
print([t for t in tables if "THREAD" in t or "TARGET" in t or "PID" in t])
for t in [x for x in tables if "THREAD" in x][:4]:
    cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})")]
    print(t, cols)
    for row in cur.execute(f"SELECT * FROM {t} LIMIT 3"):
        print("  ", row)
