import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_decode2.sqlite")
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))
tables = [t for (t,) in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")]
print("osrt-ish tables:", [t for t in tables
                          if "OSRT" in t or "CTX" in t or "THREAD" in t])

rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(4.6e9)
iv = [(s, e) for s, e in rows if s >= w0]
gaps = [(iv[i - 1][1], iv[i][0]) for i in range(1, len(iv))
        if iv[i][0] - iv[i - 1][1] > 50_000_000]
print(f"gaps: {len(gaps)}")
gs, ge = gaps[len(gaps) // 2]

if "OSRT_API" in tables:
    print(f"\nOSRT >0.5ms inside gap {(ge-gs)/1e6:.1f}ms:")
    for st, en, nid in cur.execute(
            "SELECT start, end, nameId FROM OSRT_API "
            "WHERE start >= ? AND start <= ? AND end-start > 500000 "
            "ORDER BY end-start DESC LIMIT 15", (gs, ge)):
        print(f"  {(en-st)/1e6:8.1f}ms  {names.get(nid, '?')}")

print("\nRUNTIME >20ms inside gap (all processes):")
for st, en, nid, tid in cur.execute(
        "SELECT start, end, nameId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "WHERE start >= ? AND start <= ? AND end-start > 20000000 "
        "ORDER BY start", (gs, ge)):
    print(f"  t={(st-gs)/1e6:6.1f} dur={(en-st)/1e6:6.1f}ms "
          f"{names.get(nid, '?')} tid={tid}")
