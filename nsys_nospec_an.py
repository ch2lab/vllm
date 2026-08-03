import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_nospec.sqlite")
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))

rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(10e9)  # last 10s covers steady decode
iv = [(s, e) for s, e in rows if s >= w0]
span = iv[-1][1] - w0

busy = 0
cs, ce = iv[0]
for s, e in iv[1:]:
    if s <= ce:
        ce = max(ce, e)
    else:
        busy += ce - cs
        cs, ce = s, e
busy += ce - cs
print(f"dev0 last-10s: {len(iv)} kernels, busy {busy/1e9:.2f}s = "
      f"{busy/span*100:.1f}%")

gaps = [(iv[i - 1][1], iv[i][0]) for i in range(1, len(iv))
        if iv[i][0] - iv[i - 1][1] > 20_000_000]
print(f"gaps >20ms: {len(gaps)}, sum {sum(g-s for s, g in gaps)/1e9:.2f}s")
if gaps:
    gs, ge = gaps[len(gaps)//2]
    print(f"\nmid gap {(ge-gs)/1e6:.1f}ms; RUNTIME >5ms inside:")
    for st, en, nid, tid in cur.execute(
            "SELECT start, end, nameId, globalTid "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            "WHERE start >= ? AND start <= ? AND end-start > 5000000 "
            "ORDER BY start", (gs, ge)):
        print(f"  t={(st-gs)/1e6:6.1f} dur={(en-st)/1e6:6.1f}ms "
              f"{names.get(nid, '?')}")
    # kernel burst right before and after
    print("OSRT >5ms inside gap:")
    for st, en, nid in cur.execute(
            "SELECT start, end, nameId FROM OSRT_API "
            "WHERE start >= ? AND start <= ? AND end-start > 5000000 "
            "ORDER BY end-start DESC LIMIT 8", (gs, ge)):
        print(f"  {(en-st)/1e6:8.1f}ms  {names.get(nid, '?')}")
