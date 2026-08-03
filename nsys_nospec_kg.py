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

print(f"gap {(ge-gs)/1e6:.1f}ms — kernels on dev0 inside:")
for s, e, n in cur.execute(
        "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE deviceId = 0 AND start >= ? AND start <= ? ORDER BY start",
        (gs, ge)):
    print(f"  @ {(s-gs)/1e6:7.2f} dur {(e-s)/1e6:7.3f}ms "
          f"{names.get(n, '?')[:60]}")

print("\nkernels right before gap (last 8):")
for s, e, n in cur.execute(
        "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE deviceId = 0 AND end <= ? ORDER BY end DESC LIMIT 8",
        (gs,)):
    print(f"  ends @ {(s-gs)/1e6:7.2f} dur {(e-s)/1e6:7.3f}ms "
          f"{names.get(n, '?')[:60]}")

print("\nkernels right after gap (first 8):")
for s, e, n in cur.execute(
        "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE deviceId = 0 AND start >= ? ORDER BY start LIMIT 8",
        (ge,)):
    print(f"  @ {(s-gs)/1e6:7.2f} dur {(e-s)/1e6:7.3f}ms "
          f"{names.get(n, '?')[:60]}")
