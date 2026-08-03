import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_decode.sqlite")
cur = db.cursor()

tables = [t for (t,) in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")]
print([t for t in tables if "OSRT" in t])

rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(4.6e9)

for dev in (0, 1):
    iv = sorted((s, e) for s, e in cur.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
        f"WHERE deviceId = {dev} AND start >= {w0}"))
    busy = 0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            busy += ce - cs
            cs, ce = s, e
    busy += ce - cs
    span = iv[-1][1] - w0
    print(f"device {dev}: {len(iv)} kernels, busy {busy/1e9:.2f}s "
          f"= {busy/span*100:.1f}%")

# device-0 gap #15 vs device-1 activity in the same interval
iv0 = sorted(cur.execute(
    f"SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
    f"WHERE deviceId = 0 AND start >= {w0}"))
gaps = [(iv0[i - 1][1], iv0[i][0]) for i in range(1, len(iv0))
        if iv0[i][0] - iv0[i - 1][1] > 50_000_000]
gs, ge = gaps[15]
busy1 = sum(e - s for s, e in cur.execute(
    f"SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
    f"WHERE deviceId = 1 AND start >= {gs} AND start <= {ge}"))
print(f"\ndev0 gap {(ge-gs)/1e6:.1f}ms: dev1 busy inside = {busy1/1e6:.1f}ms")
