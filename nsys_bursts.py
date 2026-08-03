import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_decode2.sqlite")
cur = db.cursor()

rows0 = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows0[-1][1]
w0 = t_end - int(4.6e9)

for dev in (0, 1):
    iv = sorted((s, e) for s, e in cur.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
        f"WHERE deviceId = {dev} AND start >= {w0}"))
    # merge to bursts separated by >5ms
    bursts = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s - ce < 5_000_000:
            ce = max(ce, e)
        else:
            bursts.append((cs, ce))
            cs, ce = s, e
    bursts.append((cs, ce))
    durs = sorted((e - s) / 1e6 for s, e in bursts)
    tot = sum(durs)
    print(f"dev{dev}: {len(bursts)} bursts, total busy {tot:.0f}ms, "
          f"median {durs[len(durs)//2]:.1f}ms, "
          f"p90 {durs[int(len(durs)*0.9)]:.1f}ms, max {durs[-1]:.1f}ms")
    # long bursts distribution
    long_b = [d for d in durs if d > 40]
    print(f"   bursts >40ms: {len(long_b)}, sum {sum(long_b):.0f}ms")
