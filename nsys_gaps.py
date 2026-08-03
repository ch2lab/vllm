import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_decode.sqlite")
cur = db.cursor()
rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(4.6e9)
iv = [(s, e) for s, e in rows if s >= w0]
gaps = []
for i in range(1, len(iv)):
    g = iv[i][0] - iv[i - 1][1]
    if g > 2_000_000:
        gaps.append((g / 1e6, (iv[i - 1][1] - w0) / 1e9))
print("gaps >2ms (gap_ms @ offset_s):")
for g, t in gaps:
    print(f"  {g:8.1f}  @{t:6.2f}")
print(f"count={len(gaps)} sum={sum(g for g, _ in gaps):.0f}ms")
