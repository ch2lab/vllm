import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_nvtx.sqlite")
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))

rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(30e9)

# worker phases only
ph = {}
for txtid, st, en, tid in cur.execute(
        "SELECT textId, start, end, globalTid FROM NVTX_EVENTS "
        "WHERE start >= ? AND end IS NOT NULL", (w0,)):
    nm = names.get(txtid, "")
    if nm.startswith("gpu_model_runner"):
        ph.setdefault(tid, []).append((st, en, nm))

for tid, evs in ph.items():
    evs.sort()
    if len(evs) < 100:
        continue
    print(f"\n=== rank tid={tid}: {len(evs)} phase ranges ===")
    # one representative step: take a middle forward and walk +-
    mid = len(evs) // 2
    fwd = next(i for i in range(mid, mid + 50)
               if evs[i][2].endswith("forward"))
    seq = evs[fwd - 4: fwd + 8]
    t0 = seq[0][0]
    prev_end = None
    for st, en, nm in seq:
        gap = (st - prev_end) / 1e6 if prev_end else 0.0
        print(f"  @ {(st-t0)/1e6:8.2f}ms  dur {(en-st)/1e6:7.2f}ms  "
              f"(gap {gap:7.2f}ms)  {nm}")
        prev_end = en
    # aggregate inter-phase gap total across all steps
    tot_gap = 0.0
    for i in range(1, len(evs)):
        g = evs[i][0] - evs[i - 1][1]
        if g > 0:
            tot_gap += g
    span = evs[-1][1] - evs[0][0]
    busy = sum(e - s for s, e, _ in evs)
    print(f"  total span {span/1e9:.1f}s, phase busy {busy/1e9:.1f}s, "
          f"inter-phase gaps {tot_gap/1e9:.1f}s")
