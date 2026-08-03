import sqlite3

db = sqlite3.connect("/data/src/vllm/nsys_nvtx.sqlite")
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))

rows = sorted(cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"))
t_end = rows[-1][1]
w0 = t_end - int(30e9)

worker_tid = 319916681131624
# all NVTX ranges from OTHER threads
agg = {}
for txtid, st, en, tid in cur.execute(
        "SELECT textId, start, end, globalTid FROM NVTX_EVENTS "
        "WHERE start >= ? AND end IS NOT NULL AND globalTid != ? "
        "AND globalTid != 319916429473369", (w0, worker_tid)):
    nm = names.get(txtid, "")
    a = agg.setdefault((tid, nm), [0, 0])
    a[0] += en - st
    a[1] += 1
print("non-worker NVTX ranges:")
for (tid, nm), (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:15]:
    print(f"  tid={tid}  {t/1e6:9.1f}ms  x{c:4d}  {nm}")

# OSRT blocking of the worker thread in the 54ms gap
evs = sorted(cur.execute(
    "SELECT start, end, textId FROM NVTX_EVENTS "
    "WHERE globalTid = ? AND start >= ? AND end IS NOT NULL "
    "AND textId IN (SELECT id FROM StringIds "
    "WHERE value='gpu_model_runner: set_async_sampled_token_ids' "
    "OR value='gpu_model_runner: preprocess')", (worker_tid, w0)))
# pick middle pair: set_async... then preprocess
mid = len(evs) // 2
st_a, en_a, _ = evs[mid - 1]
st_b, en_b, _ = evs[mid]
if names.get(evs[mid - 1][2], "").endswith("preprocess"):
    st_a, en_a, st_b, en_b = st_b, en_b, st_a, en_a
print(f"\ngap between steps: {(st_b-en_a)/1e6:.1f}ms; worker OSRT inside:")
for st, en, nid in cur.execute(
        "SELECT start, end, nameId FROM OSRT_API "
        "WHERE globalTid = ? AND start >= ? AND start <= ? "
        "AND end-start > 1000000 ORDER BY end-start DESC LIMIT 10",
        (worker_tid, en_a, st_b)):
    print(f"  {(en-st)/1e6:8.1f}ms  {names.get(nid, '?')}")
