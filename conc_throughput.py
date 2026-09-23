import json
import statistics
import sys
import threading
import time
import urllib.request

# Concurrent SUM-throughput bench for load-pattern B. R streams fire
# simultaneously at a realistic ctx; report system throughput =
# total completion_tokens / wall, plus per-stream. Usage-based (no
# per-delta miscount). argv: [ctx_tokens] [R] [n_out]
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
R = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NOUT = int(sys.argv[3]) if len(sys.argv) > 3 else 200
URL = "http://127.0.0.1:8002/v1/chat/completions"
HDR = {"Authorization": "Bearer 0464b22a751c4a5390f0d83c0bbf4fad",
       "Content-Type": "application/json"}

FILLER = ("人工智能与机器学习在当代科学和工程中的广泛应用正在快速改变各行各业"
          "的发展轨迹并且推动着技术革命向前发展。") * 40


def body():
    return json.dumps({
        "model": "Qwen3.8-27B",
        "messages": [{"role": "user",
                      "content": FILLER[:CTX * 12]
                      + "请用一句话总结上文主题。"}],
        "max_tokens": NOUT,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()


res = []
lock = threading.Lock()


def worker():
    usage = {}
    req = urllib.request.Request(URL, data=body(), headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if line.startswith(b"data:"):
                    try:
                        o = json.loads(line[5:])
                    except Exception:
                        continue
                    if o.get("usage"):
                        usage = o["usage"]
    except Exception as e:  # noqa
        usage = {"completion_tokens": 0, "err": str(e)[:60]}
    with lock:
        res.append(usage)


if __name__ == "__main__":
    # warmup
    run_one = threading.Thread(target=worker)
    run_one.start()
    run_one.join()
    res.clear()
    t0 = time.time()
    ths = [threading.Thread(target=worker) for _ in range(R)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.time() - t0
    total = sum(u.get("completion_tokens", 0) for u in res)
    errs = [u.get("err") for u in res if u.get("err")]
    ptoks = statistics.median(
        [u.get("prompt_tokens", 0) for u in res] or [0])
    print(f"CONC R={R} ctx~{int(ptoks)} nout={NOUT}: "
          f"wall={wall:.1f}s total_tok={total} "
          f"SUM={total / wall:.1f} tok/s "
          f"per-stream={total / wall / R:.1f}"
          + (f" ERRS={errs}" if errs else ""), flush=True)
