"""MTP3 decode step anatomy: per-ctx decode_tps, tok/step, ms/step.

ms/step = decode_wall / n_steps, n_steps from spec-decode draft counters.
Fresh seeds per length so prefix cache never pollutes prefill/decode split.
"""
import time, json, urllib.request, random, sys

BASE = "http://127.0.0.1:8002"
KEY = "0464b22a751c4a5390f0d83c0bbf4fad"
MODEL = "Qwen3.8-27B"
GEN = int(sys.argv[1]) if len(sys.argv) > 1 else 256

words = ["quantum","computing","optimization","network","algorithm",
    "distributed","vector","matrix","tensor","neural","gradient","sampling",
    "inference","memory","bandwidth","pipeline","scheduler","kernel","attention",
    "embedding","cache","latency","throughput","precision","float",
    "activation","quantization","speculative","draft","expert","routing"]

def body_for_tok(target, seed):
    # Unique per-invocation salt prefix defeats vLLM prefix caching
    # (mirrors llm_speedtest's cache-busting; same-seed repeat runs
    # would otherwise measure a cache-hit prefill).
    import os as _os, time as _t
    salt = _os.environ.get("BENCH_SALT")
    seed = seed + (int(salt) if salt and salt.isdigit()
                   else int(_t.time_ns()) % 1_000_003 + _os.getpid())
    r = random.Random(seed); toks = []
    while len(toks) < target * 4:
        toks.extend(r.choices(words, k=30))
    return " ".join(toks)

def get(key_filter):
    txt = urllib.request.urlopen(urllib.request.Request(
        BASE + "/metrics",
        headers={"Authorization": "Bearer " + KEY}), timeout=30).read().decode()
    d = {}
    for line in txt.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        head, _, val = line.rpartition(" ")
        if key_filter in head:
            base = head.split("{")[0]
            try:
                d[base] = d.get(base, 0.0) + float(val)
            except ValueError:
                pass
    return d

def stream_chat(body):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": body}],
               "max_tokens": GEN, "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    times = []; ntok = 0; usage = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                o = json.loads(data)
            except Exception:
                continue
            if o.get("usage"):
                usage = o["usage"]
            ch = o.get("choices") or []
            if not ch:
                continue
            delta = ch[0].get("delta", {})
            txt = delta.get("content") or delta.get("reasoning") or ""
            if txt:
                ntok += 1; times.append(time.time())
    return ntok, times, usage

stream_chat("hi")
print("warmup done", flush=True)
print(f"{'ctx':>8} {'gen':>5} {'pref_s':>8} {'tps':>7} {'tok/step':>8} {'acc':>6} {'ms/step':>8}", flush=True)  # tps=true tokens/s (usage-based)
# body_for_tok(C) builds C*4 tokens; re-tokenization inflates ~3.2%
# (32768*4 -> measured 135316 ctx). Top target 63000 -> ~260.2K real
# prompt, fits max_model_len=262144 with GEN headroom. Old 65536/131072
# targets (262144/524288) were always over-context dead items.
for i, C in enumerate([8192, 16384, 32768, 63000]):
    body = body_for_tok(C, seed=7000 + i * 131)
    m0 = get("spec_decode")
    t0 = time.time()
    try:
        ntok, times, usage = stream_chat(body)
    except urllib.error.HTTPError as e:
        # top target intentionally exceeds max_model_len (server-side 400);
        # stop cleanly instead of dumping a traceback.
        print(f"BENCH_STOP_EARLY ctx>={C}: HTTP {e.code}", flush=True)
        break
    m1 = get("spec_decode")
    ctx = usage.get("prompt_tokens", C)
    # SSE deltas batch multiple accepted tokens per step under spec decode;
    # count from usage.completion_tokens so tok/s is true token throughput.
    ntok_real = usage.get("completion_tokens") or ntok
    drafts = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    dtok = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get("vllm:spec_decode_num_draft_tokens_total", 0)
    atok = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    # steps ~= drafts (1 draft per step) ; tokens out = 1 + accepted per step
    steps = max(drafts, 1)
    tps = (ntok_real - 1) / (times[-1] - times[0]) if len(times) >= 2 else 0
    tok_step = ntok_real / steps
    ms_step = (times[-1] - times[0]) * 1e3 / steps if len(times) >= 2 else 0
    acc = atok / dtok if dtok else 0
    print(f"{ctx:>8} {ntok:>5} {times[0]-t0:>8.1f} {tps:>7.2f} {tok_step:>8.2f} {acc:>6.3f} {ms_step:>8.1f}", flush=True)
print("BENCH_DONE", flush=True)
