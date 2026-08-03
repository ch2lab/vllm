#!/usr/bin/env python3
"""Full SM70 benchmark suite — 60 test combinations."""
import os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
import time, requests, subprocess, sys, signal, json

PORT = 8199
BASE_URL = f"http://localhost:{PORT}"
RESULTS_FILE = "/data/src/vllm/bench_results.json"

PREFILL_PROMPT = (
    "Explain the theory of relativity in detail, covering special relativity, "
    "general relativity, spacetime curvature, gravitational waves, and the "
    "experimental evidence that supports each of these concepts. "
) * 8

MODELS = {
    "4B": "/data/models/Qwen3.5-4B",
    "4B-AWQ": "/data/models/Qwen3.5-4B-AWQ",
    "4B-FP8": "/data/models/Qwen3.5-4B-FP8",
    "27B-AWQ": "/data/models/Qwen3.6-27B-AWQ",
    "27B-AWQ-MTP": "/data/models/Qwen3.6-27B-AWQ-MTP",
    "35B-A3B-AWQ": "/data/models/Qwen3.6-35B-A3B-AWQ",
    "27B-FP8": "/data/models/Qwen3.6-27B-FP8",
    "Ornith-35B-FP8": "/data/models/Ornith-1.0-35B-FP8",
    "27B-NVFP4-MTP": "/data/models/Qwen3.6-27B-Text-NVFP4-MTP",
}

KV_TYPES = {
    "FP16": "auto",
    "FP8": "fp8",
    "INT4": "int4_per_token_head",
    "NVFP4": "nvfp4",
}

TESTS = []
# Group 1: 4B models TP1
for model_key in ["4B", "4B-AWQ", "4B-FP8"]:
    for kv_label, kv_dtype in KV_TYPES.items():
        TESTS.append({
            "id": f"T{len(TESTS)+1:02d}",
            "model_key": model_key,
            "kv_label": kv_label,
            "kv_dtype": kv_dtype,
            "tp": 1, "pp": 1, "mtp": False,
        })

# Group 2: 27B-AWQ TP2 / MTP4 / PP2
for kv_label, kv_dtype in KV_TYPES.items():
    TESTS.append({"id": f"T{len(TESTS)+1:02d}", "model_key": "27B-AWQ",
                  "kv_label": kv_label, "kv_dtype": kv_dtype,
                  "tp": 2, "pp": 1, "mtp": False})
for kv_label, kv_dtype in KV_TYPES.items():
    TESTS.append({"id": f"T{len(TESTS)+1:02d}", "model_key": "27B-AWQ-MTP",
                  "kv_label": kv_label, "kv_dtype": kv_dtype,
                  "tp": 2, "pp": 1, "mtp": True})
for kv_label, kv_dtype in KV_TYPES.items():
    TESTS.append({"id": f"T{len(TESTS)+1:02d}", "model_key": "27B-AWQ",
                  "kv_label": kv_label, "kv_dtype": kv_dtype,
                  "tp": 1, "pp": 2, "mtp": False})

# Group 3: 35B-A3B-AWQ MoE
for kv_label, kv_dtype in KV_TYPES.items():
    TESTS.append({"id": f"T{len(TESTS)+1:02d}", "model_key": "35B-A3B-AWQ",
                  "kv_label": kv_label, "kv_dtype": kv_dtype,
                  "tp": 2, "pp": 1, "mtp": False})
for kv_label, kv_dtype in KV_TYPES.items():
    TESTS.append({"id": f"T{len(TESTS)+1:02d}", "model_key": "35B-A3B-AWQ",
                  "kv_label": kv_label, "kv_dtype": kv_dtype,
                  "tp": 2, "pp": 1, "mtp": True})
for kv_label, kv_dtype in KV_TYPES.items():
    TESTS.append({"id": f"T{len(TESTS)+1:02d}", "model_key": "35B-A3B-AWQ",
                  "kv_label": kv_label, "kv_dtype": kv_dtype,
                  "tp": 1, "pp": 2, "mtp": False})

# Group 4: FP8 weight models
for model_key in ["27B-FP8", "Ornith-35B-FP8"]:
    for kv_label, kv_dtype in KV_TYPES.items():
        TESTS.append({"id": f"T{len(TESTS)+1:02d}", "model_key": model_key,
                      "kv_label": kv_label, "kv_dtype": kv_dtype,
                      "tp": 2, "pp": 1, "mtp": False})
    for kv_label, kv_dtype in KV_TYPES.items():
        TESTS.append({"id": f"T{len(TESTS)+1:02d}", "model_key": model_key,
                      "kv_label": kv_label, "kv_dtype": kv_dtype,
                      "tp": 1, "pp": 2, "mtp": False})

# Group 5: NVFP4 weight model
for kv_label, kv_dtype in KV_TYPES.items():
    TESTS.append({"id": f"T{len(TESTS)+1:02d}", "model_key": "27B-NVFP4-MTP",
                  "kv_label": kv_label, "kv_dtype": kv_dtype,
                  "tp": 2, "pp": 1, "mtp": False})
for kv_label, kv_dtype in KV_TYPES.items():
    TESTS.append({"id": f"T{len(TESTS)+1:02d}", "model_key": "27B-NVFP4-MTP",
                  "kv_label": kv_label, "kv_dtype": kv_dtype,
                  "tp": 2, "pp": 1, "mtp": True})


def run_test(test):
    model_path = MODELS[test["model_key"]]
    kv_dtype = test["kv_dtype"]
    parallel = test["parallel"] = (
        f"TP{test['tp']}" if test["tp"] > 1 else
        f"PP{test['pp']}" if test["pp"] > 1 else "TP1"
    )
    mtp_str = "+MTP4" if test["mtp"] else ""
    label = f"{test['id']} {test['model_key']} KV={test['kv_label']} {parallel}{mtp_str}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--kv-cache-dtype", kv_dtype,
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.92",
        "--port", str(PORT),
        "--language-model-only",
    ]
    if test["tp"] > 1:
        cmd += ["--tensor-parallel-size", str(test["tp"])]
    if test["pp"] > 1:
        cmd += ["--pipeline-parallel-size", str(test["pp"])]
    if test["mtp"]:
        cmd += ["--spec-method", "mtp", "--spec-tokens", "4"]

    log_path = f"/data/src/vllm/bench_log_{test['id']}.txt"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    deadline = time.time() + 300
    ready = False
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        if proc.poll() is not None:
            break
        time.sleep(2)

    if not ready:
        log_file.flush()
        with open(log_path) as f:
            lines = f.read().strip().split('\n')
        err = ""
        for l in reversed(lines):
            if "Error" in l or "error" in l or "OOM" in l:
                err = l.strip()[:120]
                break
        if not err and lines:
            err = lines[-1][:120]
        print(f"  FAILED: {err}")
        proc.kill()
        return {"status": "FAIL", "error": err}

    # Warmup
    try:
        requests.post(f"{BASE_URL}/v1/completions", json={
            "model": model_path, "prompt": "Hello world", "max_tokens": 16, "temperature": 0,
        }, timeout=120)
    except Exception:
        pass
    time.sleep(1)

    # Prefill
    try:
        t0 = time.time()
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": model_path, "prompt": PREFILL_PROMPT,
            "max_tokens": 1, "temperature": 0,
        }, timeout=300)
        prefill_time = time.time() - t0
        if r.status_code == 200:
            prefill_tokens = r.json()["usage"]["prompt_tokens"]
            prefill_speed = prefill_tokens / prefill_time
        else:
            prefill_speed = 0
    except Exception:
        prefill_speed = 0

    # Decode
    try:
        t0 = time.time()
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": model_path, "prompt": "Write a detailed essay.",
            "max_tokens": 256, "temperature": 0,
        }, timeout=300)
        decode_time = time.time() - t0
        if r.status_code == 200:
            decode_tokens = r.json()["usage"]["completion_tokens"]
            decode_speed = decode_tokens / decode_time
        else:
            decode_speed = 0
    except Exception:
        decode_speed = 0

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)

    result = {
        "status": "OK",
        "prefill": round(prefill_speed, 1),
        "decode": round(decode_speed, 1),
    }
    print(f"  Prefill: {prefill_speed:.1f} tok/s, Decode: {decode_speed:.1f} tok/s")
    return result


def main():
    # Load existing results to allow resume
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    start_idx = 0
    if len(sys.argv) > 1:
        start_idx = int(sys.argv[1])

    for i, test in enumerate(TESTS):
        if i < start_idx:
            continue
        tid = test["id"]
        if tid in results and results[tid].get("status") == "OK":
            print(f"  {tid} already done, skipping")
            continue
        result = run_test(test)
        results[tid] = {
            "model": test["model_key"],
            "kv": test["kv_label"],
            "parallel": test.get("parallel", ""),
            "mtp": test["mtp"],
            **result,
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'ID':<5} {'Model':<18} {'KV':<7} {'Par':<5} {'MTP':<4} {'Prefill':<10} {'Decode':<10} {'Status'}")
    print(f"{'-'*70}")
    for test in TESTS:
        tid = test["id"]
        r = results.get(tid, {})
        if r.get("status") == "OK":
            print(f"{tid:<5} {r['model']:<18} {r['kv']:<7} {r['parallel']:<5} "
                  f"{'Y' if r['mtp'] else '-':<4} {r['prefill']:<10.1f} {r['decode']:<10.1f} OK")
        elif r.get("status") == "FAIL":
            print(f"{tid:<5} {test['model_key']:<18} {test['kv_label']:<7} "
                  f"{'':<5} {'':<4} {'':<10} {'':<10} FAIL: {r.get('error','')[:40]}")
        else:
            print(f"{tid:<5} {test['model_key']:<18} {test['kv_label']:<7} "
                  f"{'':<5} {'':<4} {'':<10} {'':<10} PENDING")


if __name__ == "__main__":
    main()
