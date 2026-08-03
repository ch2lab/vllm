import os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
import time, requests, subprocess, sys, signal

MODEL = "/data/models/Qwen3.5-4B"
PORT = 8199

def run_test(kv_dtype, extra_args=None):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--tensor-parallel-size", "1",
        "--kv-cache-dtype", kv_dtype,
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.92",
        "--port", str(PORT),
        "--language-model-only",
    ]
    if extra_args:
        cmd.extend(extra_args)
    log_file = open(f"/data/src/vllm/bench_{kv_dtype}_log.txt", "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 180
    ready = False
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                ready = True
                break
        except:
            pass
        if proc.poll() is not None:
            break
        time.sleep(2)
    if not ready:
        log_file.flush()
        with open(f"/data/src/vllm/bench_{kv_dtype}_log.txt") as f:
            lines = f.read().strip().split('\n')
        print(f"  FAILED: {lines[-1] if lines else 'unknown'}")
        proc.kill()
        return None, None

    # Prefill test
    prompt = "Explain the theory of relativity in detail, covering special relativity, general relativity, spacetime curvature, gravitational waves, and the experimental evidence that supports each of these concepts. " * 8
    t0 = time.time()
    r = requests.post(f"http://localhost:{PORT}/v1/completions", json={
        "model": MODEL, "prompt": prompt, "max_tokens": 1, "temperature": 0,
    })
    prefill_time = time.time() - t0
    prefill_tokens = r.json()["usage"]["prompt_tokens"] if r.status_code == 200 else 0
    prefill_speed = prefill_tokens / prefill_time if prefill_time > 0 else 0

    # Decode test
    t0 = time.time()
    r = requests.post(f"http://localhost:{PORT}/v1/completions", json={
        "model": MODEL, "prompt": "Write a long essay about AI.", "max_tokens": 256, "temperature": 0,
    })
    decode_time = time.time() - t0
    decode_tokens = r.json()["usage"]["completion_tokens"] if r.status_code == 200 else 0
    decode_speed = decode_tokens / decode_time if decode_time > 0 else 0

    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=10)
    time.sleep(3)
    return prefill_speed, decode_speed

results = {}
for kv in ["auto", "fp8", "fp8_e4m3"]:
    label = {"auto": "FP16", "fp8": "FP8", "fp8_e4m3": "FP8_E4M3"}[kv]
    print(f"=== KV={label} (CUDA Graph) ===")
    p, d = run_test(kv)
    if p is not None:
        print(f"  Prefill: {p:.1f} tok/s, Decode: {d:.1f} tok/s")
        results[label] = (p, d)
    else:
        results[label] = None

print("\n=== RESULTS: 4B TP1 + CUDA Graph ===")
print(f"{'KV-Type':<10} {'Prefill':<14} {'Decode':<14} {'Status'}")
for label, v in results.items():
    if v:
        print(f"{label:<10} {v[0]:<14.1f} {v[1]:<14.1f} OK")
    else:
        print(f"{label:<10} {'FAIL':<14} {'FAIL':<14} FAILED")
