import subprocess
import sys

log = open("/tmp/e2e_phase.log", "w")
p = subprocess.run(
    [sys.executable, "/data/src/vllm/test_27b_phase.py"],
    stdout=log, stderr=subprocess.STDOUT,
)
log.write(f"EXIT={p.returncode}\n")
log.close()
