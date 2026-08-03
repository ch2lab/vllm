import subprocess
import sys

log = open("/tmp/acceptance_probe.log", "w")
p = subprocess.run(
    [sys.executable, "/data/src/vllm/test_27b_acceptance.py"],
    stdout=log, stderr=subprocess.STDOUT,
)
log.write(f"EXIT={p.returncode}\n")
log.close()
