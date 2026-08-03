import subprocess, sys

for script, log in [
    ("test_27b_mtp_long_fp16.py", "e2e_long_fp16.log"),
    ("test_27b_mtp_fp8.py", "e2e_fp8.log"),
]:
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, script], stdout=f, stderr=subprocess.STDOUT)
        f.write(f"EXIT={r.returncode}\n")
