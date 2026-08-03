import subprocess, sys

with open("e2e_long_rerun.log", "w") as f:
    r = subprocess.run([sys.executable, "test_27b_mtp_long_fp16.py"], stdout=f, stderr=subprocess.STDOUT)
    f.write(f"EXIT={r.returncode}\n")
