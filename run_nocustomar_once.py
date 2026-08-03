import subprocess, sys

with open("e2e_nocustomar.log", "w") as f:
    r = subprocess.run([sys.executable, "test_27b_mtp_fp8_nocustomar.py"], stdout=f, stderr=subprocess.STDOUT)
    f.write(f"EXIT={r.returncode}\n")
