#!/bin/bash
# Isolate: turbomind only (fused AR off). Log to /tmp/turbomind_only.log
pkill -f test_run.py; sleep 2
cd /data/src/vllm
VLLM_SM70_FUSED_AR_RMSNORM=0 VLLM_SM70_AWQ_BACKEND=turbomind \
    timeout 400 python3 test_run.py > /tmp/turbomind_only.log 2>&1
echo "EXIT=$?" > /tmp/turbomind_only_exit.txt