#!/bin/bash
cd /data/src/vllm
python3 test_run.py 2>/dev/null
cat test_run.log
