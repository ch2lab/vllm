"""SM70 kernel quick validation script.
Usage: python3 test_sm70_kernels.py
"""
import sys
sys.path.insert(0, '/data/src/vllm')

import torch
from vllm import _custom_ops as ops

def test_awq_gemm():
    torch.manual_seed(42)
    M, K, N = 5, 4096, 4096
    x = torch.randn(M, K, dtype=torch.float16, device='cuda')
    qw = torch.randint(0, 2**31, (K, N//8), dtype=torch.int32, device='cuda')
    sc = torch.randn(K//128, N, dtype=torch.float16, device='cuda').abs() + 0.1
    qz = torch.randint(0, 2**31, (K//128, N//8), dtype=torch.int32, device='cuda')
    out = ops.awq_gemm_sm70(x, qw, sc, qz, 128, 0)
    ref = torch.matmul(x, ops.awq_dequantize(qw, sc, qz, 0, 0, 0))
    err = (out.float() - ref.float()).abs().max().item()
    ok = err < 1.0
    print(f'AWQ GEMM:    max_err={err:.4f} {"PASS" if ok else "FAIL"}')
    return ok

def test_attention():
    torch.manual_seed(42)
    B, H, Sq, Skv, D = 1, 4, 16, 32, 128
    Q = torch.randn(B, H, Sq, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, Skv, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, Skv, D, dtype=torch.float16, device='cuda')
    scale = 1.0 / (D ** 0.5)
    for causal in [False, True]:
        out = ops.flash_attn_sm70_prefill(Q, K, V, scale, causal)
        with torch.no_grad():
            ref = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, is_causal=causal)
        err = (out.float() - ref.float()).abs().max().item()
        ok = err < 0.01
        tag = "causal" if causal else "non-causal"
        print(f'Attention ({tag}): max_err={err:.4f} {"PASS" if ok else "FAIL"}')
        if not ok:
            return False
    return True

if __name__ == '__main__':
    print('=== SM70 Kernel Validation ===')
    r1 = test_awq_gemm()
    r2 = test_attention()
    print(f'\nResult: {"ALL PASS" if r1 and r2 else "SOME FAILED"}')
    sys.exit(0 if (r1 and r2) else 1)
