"""Equivalence test: fused_sigmoid_gating_delta_rule_update (MTP verify GDN
path) vs fused_recurrent_gated_delta_rule reference. Qwen3.8 dims, SM70."""

import torch
import torch.nn.functional as F

from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

DEVICE = "cuda:0"


def run_spec(num_reqs, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
             num_spec_tokens, dtype):
    torch.set_default_device(DEVICE)
    torch.manual_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = key_dim * 2 + value_dim
    num_tokens = num_reqs * (num_spec_tokens + 1)
    total_entries = num_tokens + 4

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads, dtype=dtype)
    dt_bias = torch.rand(num_v_heads, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    ssm_state = torch.rand(total_entries, num_v_heads, head_k_dim, head_v_dim,
                           dtype=dtype)
    # indices must be >=1 (0 is the NULL state sentinel) and < total_entries
    state_indices = torch.arange(1, num_tokens + 1, dtype=torch.int32).view(
        num_reqs, num_spec_tokens + 1)
    num_accepted_tokens = torch.randint(
        1, num_spec_tokens + 1, (num_reqs,), dtype=torch.int32)
    cu_seqlens = torch.arange(0, num_tokens + 1, num_spec_tokens + 1,
                              dtype=torch.int32)

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    out_ref, state_ref = fused_recurrent_gated_delta_rule(
        q=query, k=key, v=value, g=g.unsqueeze(0), beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(), inplace_final_state=True,
        ssm_state_indices=state_indices, cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )
    out, state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=query, k=key, v=value,
        initial_state=ssm_state, inplace_final_state=True,
        ssm_state_indices=state_indices, cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )
    d_out = (out.float() - out_ref.float()).abs().max().item()
    d_st = (state.float() - state_ref.float()).abs().max().item()
    ok = d_out < 5e-2 and d_st < 5e-2
    print(f"reqs={num_reqs} spec={num_spec_tokens} dtype={dtype} "
          f"d_out={d_out:.4f} d_state={d_st:.4f} {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    all_ok = True
    for dtype in (torch.float32, torch.float16):
        for nreq in (1, 2, 4):
            for spec in (1, 2, 3):
                # Qwen3.8 GDN: Hk=16 Hv=48 K=V=128 (per-GPU TP1 dims)
                all_ok &= run_spec(nreq, 16, 48, 128, 128, spec, dtype)
    print("ALL PASS" if all_ok else "SOME FAIL")
