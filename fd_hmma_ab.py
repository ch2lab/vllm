#!/usr/bin/env python3
"""A/B: sm70_flashdecode_hmma (cute v4) vs v3h vs numpy reference."""
import sys
import torch

sys.path.insert(0, "/data/src/vllm")
from test_flashdecode_unit import make_case, D, HQ  # noqa: E402
from vllm.v1.attention.ops.sm70_flashdecode import (  # noqa: E402
    sm70_nvfp4_reference_attention,
)

OP_V3H = torch.ops._C.sm70_flashdecode
OP_HMMA = torch.ops._C.sm70_flashdecode_hmma


def run(op, q, kv, bt, kv_lens, wl, splits, q_rows, ws_scale=1.0):
    R, Q, _, _ = q.shape
    H = kv.shape[1] // 2
    ws = torch.zeros(
        R * H * splits * q_rows * (HQ // H) * (D + 2)
        + 4 * R * H * q_rows * (HQ // H) * (D + 2),
        dtype=torch.float32, device="cuda")
    out = torch.empty_like(q, device="cuda")
    op(q.cuda(), kv.cuda(), bt.cuda(), kv_lens.cuda(), out,
       ws_scale, ws_scale, 1.0 / 16, int(wl), int(splits), int(q_rows), ws)
    torch.cuda.synchronize()
    return out


def case(name, seed, kv_lens, q_rows, page=16, wl=-1, splits=4,
         ws_scale=1.0):
    q, kv, bt, kl = make_case(R=1, Q=q_rows, seed=seed, kv_lens=kv_lens,
                              page=page)
    ref = sm70_nvfp4_reference_attention(
        q, kv, bt, kl, ws_scale, ws_scale, 1.0 / 16, wl)
    o3 = run(OP_V3H, q, kv, bt, kl, wl, splits, q_rows, ws_scale)
    o4 = run(OP_HMMA, q, kv, bt, kl, wl, splits, q_rows, ws_scale)
    e3 = (o3.cpu().float() - ref).abs().max().item()
    e4 = (o4.cpu().float() - ref).abs().max().item()
    e34 = (o3 - o4).abs().max().item()
    ok = e4 < 2e-2 and e34 < 2e-2
    print(f"{'PASS' if ok else 'FAIL'} {name:28s} "
          f"err_ref_v3h={e3:.5f} err_ref_hmma={e4:.5f} err_v3h_hmma={e34:.5f}")
    return ok


def main():
    torch.cuda.init()
    results = [
        case("q1 basic 275/33", 5, (275,), 1),
        case("q1 chunked page32", 8, (100,), 1, page=32, splits=3),
        case("q4 verify", 6, (123,), 4),
        case("q8 verify", 11, (200,), 8, splits=5),
        case("window wl=32", 7, (60,), 1, wl=32),
        case("long 1000 splits16", 12, (1000,), 1, splits=16),
        case("empty splits kv=17", 13, (17,), 1, splits=8),
        case("scales 2.0/0.5", 14, (300,), 1, ws_scale=0.5),
        case("page32 q4 900 splits32", 15, (900,), 4, page=32, splits=32),
    ]
    print(f"== {sum(results)}/{len(results)} PASS")
    sys.exit(0 if all(results) else 1)


main()
