"""A/B timing: v3h SIMT vs v4 HMMA flashdecode partition+reduce, same calls."""
import time
import torch

HQ, H, D, N, F = 12, 2, 256, 16, 144
R = 1
SM = torch.cuda.get_device_properties(0).multi_processor_count
SPLITS = (SM * 6 + 2 * H - 1) // (2 * H)

from vllm.v1.attention.ops.sm70_flashdecode import sm70_flashdecode_op

OP_H = torch.ops._C.sm70_flashdecode_hmma

dev = "cuda:0"
MAX_CTX = 262144
nb_max = (MAX_CTX + N - 1) // N
kv = torch.randint(0, 256, (nb_max, 2 * H, N, F), dtype=torch.uint8, device=dev)
kv[kv == 255] = 66
ws = torch.zeros((SM * 4 + 8 * H) * 8 * (HQ // H) * (D + 2),
                 dtype=torch.float32, device=dev)
bt_full = torch.arange(nb_max, dtype=torch.int32, device=dev)


def call_v(q, bt, kl, out, q_rows):
    sm70_flashdecode_op(q, kv, bt, kl, out, 1.0, 1.0, 1.0 / D ** 0.5, -1,
                        SPLITS, q_rows, ws)


def call_h(q, bt, kl, out, q_rows):
    OP_H(q, kv, bt, kl, out, 1.0, 1.0, 1.0 / D ** 0.5, -1,
         SPLITS, q_rows, ws)


def time_call(fn, ctx, q_rows):
    nb = (ctx + N - 1) // N
    bt = bt_full[:nb].reshape(R, nb)
    q = torch.randn(R, q_rows, HQ, D, dtype=torch.float16, device=dev)
    out = torch.empty_like(q)
    kl = torch.tensor([ctx], dtype=torch.int32, device=dev)
    for _ in range(3):
        fn(q, bt, kl, out, q_rows)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        fn(q, bt, kl, out, q_rows)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


print(f"SPLITS={SPLITS}  (per-layer call, ms)")
print(f"{'ctx':>8} " + "  ".join(
    f"q={q}: v3h  hmma  x" for q in (1, 4)))
for ctx in (4096, 16384, 32768, 65536, 131072, 215212):
    cells = []
    for q in (1, 4):
        a = time_call(call_v, ctx, q)
        b = time_call(call_h, ctx, q)
        cells.append(f"{a:5.2f} {b:5.2f} {a / b:4.2f}")
    print(f"{ctx:>8} " + "  ".join(cells))
print("SWEEP_DONE")
