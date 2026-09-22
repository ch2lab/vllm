"""Flashdecode scaling sweep: time vs (q_rows, ctx) to bound verify-path cost."""
import torch

HQ, H, D, N, F = 12, 2, 256, 16, 144
R = 1
SM = torch.cuda.get_device_properties(0).multi_processor_count
SPLITS = (SM * 6 + 2 * H - 1) // (2 * H)

from vllm.v1.attention.ops.sm70_flashdecode import sm70_flashdecode_op

dev = "cuda:0"
MAX_CTX = 262144
nb_max = (MAX_CTX + N - 1) // N
kv = torch.randint(0, 256, (nb_max, 2 * H, N, F), dtype=torch.uint8, device=dev)
kv[kv == 255] = 66
ws = torch.empty((SM * 4 + 8 * H) * 8 * (HQ // H) * (D + 2), dtype=torch.float32, device=dev)

bt_full = torch.arange(nb_max, dtype=torch.int32, device=dev)

def time_call(ctx, q_rows):
    nb = (ctx + N - 1) // N
    bt = bt_full[:nb].reshape(R, nb)
    q = torch.randn(R, q_rows, HQ, D, dtype=torch.float16, device=dev)
    out = torch.empty_like(q)
    kl = torch.tensor([ctx], dtype=torch.int32, device=dev)
    for _ in range(3):
        sm70_flashdecode_op(q, kv, bt, kl, out, 1.0, 1.0, 1.0 / D ** 0.5, -1,
                            SPLITS, q_rows, ws)
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        sm70_flashdecode_op(q, kv, bt, kl, out, 1.0, 1.0, 1.0 / D ** 0.5, -1,
                            SPLITS, q_rows, ws)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

print(f"SPLITS={SPLITS}  (per-layer call, ms; 16 attn layers/step)")
print(f"{'ctx':>8} " + " ".join(f"q={q:>2}" for q in (1, 2, 3, 4)))
for ctx in (16384, 32768, 65536, 131072, 215212):
    row = []
    for q in (1, 2, 3, 4):
        row.append(time_call(ctx, q))
    print(f"{ctx:>8} " + " ".join(f"{t:5.2f}" for t in row))
print("SWEEP_DONE")
