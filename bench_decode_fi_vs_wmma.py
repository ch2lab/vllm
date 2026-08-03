"""Decode-kernel perf comparison on SM70: FlashInfer FA2 paged decode vs
our WMMA batched paged kernel (fp16 KV, Sq=1).

Run with:
  FLASHINFER_ALLOW_SM70=1 FLASHINFER_CUDA_ARCH_LIST="7.0" \
  PYTHONPATH=/data/src/flashinfer python3 bench_decode_fi_vs_wmma.py
"""
import time

import torch
import torch.nn.functional as F

import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
import flashinfer

print("flashinfer:", flashinfer.__file__)

dev = "cuda"
# Qwen3.6-27B per-GPU under TP2: 12 Q heads, 2 KV heads, head_dim 256.
# NOTE: FlashInfer decode rejects GQA group_size 6 (only 1/2/4/8), so this
# comparison is reference-only for the target model (use group 4 shapes).
H, KVH, HD = 16, 4, 128
BS = 16  # vLLM block size == flashinfer page size
scale = HD**-0.5

def make_cache(seq_len, batch):
    nb = batch * ((seq_len + BS - 1) // BS)
    cache_fi = torch.randn(nb, 2, BS, KVH, HD, device=dev, dtype=torch.float16) * 0.1
    # WMMA layout: (NB, KVH, BS, 2*HD) — K then V in content dim
    cache_wm = torch.empty(nb, KVH, BS, 2 * HD, device=dev, dtype=torch.float16)
    cache_wm[:, :, :, :HD] = cache_fi[:, 0].permute(0, 2, 1, 3)
    cache_wm[:, :, :, HD:] = cache_fi[:, 1].permute(0, 2, 1, 3)
    pages = torch.arange(nb, dtype=torch.int32, device=dev)
    return cache_fi, cache_wm, pages

def bench(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us

workspaces = {}
for seq_len in (1024, 4096, 8192):
    for batch in (1, 8, 32):
        cache_fi, cache_wm, pages = make_cache(seq_len, batch)
        q = torch.randn(batch, H, HD, device=dev, dtype=torch.float16) * 0.1

        # --- FlashInfer FA2 decode ---
        npp = (seq_len + BS - 1) // BS  # pages per seq
        indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=dev) * npp
        bt_fi = pages.view(batch, npp).contiguous()
        last = torch.full((batch,), seq_len - (seq_len // BS) * BS or BS,
                          dtype=torch.int32, device=dev)
        ws = workspaces.setdefault(
            batch, torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev))
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="NHD")

        def fi_run():
            wrapper.plan(indptr, bt_fi.view(-1), last, H, KVH, HD, BS)
            return wrapper.run(q, cache_fi)

        # correctness once
        o_fi = fi_run()
        torch.cuda.synchronize()

        # --- WMMA batched kernel (Sq=1) ---
        qsl = torch.arange(0, batch + 1, dtype=torch.int32, device=dev)
        sl = torch.full((batch,), seq_len, dtype=torch.int32, device=dev)
        bt_wm = bt_fi.view(batch, npp)
        bt_pad = torch.zeros(batch, npp, dtype=torch.int32, device=dev)
        bt_pad[:, :npp] = bt_wm

        def wm_run():
            return torch.ops._C.flash_attn_sm70_prefill_paged_batched(
                q.view(batch, H, HD), cache_wm, bt_pad, qsl, sl,
                batch, 1, scale, True, 1.0, 1.0, 0,
            )

        o_wm = wm_run()
        torch.cuda.synchronize()

        diff = (o_fi.float() - o_wm.float()).abs().max().item()

        t_fi = bench(fi_run)
        t_wm = bench(wm_run)
        print(f"seq={seq_len:5d} b={batch:3d} | FI={t_fi:8.1f}us "
              f"WMMA={t_wm:8.1f}us  speedup_wmma={t_fi / t_wm:.2f}x  "
              f"maxdiff={diff:.4f}")
