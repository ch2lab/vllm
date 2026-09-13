# SPDX-License-Identifier: Apache-2.0
# Env-gated CUDA-event timing for SM70 profiling (VLLM_SM70_PROFILE=1).
"""Per-op CUDA timing helper used to profile the GDN prefill chain."""

import os as _os

import torch

ENABLED = _os.environ.get("VLLM_SM70_PROFILE", "0") == "1"
_STATE = {}


def profiled(tag):
    """Decorator timing fn's launches on the current CUDA stream."""

    if not ENABLED:
        return lambda fn: fn

    _seen = {"n": 0}

    def deco(fn):

        def wrap(*args, **kwargs):
            if _seen["n"] < 3:
                k = kwargs.get("k", args[1] if len(args) > 1 else None)
                cu = kwargs.get("cu_seqlens", args[9] if len(args) > 9 else None)
                if k is not None:
                    B, T, Hg, K = k.shape
                    print(
                        f"[gdn-profile] {tag} call#{_seen['n']}: "
                        f"T={T} B={B} Hg={Hg} K={K} cu_seqlens={cu}",
                        flush=True,
                    )
                _seen["n"] += 1
            # CUDA events with timing are not capturable; skip timing while
            # the current stream is being captured so profiling never breaks
            # CUDA graph capture.
            if torch.cuda.is_current_stream_capturing():
                return fn(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            acc = _STATE.setdefault(tag, [])
            acc.append((start, end))
            if len(acc) >= 50:
                torch.cuda.synchronize()
                total = sum(s.elapsed_time(e) for s, e in acc)
                cur = acc.copy()
                acc.clear()
                print(
                    f"[gdn-profile] {tag}: {len(cur)} calls, "
                    f"{total:.1f} ms (avg {total / len(cur):.3f} ms)",
                    flush=True,
                )
            return out

        return wrap

    return deco