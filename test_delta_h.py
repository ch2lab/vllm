"""Test fla_delta_h_sm70 in isolation."""
import torch, os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
import vllm  # load _C module

torch.manual_seed(42)
dev = "cuda"
T, H, Hg, K, V = 64, 12, 2, 256, 256

k = torch.randn(1, T, Hg, K, dtype=torch.float16, device=dev) * 0.1
w = torch.randn(1, T, H, K, dtype=torch.float16, device=dev) * 0.1
u = torch.randn(1, T, H, V, dtype=torch.float16, device=dev) * 0.1
g = torch.randn(1, T, H, dtype=torch.float16, device=dev) * 0.01
cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=dev)
chunk_offsets = torch.tensor([0], dtype=torch.int32, device=dev)
h0 = torch.empty(0, dtype=torch.float32, device=dev)

print("Calling fla_delta_h_sm70...")
try:
    h, v_new, final_state = torch.ops._C.fla_delta_h_sm70(
        k, w, u, g, cu_seqlens, chunk_offsets, h0, True, 1)
    torch.cuda.synchronize()
    print("SUCCESS")
    print("h shape:", h.shape, "dtype:", h.dtype)
    print("v_new shape:", v_new.shape)
    print("final_state shape:", final_state.shape)
    print("h finite:", torch.isfinite(h).all().item())
    print("v_new finite:", torch.isfinite(v_new).all().item())
    print("h[0,0,0,:5,:5]:", h[0,0,0,:2,:2])
except Exception as e:
    print("FAILED:", e)
    import traceback; traceback.print_exc()