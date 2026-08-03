"""Microbenchmark: CUDA graph node dispatch overhead on V100.
Hypothesis: graph execution time ~ per-node overhead * node_count."""
import time
import torch

def bench_graph(num_nodes, num_iters=30, do_eager=False):
    x = torch.zeros(16, device='cuda')
    def f():
        for _ in range(num_nodes):
            x.add_(1.0)

    # Eager baseline
    if do_eager:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            f()
        torch.cuda.synchronize()
        eager = (time.perf_counter() - t0) / num_iters * 1000
    else:
        eager = None

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    torch.cuda.synchronize()

    # warmup
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        g.replay()
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - t0) / num_iters * 1000
    return graph_ms, eager, x.sum().item()

print(f"device: {torch.cuda.get_device_name(0)}")
for n in (10, 27, 50, 100, 200):
    gm, em, _ = bench_graph(n)
    per_node = gm / n
    print(f"nodes={n:4d}  graph={gm:8.3f} ms  per_node={per_node:6.3f} ms")
for n in (27, 100):
    gm, em, _ = bench_graph(n, do_eager=True)
    print(f"nodes={n:4d}  eager={em:8.3f} ms  graph={gm:8.3f} ms")
