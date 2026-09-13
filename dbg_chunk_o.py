"""Direct debug of fla_chunk_o_sm70 vs torch reference."""
import struct
import torch
import vllm  # noqa: F401  (loads _C ops)

dev = "cuda"
torch.manual_seed(0)

B, T, Hg, H, K, V, BT = 1, 130, 16, 48, 128, 128, 64
NT = (T + BT - 1) // BT
q = torch.randn(B, T, Hg, K, device=dev, dtype=torch.float16) * 0.3
k = torch.randn(B, T, Hg, K, device=dev, dtype=torch.float16) * 0.3
v = torch.randn(B, T, H, V, device=dev, dtype=torch.float16) * 0.3
h = torch.randn(NT, H, V, K, device=dev, dtype=torch.float16) * 0.3
g = torch.rand(B, T, H, device=dev, dtype=torch.float32) * -2.0
cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
ci = torch.tensor([[0, i] for i in range(NT)], device=dev, dtype=torch.int32).flatten().contiguous()
scale = K ** -0.5


def torch_ref(gate=True):
    qf, kf, vf, hf, gf = q.float(), k.float(), v.float(), h.float(), g.float()
    o = torch.zeros(B, T, H, V, device=dev, dtype=torch.float32)
    per_kv = H // Hg
    for i_h in range(H):
        hg = i_h // per_kv
        for c in range(NT):
            s0, s1 = c * BT, min(T, (c + 1) * BT)
            L = s1 - s0
            qc = qf[0, s0:s1, hg]          # [L,K]
            kc = kf[0, s0:s1, hg]
            vc = vf[0, s0:s1, i_h]
            hc = hf[c, i_h]                 # [V,K]
            gc = gf[0, s0:s1, i_h]          # [L]
            inter = qc @ hc.t()             # [L,V]
            A = qc @ kc.t()                 # [L,L]
            if gate:
                inter = inter * torch.exp(gc)[:, None]
                A = A * torch.exp(gc[:, None] - gc[None, :])
            mask = torch.ones(L, L, device=dev).tril()
            o[0, s0:s1, i_h] = (inter + (A * mask) @ vc) * scale
    return o


def run_kernel():
    o = torch.empty(B, T, H, V, device=dev, dtype=torch.float16)
    sb = struct.unpack("q", struct.pack("<f", scale) + b"\0" * 4)[0]
    torch.ops._C.fla_chunk_o_sm70(
        q.reshape(-1, Hg, K), k.reshape(-1, Hg, K), v.reshape(-1, H, V),
        h, g.half().reshape(-1, H), o.reshape(-1, H, V),
        cu, ci, sb, NT)
    torch.cuda.synchronize()
    return o.float()


ref = torch_ref(True)
got = run_kernel()
d = (got - ref).abs()
print("full: max_abs", d.max().item(), "mean", d.mean().item())

# Component isolation via g=0
g.zero_()
ref0 = torch_ref(True)  # exp(0)=1, same as ungated w/ mask
got0 = run_kernel()
d0 = (got0 - ref0).abs()
print("g=0 : max_abs", d0.max().item(), "mean", d0.mean().item())

# Inter-only check: v=0 -> only q@h^T gated
g.copy_(torch.rand(B, T, H, device=dev) * -2.0)
v_save = v.clone()
v.zero_()
ref_i = torch_ref(True)
got_i = run_kernel()
di = (got_i - ref_i).abs()
print("v=0 (inter only): max_abs", di.max().item(), "mean", di.mean().item())
v.copy_(v_save)

# Intra-only check: h=0
h_save = h.clone()
h.zero_()
ref_a = torch_ref(True)
got_a = run_kernel()
da = (got_a - ref_a).abs()
print("h=0 (intra only): max_abs", da.max().item(), "mean", da.mean().item())
h.copy_(h_save)

# Intra with g=0 (no gating): isolates A accumulation + mma + mask
g.zero_()
h.zero_()
ref_a0 = torch_ref(True)
got_a0 = run_kernel()
da0 = (got_a0 - ref_a0).abs()
print("h=0,g=0 (A@v ungated): max_abs", da0.max().item(), "mean", da0.mean().item())
# position of worst error
idx = (got_a0 - ref_a0).abs().argmax()
b, t, hh, vv = torch.unravel_index(idx, got_a0.shape)
print("worst at t,h,v =", t.item(), hh.item(), vv.item(),
      "got", got_a0[b,t,hh,vv].item(), "ref", ref_a0[b,t,hh,vv].item())

# Probe raw A: v[n] = e_0 at n==r  => o[m,0] = A[m,r] for m>=r (g=0,h=0)
h.zero_(); g.zero_(); v.zero_()
r = 5
v[0, r, :, 0] = 1.0
got_A = run_kernel()[0, :, 0, 0]
qf, kf = q.float(), k.float()
Aref = torch.zeros(T, T, device=dev)
per_kv = H // Hg
for i_h in [0]:
    hg = 0
    for c in range(NT):
        s0, s1 = c*BT, min(T, (c+1)*BT)
        A = qf[0, s0:s1, hg] @ kf[0, s0:s1, hg].t()
        Aref[s0:s1, s0:s1] = A
col = Aref[:, r] * scale
col[:r] = 0
# also zero cross-chunk (kernel is chunk-local)
err = (got_A - col).abs()
print("A-probe: max_abs", err.max().item())
bad = (err > 0.01).nonzero().flatten()
print("bad rows:", bad[:20].tolist(), "of", T)
for t in bad[:5].tolist():
    print(f"  t={t} got={got_A[t].item():.4f} ref={col[t].item():.4f}")

# Full A probe: v = identity -> o[m,d] = A[m,d] (chunk-local, m>=d)
h.zero_(); g.zero_(); v.zero_()
n_id = min(T, V)
for n in range(n_id):
    v[0, n, :, n % V] = 1.0
got_Id = run_kernel()
errmap = torch.zeros(T, V, device=dev)
for i_h in range(H):
    hg = i_h // per_kv
    for c in range(NT):
        s0, s1 = c*BT, min(T, (c+1)*BT)
        A = qf[0, s0:s1, hg] @ kf[0, s0:s1, hg].t() * scale
        L = s1 - s0
        exp_o = torch.zeros(L, V, device=dev)
        for m in range(L):
            for n2 in range(min(L, V)):
                if m >= n2:
                    exp_o[m, n2] += A[m, n2]
        got_blk = got_Id[0, s0:s1, i_h]
        e = (got_blk - exp_o).abs()
        if e.max().item() > 0.02:
            mm = (e > 0.02)
            bad_idx = mm.nonzero()[:3]
            print(f"h={i_h} chunk={c} max_err={e.max().item():.4f} "
                  f"bad first: {bad_idx.tolist()}")
            for bi in bad_idx.tolist():
                m, d = bi
                print(f"   m={m} d={d} got={got_blk[m,d].item():.4f} "
                      f"ref={exp_o[m,d].item():.4f}")
print("identity probe done")
