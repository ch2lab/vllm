import struct
import torch
import vllm  # noqa

dev = "cuda"
torch.manual_seed(0)
Hg, H, K, V, BT = 16, 48, 128, 128, 64
seqlens = [100, 70]
T = sum(seqlens)
NT = sum((s + BT - 1) // BT for s in seqlens)
B = len(seqlens)
q = torch.randn(B, T, Hg, K, device=dev, dtype=torch.float16) * 0.3
k = torch.randn(B, T, Hg, K, device=dev, dtype=torch.float16) * 0.3
v = torch.randn(B, T, H, V, device=dev, dtype=torch.float16) * 0.3
h = torch.randn(NT, H, V, K, device=dev, dtype=torch.float16) * 0.3
scale = K ** -0.5
cu = torch.tensor([0, 100, 170], device=dev, dtype=torch.int32)
ci = torch.tensor([[0,0],[0,1],[1,0],[1,1]], device=dev, dtype=torch.int32).flatten().contiguous()

def torch_ref(b, s0, s1, i_h, hg, c):
    qf, kf, vf, hf = q.float(), k.float(), v.float(), h.float()
    L = s1 - s0
    qc = qf[b, s0:s1, hg]; kc = kf[b, s0:s1, hg]
    vc = vf[b, s0:s1, i_h]; hc = hf[c, i_h]
    o = qc @ hc.t() + torch.tril(qc @ kc.t()) @ vc
    return o * scale

o = torch.empty(B, T, H, V, device=dev, dtype=torch.float16)
sb = struct.unpack("q", struct.pack("<f", scale) + b"\0" * 4)[0]
# NOTE: kernel expects packed [T_total, ...] with bos from cu_seqlens
torch.ops._C.fla_chunk_o_sm70(
    q.reshape(-1, Hg, K), k.reshape(-1, Hg, K), v.reshape(-1, H, V),
    h, torch.zeros(T, H, device=dev, dtype=torch.float16), o.reshape(-1, H, V),
    cu, ci, sb, NT)
torch.cuda.synchronize()

pos = 0
for b, s in enumerate(seqlens):
    for c in range((s + BT - 1) // BT):
        s0 = b * T + c * BT  # wrong: packed layout is flat across seqs
# packed: seq b occupies [bos, bos+s) in flat dim
bos_all = [0, 100]
chunk_no = 0
for b, s in enumerate(seqlens):
    bos = bos_all[b]
    for c in range((s + BT - 1) // BT):
        s0, s1 = bos + c * BT, min(bos + s, bos + (c + 1) * BT)
        for i_h in [0, 47]:
            hg = i_h // (H // Hg)
            ref = torch_ref(b, s0, s1, i_h, hg, chunk_no)
            got = o.reshape(-1, H, V)[s0:s1, i_h].float()
            e = (got - ref).abs().max().item()
            print(f"b={b} chunk={c} h={i_h}: max_err={e:.5f}")
        chunk_no += 1
