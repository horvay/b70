#!/usr/bin/env python3
"""Ripple-65B reference forward in numpy (fp32), from the checkpoint's modeling_ricursive.py read by hand, for the first few layers of
the real weights. Writes each position's top-k logprobs in rp_gen's score format, and compares two score files:
   ripple_forward_ref.py ref <dir> <layers> <ids.txt> <out.jsonl> [topk]
   ripple_forward_ref.py draft <dir> <layers> <ids.txt> <dflash dir>      the DFlash2 chain after ids[:-1], anchor ids[-1]; taps layer id % layers
   ripple_forward_ref.py cmp <ref.jsonl> <test.jsonl>
"""
import json, os, struct, sys
import numpy as np

class St:
    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
        self.base, self.path = 8 + n, path
    def get(self, name):
        m = self.h[name]
        a = np.memmap(self.path, dtype=np.uint16, mode="r", offset=self.base + m["data_offsets"][0],
                      shape=tuple(m["shape"]) or (1,))
        # bf16 is the top half of fp32
        return (a.astype(np.uint32) << 16).view(np.float32).astype(np.float32)

def rms(x, w, eps=1e-6):
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w

def ref(d, nl, ids, out, topk, want_taps=None):
    st = St(d + "/model.safetensors")
    D, NH, NKV, HD = 5120, 80, 8, 128
    cfg = json.load(open(d + "/config.json"))
    T = len(ids)
    E = st.get("model.embed_tokens.weight")
    # bf16 embedding * bf16(scale), rounded to bf16 (8 bits of mantissa)
    def to_bf16(a):
        u = a.view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)
    mul = to_bf16(np.array([cfg["embed_scale"]], dtype=np.float32))[0]
    e0 = to_bf16(np.asarray(E[np.array(ids)], dtype=np.float32) * mul)
    x = e0.copy()
    inv = 1.0 / (10000.0 ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
    th = np.arange(T, dtype=np.float32)[:, None] * inv
    cos, sin = np.cos(th), np.sin(th)
    for li in range(nl):
        p = f"model.layers.{li}."; a = p + "self_attn."
        sliding = cfg["layer_types"][li] == "sliding_attention"
        print("layer", li, "sliding" if sliding else "global", flush=True)
        n = rms(x, st.get(p + "input_layernorm.weight"))
        q = (n @ st.get(a + "q_proj.weight").T).reshape(T, NH, HD)
        k = (n @ st.get(a + "k_proj.weight").T).reshape(T, NKV, HD)
        w = 1.0 / (1.0 + np.exp(-st.get(a + "value_res_lambda")[0]))
        vs = e0 + w * (n - e0)
        v = (vs @ st.get(a + "v_proj.weight").T).reshape(T, NKV, HD)
        g = n @ st.get(a + "g_proj.weight").T
        q, k = rms(q, st.get(a + "q_norm.weight")), rms(k, st.get(a + "k_norm.weight"))
        if sliding:
            def rope(y):
                c, s = cos[:, None, :], sin[:, None, :]
                y1, y2 = y[..., :HD // 2], y[..., HD // 2:]
                return np.concatenate([y1 * c - y2 * s, y2 * c + y1 * s], -1)
            q, k = rope(q), rope(k)
        o = np.zeros((T, NH, HD), dtype=np.float32)
        for h in range(NKV):
            K, V = k[:, h, :], v[:, h, :]
            for qh in range(h * 10, h * 10 + 10):
                S = (q[:, qh, :] @ K.T) / np.sqrt(HD)
                mask = np.tri(T, T, 0, dtype=bool)
                if sliding: mask &= ~np.tri(T, T, -1024, dtype=bool)
                S = np.where(mask, S, -np.inf)
                S -= S.max(-1, keepdims=True)
                P = np.exp(S); P /= P.sum(-1, keepdims=True)
                o[:, qh, :] = P @ V
        o = o.reshape(T, NH * HD) * (1.0 / (1.0 + np.exp(-g)))
        a_ = o @ st.get(a + "o_proj.weight").T
        x = x + rms(a_, st.get(p + "post_attention_layernorm.weight"))
        m = rms(x, st.get(p + "pre_feedforward_layernorm.weight"))
        gt = np.minimum(m @ st.get(p + "mlp.gate_proj.weight").T, 20.0)
        up = np.clip(m @ st.get(p + "mlp.up_proj.weight").T, -20.0, 20.0)
        f = (gt / (1.0 + np.exp(-gt)) * up) @ st.get(p + "mlp.down_proj.weight").T
        x = x + rms(f, st.get(p + "post_feedforward_layernorm.weight"))
        if want_taps is not None and li in want_taps: want_taps[li] = x.copy()
    if out is None: return st, mul
    fin = rms(x, st.get("model.norm.weight"))
    lg = fin @ st.get("lm_head.weight").T
    lg -= lg.max(-1, keepdims=True)
    lse = np.log(np.exp(lg.astype(np.float64)).sum(-1, keepdims=True))
    lp = lg - lse
    with open(out, "w") as fo:
        for t in range(T):
            idx = np.argpartition(-lp[t], topk)[:topk]
            idx = idx[np.argsort(-lp[t][idx])]
            fo.write(json.dumps({"pos": t, "top": [[int(i), float(lp[t][i])] for i in idx]}) + "\n")

def draft(d, nl, ids, dd):
    D, NH, NKV, HD, I = 5120, 80, 8, 128, 40960
    taps = {i % nl: None for i in (1, 20, 39, 58, 77)}
    st, mul = ref(d, nl, ids[:-1], None, 0, taps)
    ds = St(dd + "/model.safetensors")
    cat = np.concatenate([taps[i % nl] for i in (1, 20, 39, 58, 77)], -1)
    tn = rms(cat @ ds.get("fc.weight").T, ds.get("hidden_norm.weight"))
    Tc = tn.shape[0]
    E = st.get("model.embed_tokens.weight")
    def to_bf16(a):
        u = a.view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)
    x = to_bf16(np.asarray(E[np.array([ids[-1]] + [100278] * 15)], dtype=np.float32) * mul)
    R = 16
    inv = 1.0 / (10000.0 ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
    th = np.arange(Tc + R, dtype=np.float32)[:, None] * inv
    cos, sin = np.cos(th), np.sin(th)
    def rope(y, pos):
        c, s = cos[pos][:, None, :], sin[pos][:, None, :]
        y1, y2 = y[..., :HD // 2], y[..., HD // 2:]
        return np.concatenate([y1 * c - y2 * s, y2 * c + y1 * s], -1)
    def conv(xx, dyn, base):
        out = np.zeros_like(xx, dtype=np.float32)
        for t in range(R):
            for tap in range(min(4, t + 1)):
                w = base[tap] + np.repeat(dyn[t, tap], 16)
                out[t] += w * xx[t - tap]
        return out
    ctx = np.arange(Tc)
    blk = np.arange(Tc, Tc + R)
    for li in range(5):
        p = f"layers.{li}."; a = p + "self_attn."
        sliding = li < 4
        kp, vp = ds.get(a + "k_proj.weight"), ds.get(a + "v_proj.weight")
        Kc = rms((tn @ kp.T).reshape(Tc, NKV, HD), ds.get(a + "k_norm.weight"))
        Vc = (tn @ vp.T).reshape(Tc, NKV, HD)
        if sliding: Kc = rope(Kc, ctx)
        y = rms(x, ds.get(p + "input_layernorm.weight"))
        dyn = (y @ ds.get(p + "attention_conv.kernel_projection.weight").T).reshape(R, 2, 4, 320)
        base = ds.get(p + "attention_conv.base_kernel")
        y2 = conv(y, dyn[:, 0], base[0])
        q = rms((y2 @ ds.get(a + "q_proj.weight").T).reshape(R, NH, HD), ds.get(a + "q_norm.weight"))
        kb = rms((y2 @ kp.T).reshape(R, NKV, HD), ds.get(a + "k_norm.weight"))
        vb = (y2 @ vp.T).reshape(R, NKV, HD)
        if sliding: q, kb = rope(q, blk), rope(kb, blk)
        K, V = np.concatenate([Kc, kb], 0), np.concatenate([Vc, vb], 0)
        o = np.zeros((R, NH, HD), dtype=np.float32)
        for h in range(NKV):
            for qh in range(h * 10, h * 10 + 10):
                S = (q[:, qh, :] @ K[:, h, :].T) / np.sqrt(HD)
                S -= S.max(-1, keepdims=True)
                P = np.exp(S); P /= P.sum(-1, keepdims=True)
                o[:, qh, :] = P @ V[:, h, :]
        a_ = o.reshape(R, NH * HD) @ ds.get(a + "o_proj.weight").T
        x = x + conv(a_, dyn[:, 1], base[1])
        y = rms(x, ds.get(p + "post_attention_layernorm.weight"))
        dyn = (y @ ds.get(p + "mlp_conv.kernel_projection.weight").T).reshape(R, 2, 4, 320)
        base = ds.get(p + "mlp_conv.base_kernel")
        y2 = conv(y, dyn[:, 0], base[0])
        g = y2 @ ds.get(p + "mlp.gate_proj.weight").T
        u = y2 @ ds.get(p + "mlp.up_proj.weight").T
        f = (g / (1.0 + np.exp(-g)) * u) @ ds.get(p + "mlp.down_proj.weight").T
        x = x + conv(f, dyn[:, 1], base[1])
        print("draft layer", li, flush=True)
        if os.environ.get("RP_DUMP"):
            dp = os.environ["RP_DUMP"] + f"/L{li + 1}"
            os.makedirs(dp, exist_ok=True)
            np.save(dp + "/x.npy", x); np.save(dp + "/q.npy", q); np.save(dp + "/o.npy", o); np.save(dp + "/ka.npy", K[-16:]); np.save(dp + "/va.npy", V[-16:])
            np.save(dp + "/cv.npy", dyn.reshape(R, 2560)); np.save(dp + "/y2.npy", y2)
    state = rms(x, ds.get("norm.weight"))[1:]
    lg = state @ st.get("lm_head.weight").T
    gate = state @ ds.get("candidate_selector.hidden_projection.weight").T
    A, B = ds.get("candidate_selector.predecessor_codebook"), ds.get("candidate_selector.successor_codebook")
    pred, chain = ids[-1], []
    for r in range(15):
        c = np.argpartition(-lg[r], 32)[:32]
        s = lg[r][c] + (B[c] @ (A[pred] * gate[r]))
        pred = int(c[np.argmax(s)])
        chain.append(pred)
    print("chain:", " ".join(map(str, chain)))
    if os.environ.get("RP_DUMP"): np.save(os.environ["RP_DUMP"] + "/state.npy", state)

def cmp(a, b):
    ra = {r["pos"]: r["top"] for r in map(json.loads, open(a))}
    rb = {r["pos"]: r["top"] for r in map(json.loads, open(b))}
    keys = sorted(set(ra) & set(rb))
    agree, kls, dmax = 0, [], 0.0
    worst = (0, -1)
    for p in keys:
        ta, tb = dict(map(tuple, ra[p])), dict(map(tuple, rb[p]))
        agree += ra[p][0][0] == rb[p][0][0]
        # KL over the reference's top k (the mass there is what matters)
        kl = 0.0
        floor = min(tb.values())
        for i, lp in ta.items():
            kl += np.exp(lp) * (lp - tb.get(i, floor))
        kls.append(kl)
        if kl > worst[0]: worst = (kl, p)
        for i, lp in ta.items():
            if i in tb: dmax = max(dmax, abs(lp - tb[i]))
    print(f"{len(keys)} positions: top-1 agree {100.0 * agree / len(keys):.2f}%, KL mean {np.mean(kls):.6f} max {worst[0]:.6f} at {worst[1]}, largest logprob gap in the top-k {dmax:.4f}")

if __name__ == "__main__":
    if sys.argv[1] == "ref":
        ids = [int(t) for t in open(sys.argv[4]).read().split()]
        ref(sys.argv[2], int(sys.argv[3]), ids, sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 64)
    elif sys.argv[1] == "draft":
        ids = [int(t) for t in open(sys.argv[4]).read().split()]
        draft(sys.argv[2], int(sys.argv[3]), ids, sys.argv[5])
    else:
        cmp(sys.argv[2], sys.argv[3])
