#!/usr/bin/env python3
"""Pack GLM-5.3-Flash EXL3 routed experts into one O_DIRECT-friendly store on NVMe, in exl3xpu BLOB order (B70 port).

B70 port: same records as glm-flash-lite's scripts/pack_glm53_store.py, but every record is word-permuted into the
exl3xpu expert blob (b70tier/glmperm.make_perm: planar4(gate|up), planar4(down), g.suh u.suh g.svh u.svh d.suh d.svh),
so NVMe -> RAM tier -> VRAM slot is a plain copy and the B70 MoE kernels can read a RAM-tier record zero-copy. sha256
hashes are of the stored (permuted) bytes. Needs numpy. Original docstring follows.

(Campaign N116; the measured store sits on a 4-drive NVMe RAID0 at /mnt/nvx/glm53.)

Python stdlib only, CPU only. Reads the checkpoint read-only (safetensors headers + pread), writes only under --out-dir.

Store layout (little-endian, every record 4096-aligned, layer-major):
  record index r = li * E + e     li = 0..L-1 over the MoE layers in model order (model layers 3..44), then the MTP
                                  layer (model layer 45, mtp.safetensors) as li = 42 when --mtp is on
  record bytes  [gate.trellis][up.trellis][down.trellis] [gate.suh][gate.svh][up.suh][up.svh][down.suh][down.svh] [pad]
  The trellis part (first `trellis_bytes` = 9,437,184 B for 3.05bpw) is byte-identical to one VRAM slot of
  expert_cache.py (slot = gate | up @ off_u | down @ off_d, each 256-aligned; here all three are 3,145,728 B so
  off_u = 3,145,728, off_d = 6,291,456 and the slot has no padding), so an NVMe -> RAM slot -> VRAM slot fill is one
  contiguous copy and the pointer tables become (a, a + off_u, a + off_d). A reader that keeps suh/svh resident reads
  only the trellis part (also 4096-aligned).
  The .mul1 codebook markers (one I32 scalar per matrix) are checked to be identical and recorded in the manifest.

Outputs in --out-dir: <name>.bin (records), <name>.json (layout + per-record sha256 of the source bytes).

Modes:
  pack            write the store (O_DIRECT, preallocated), hash every record from the SOURCE buffer before the write
  verify          O_DIRECT re-read of every record, sha256 vs the manifest (independent of the page cache)
  cmp             byte-exact compare of --sample N random records (+ first/last of every layer) against fresh source reads
"""
import argparse, concurrent.futures as cf, hashlib, json, mmap, os, random, re, struct, sys, time
import numpy as np


def blob_perm(H=4096, I=2048, K=3):
    """numpy int64 [rec/4]: blob word j = raw record word perm[j] (same as glm-flash-lite b70tier/glmperm.py for K=3).

    A 16x16 tile is G periods of D words (D = K / gcd(K, 32), G = 8K / D). planar4 regroups 4 consecutive tiles plane by
    plane ([D][4 tiles][G]), which is what the kernel's load_planar expects; for K = 2 or 4 (D = 1) it is the identity."""
    from math import gcd
    W = 8 * K
    D = K // gcd(K, 32)
    G = W // D
    ng = (H // 16) * (I // 16) * W
    r = np.arange(3 * ng + (3 * H + 3 * I) // 2, dtype=np.int64)
    gate = r[0:ng].reshape(H // 16, I // 16, W)
    up = r[ng:2 * ng].reshape(H // 16, I // 16, W)
    down = r[2 * ng:3 * ng].reshape(I // 16, H // 16, W)

    def p4(w):
        rows, n, _ = w.shape
        return w.reshape(rows, n // 4, 4, G, D).transpose(0, 1, 4, 2, 3).reshape(-1)

    o = 3 * ng
    sec = {}
    for name, n in (("g_suh", H), ("g_svh", I), ("u_suh", H), ("u_svh", I), ("d_suh", I), ("d_svh", H)):
        sec[name] = r[o:o + n // 2]
        o += n // 2
    assert o == r.size
    perm = np.concatenate([p4(np.concatenate([gate, up], axis=1)), p4(down), sec["g_suh"], sec["u_suh"], sec["g_svh"],
                           sec["u_svh"], sec["d_suh"], sec["d_svh"]])
    assert perm.size == r.size and np.array_equal(np.sort(perm), r)
    return perm


PERMS = {}


def permute_into(raw_mv, out_mv, K):
    """out (blob order) = raw record words gathered by blob_perm(K)."""
    perm = PERMS.get(K)
    if perm is None:
        perm = PERMS[K] = blob_perm(K=K)
    src = np.frombuffer(raw_mv, dtype=np.uint32, count=perm.size)
    dst = np.frombuffer(out_mv, dtype=np.uint32, count=perm.size)
    np.take(src, perm, out=dst)

ALIGN = 4096
MATS = ("gate_proj", "up_proj", "down_proj")


def al(x, a=ALIGN):
    return (x + a - 1) // a * a


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


class Src:
    """Tensor locator over the checkpoint shards (read-only)."""

    def __init__(self, model_dir, use_mtp):
        self.dir = model_dir
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        files = sorted(set(idx.values()))
        if use_mtp and os.path.exists(os.path.join(model_dir, "mtp.safetensors")):
            files.append("mtp.safetensors")
        self.loc = {}
        self.fds = {}
        for fn in files:
            base, h = st_header(os.path.join(model_dir, fn))
            for k, v in h.items():
                if ".mlp.experts." not in k:
                    continue
                a, b = v["data_offsets"]
                self.loc[k] = (fn, base + a, b - a, v["dtype"], tuple(v["shape"]))
            self.fds[fn] = os.open(os.path.join(model_dir, fn), os.O_RDONLY)
        layers = {}
        for k in self.loc:
            m = re.match(r"(.*\.layers\.(\d+)\.mlp)\.experts\.(\d+)\.", k)
            layers.setdefault(int(m.group(2)), [m.group(1), set()])[1].add(int(m.group(3)))
        self.layers = [(l, layers[l][0], len(layers[l][1])) for l in sorted(layers)]

    def read(self, key, mv):
        fn, off, n, _, _ = self.loc[key]
        got = 0
        while got < n:
            r = os.preadv(self.fds[fn], [mv[got:n]], off + got)
            assert r > 0, (key, got, n)
            got += r
        return n


def layout(src, li):
    """Field list (key suffix, offset, bytes) for one record of layer li, from its expert-0 shapes (asserted for all).
    Returns (fields, trellis bytes, record bytes, K)."""
    _, pfx, _ = src.layers[li]
    fields, off = [], 0
    for m in MATS:
        k = f"{pfx}.experts.0.{m}.trellis"
        n = src.loc[k][2]
        fields.append((f"{m}.trellis", off, n, src.loc[k][3], src.loc[k][4])); off += n
    trellis = off
    assert trellis % ALIGN == 0, trellis
    for m in MATS:
        for s in ("suh", "svh"):
            k = f"{pfx}.experts.0.{m}.{s}"
            n = src.loc[k][2]
            fields.append((f"{m}.{s}", off, n, src.loc[k][3], src.loc[k][4])); off += n
    K = fields[0][4][-1] // 16                 # trellis [k/16, n/16, 16K]
    return fields, trellis, al(off), K


def check_uniform(src, lays):
    mul1 = set()
    for li, (fields, _, _, _) in enumerate(lays):
        _, pfx, E = src.layers[li]
        for e in range(E):
            for name, _, n, dt, shp in fields:
                v = src.loc[f"{pfx}.experts.{e}.{name}"]
                assert v[2] == n and v[3] == dt and v[4] == shp, (pfx, e, name, v, n, dt, shp)
            for m in MATS:
                k = f"{pfx}.experts.{e}.{m}.mul1"
                if k in src.loc:
                    fn, off, n, dt, _ = src.loc[k]
                    mul1.add(os.pread(src.fds[fn], n, off).hex())
    return sorted(mul1)


def fill_record(src, pfx, e, fields, buf):
    mv = memoryview(buf)
    for name, off, n, _, _ in fields:
        src.read(f"{pfx}.experts.{e}.{name}", mv[off:off + n])
    return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pack", "verify", "cmp"])
    ap.add_argument("--model", default=os.environ.get("GLM53_MODEL_DIR", "/models"), help="EXL3 checkpoint directory")
    ap.add_argument("--out-dir", default=os.environ.get("GLM53_NV_STORE_DIR", "/nvx"), help="store directory (NVMe, O_DIRECT capable)")
    ap.add_argument("--name", default="glm53_flash_exl3_experts_blob")
    ap.add_argument("--mtp", type=int, default=1, help="append the MTP layer's experts as the last layer")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=116)
    a = ap.parse_args()
    out_bin = os.path.join(a.out_dir, a.name + ".bin")
    out_meta = os.path.join(a.out_dir, a.name + ".json")
    assert os.path.isdir(a.out_dir), f"--out-dir {a.out_dir} does not exist"
    assert os.path.realpath(a.out_dir) != os.path.realpath(a.model), "--out-dir must not be the checkpoint directory"
    t0 = time.time()
    src = Src(a.model, bool(a.mtp))

    if a.mode == "pack":
        # Bit width (K) can differ between layers (mixed-bpw quants: e.g. 2.51 bpw = 22 layers at K=2, 20 at K=3), so
        # record size is per layer; records stay layer-major and 4096-aligned, layer li starts at layers[li].offset.
        L = len(src.layers)
        lays = [layout(src, li) for li in range(L)]
        E = src.layers[0][2]
        assert all(x[2] == E for x in src.layers)
        mul1 = check_uniform(src, lays)
        offs, total = [], 0
        for fields, trellis, rec, K in lays:
            assert rec == fields[-1][1] + fields[-1][2] == blob_perm(K=K).size * 4, (rec, K)
            offs.append(total)
            total += E * rec
        os.makedirs(a.out_dir, exist_ok=True)
        st = os.statvfs(a.out_dir)
        free = st.f_bavail * st.f_frsize
        ks = [x[3] for x in lays]
        print(f"layers {L} ({src.layers[0][0]}..{src.layers[-1][0]}), E {E}, K per layer {ks}, total "
              f"{total / 1e9:.2f} GB, free {free / 1e12:.2f} TB, mul1 markers {mul1}", flush=True)
        assert total < free * 0.9, "not enough free space"
        fd = os.open(out_bin, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        os.posix_fallocate(fd, 0, total)
        hashes = [None] * (L * E)
        import threading
        tl = threading.local()
        rec_max = max(x[2] for x in lays)

        def job(r):
            li, e = divmod(r, E)
            fields, _, rec, K = lays[li]
            if not hasattr(tl, "buf"):
                tl.buf = mmap.mmap(-1, rec_max)      # page-aligned (O_DIRECT source)
                tl.raw = mmap.mmap(-1, rec_max)
            raw = fill_record(src, src.layers[li][1], e, fields, tl.raw)
            mv = memoryview(tl.buf)[:rec]
            permute_into(raw, mv, K)
            h = hashlib.sha256(mv).hexdigest()
            n = os.pwrite(fd, mv, offs[li] + e * rec)
            assert n == rec, (r, n)
            hashes[r] = h
            return r

        done = 0
        with cf.ThreadPoolExecutor(a.threads) as ex:
            for li in range(L):
                tl0 = time.time()
                list(ex.map(job, range(li * E, (li + 1) * E)))
                done += E * lays[li][2]
                dt = time.time() - tl0
                print(f"  layer {li:2d} (model {src.layers[li][0]}, K {lays[li][3]}) {E * lays[li][2] / 1e9:.2f} GB in "
                      f"{dt:.1f} s, total {done / 1e9:.1f} GB, {time.time() - t0:.0f} s", flush=True)
        os.fsync(fd); os.close(fd)
        meta = dict(format="glm53-expert-store-blob-v2", word_order="exl3xpu blob (pack_store.blob_perm(4096, 2048, K))",
                    model_dir=a.model, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), align=ALIGN,
                    experts_per_layer=E, n_layers=L,
                    order="record (li, e) at layers[li].offset + e * layers[li].record_bytes (layer-major)",
                    layers=[dict(li=i, model_layer=l, key=p, K=lays[i][3], record_bytes=lays[i][2],
                                 trellis_bytes=lays[i][1], offset=offs[i],
                                 fields=[dict(name=n, offset=o, bytes=b, dtype=d, shape=list(s))
                                         for n, o, b, d, s in lays[i][0]])
                            for i, (l, p, _) in enumerate(src.layers)],
                    mtp_layer_li=(L - 1 if a.mtp and src.layers[-1][0] == 45 else None),
                    mul1_markers_hex=mul1, sha256=hashes, total_bytes=total)
        tmp = out_meta + ".tmp"
        json.dump(meta, open(tmp, "w"))
        os.replace(tmp, out_meta)
        print(f"PACKED {L * E} records, {total / 1e9:.2f} GB in {time.time() - t0:.0f} s -> {out_bin}", flush=True)
        return

    meta = json.load(open(out_meta))
    assert meta["format"] == "glm53-expert-store-blob-v2", meta["format"]
    E, L = meta["experts_per_layer"], meta["n_layers"]
    lays = meta["layers"]
    rec_max = max(l["record_bytes"] for l in lays)
    assert os.path.getsize(out_bin) == meta["total_bytes"]
    fd = os.open(out_bin, os.O_RDONLY | os.O_DIRECT)
    import threading
    tl = threading.local()

    def rd(r):
        li, e = divmod(r, E)
        rec = lays[li]["record_bytes"]
        if not hasattr(tl, "buf"):
            tl.buf = mmap.mmap(-1, rec_max)
        mv = memoryview(tl.buf)[:rec]
        n = os.preadv(fd, [mv], lays[li]["offset"] + e * rec)
        assert n == rec, (r, n)
        return mv

    if a.mode == "verify":
        bad = []

        def job(r):
            h = hashlib.sha256(rd(r)).hexdigest()
            if h != meta["sha256"][r]:
                bad.append(r)

        with cf.ThreadPoolExecutor(a.threads) as ex:
            for li in range(L):
                list(ex.map(job, range(li * E, (li + 1) * E)))
        dt = time.time() - t0
        tot = meta["total_bytes"]
        print(f"VERIFY {'OK' if not bad else 'FAIL'}: {L * E} records, {tot / 1e9:.2f} GB O_DIRECT re-read + sha256 "
              f"in {dt:.0f} s ({tot / 1e9 / dt:.2f} GB/s); bad {len(bad)} {bad[:20]}", flush=True)
        sys.exit(1 if bad else 0)

    # cmp: byte-exact against fresh source reads (different code path from the packer's buffer reuse)
    rng = random.Random(a.seed)
    pick = set(rng.sample(range(L * E), min(a.sample, L * E)))
    for li in range(L):
        pick.update((li * E, li * E + E - 1))
    pick = sorted(pick)
    bad = []
    for r in pick:
        li, e = divmod(r, E)
        lay = lays[li]
        rec, pfx = lay["record_bytes"], lay["key"]
        got = rd(r)
        raw = bytearray(rec)
        for f in lay["fields"]:
            name, off, n = f["name"], f["offset"], f["bytes"]
            fn, soff, sn, _, _ = src.loc[f"{pfx}.experts.{e}.{name}"]
            assert sn == n
            raw[off:off + n] = os.pread(src.fds[fn], n, soff)
        want = bytearray(rec)
        permute_into(memoryview(raw), memoryview(want), lay["K"])
        if bytes(got) != bytes(want):
            bad.append((r, "record"))
    print(f"CMP {'OK' if not bad else 'FAIL'}: {len(pick)} records byte-compared against the checkpoint (every field + "
          f"zero pad), {time.time() - t0:.0f} s; mismatches {len(bad)} {bad[:20]}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
