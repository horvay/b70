"""Routed-expert tiers for GLM-5.3-Flash on the B70: VRAM slot cache + RAM tier + NVMe store (exact, host-managed).

The store (xpu/pack_store.py) holds every routed expert as one 9,474,048-byte record in exl3xpu blob order, so a record
is copied unchanged NVMe -> RAM slot -> VRAM slot. The B70 MoE kernel (`exl3xpu_moe.moe_forward`, `_moe_glm.so`) reads
experts only from VRAM, through a per-layer pointer table.

Per MoE layer call:
  decode / small batches (picks <= ADMIT_MAX):  host sync on the router's picks; every picked expert that is not in
      a VRAM slot is admitted (LRU victim that is not picked by this call): RAM slot -> VRAM, or NVMe -> RAM slot -> VRAM.
      The layer's pointer table is updated and the kernel runs on the slots.
  prefill (more picks):  VRAM-resident experts are used in place; every other picked expert is copied into a
      dedicated staging area (one slot per expert of the layer) from RAM, or straight from NVMe (bypassing the RAM tier,
      so a long prompt does not flush it), and a temporary pointer table points at slot or staging.

The RAM tier is pinned host memory and EXCLUSIVE of VRAM (as in glm-flash-lite's nvme mode): admitting an expert to VRAM
frees its RAM slot, and a VRAM victim is written back to a RAM slot. The write-back goes VRAM slot -> VRAM victim ring
(device copy, microseconds, on the compute stream, before the slot is refilled) -> RAM slot (copy engine, on a side
stream), so it stays off the critical path; an expert whose write-back has not landed is still usable (the compute
stream waits on its event).
"""
from __future__ import annotations

import concurrent.futures as cf
import ctypes
import json
import mmap
import os
import threading
import time
from collections import OrderedDict

import numpy as np
import torch

H, I, K = 4096, 2048, 3


def kcode(K):
    """Store K (int, or half-integer for exllamav3 1.6 "sc" quants) -> the MoE kernel's K code (exl3_esimd.h Geo)."""
    K = float(K)
    if K.is_integer():
        return int(K)
    assert K * 2 == int(K * 2), K
    return int(K) * 10 + 5


def log(*a):
    print(" -- glm_xpu.tier:", *a, flush=True)


class _FlagDict(dict):
    """dict that mirrors its keys into bit `bit` of a uint8 flag array (tier.flags: 1 in flight, 2 write-back pending),
    which the native decode step (tier_native.Core) reads instead of the dicts."""
    __slots__ = ("flags", "bit")

    def __init__(self, flags, bit):
        super().__init__()
        self.flags, self.bit = flags, bit

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.flags[k] |= self.bit

    def __delitem__(self, k):
        super().__delitem__(k)
        self.flags[k] &= ~self.bit & 0xFF

    def pop(self, k, *d):
        if k in self:
            self.flags[k] &= ~self.bit & 0xFF
        return super().pop(k, *d)

    def clear(self):
        for k in self:
            self.flags[k] &= ~self.bit & 0xFF
        super().clear()


class _NativeRead:
    """Future-like handle of a tier_native read job (done / result, as the thread-pool futures it replaces)."""
    __slots__ = ("r", "id")

    def __init__(self, reader, jid):
        self.r, self.id = reader, jid

    def done(self):
        return self.r.done(self.id)

    def result(self):
        try:
            self.r.wait(self.id)
        finally:
            self.r.forget(self.id)

    def __del__(self):
        try:
            self.r.forget(self.id)
        except Exception:      # noqa: BLE001  (interpreter shutdown)
            pass


class Tier:
    def __init__(self, store_bin: str, device, vram_bytes: int, ram_bytes: int, n_stage: int = 288,
                 readers: int = 16, admit_max: int = 192, pf_slots: int = 384):
        self.dev = torch.device(device)
        meta = json.load(open(store_bin.rsplit(".", 1)[0] + ".json"))
        assert meta["format"] == "glm53-expert-store-blob-v2", meta["format"]
        self.E = int(meta["experts_per_layer"])
        self.L = int(meta["n_layers"])
        lays = meta["layers"]
        self.layer_li = {l["model_layer"]: l["li"] for l in lays}
        mtp = meta.get("mtp_layer_li")
        self.n_moe = mtp if mtp is not None else self.L       # MoE layers of the main model (the MTP layer is last)
        self.fd = os.open(store_bin, os.O_RDONLY | os.O_DIRECT)
        self.admit_max = admit_max
        # prompts with fewer picks than this read their non-resident experts on demand instead of streaming the
        # layers ahead (stage_layer)
        self.pf_min_picks = int(os.environ.get("GLM_XPU_PF_MIN_PICKS", "2048"))
        self.pf_demand = False
        self.refill_nvme = os.environ.get("GLM_XPU_REFILL_NVME", "0") == "1"

        from glm_xpu.ops import moe_ops
        self.X = moe_ops()
        # Bit width can differ per layer (2.51 bpw: K=2 and K=3 layers), so record size is per layer. Slots come in one
        # pool per size class (class c = one K); a key only ever uses slots of its own class.
        self.K_of = [kcode(l["K"]) for l in lays]           # kernel K codes (half-integer K: 10 * KA + 5)
        self.rec_of = np.array([int(l["record_bytes"]) for l in lays], np.int64)
        self.off_of = np.array([int(l["offset"]) for l in lays], np.int64)
        for K, rec in zip(self.K_of, self.rec_of):
            assert int(self.X.blob_bytes(H, I, K)) == rec, (K, rec)
        self.cK = sorted(set(self.K_of), reverse=True)        # class -> K
        self.crec = [int(self.X.blob_bytes(H, I, K)) for K in self.cK]
        cls_of_layer = np.array([self.cK.index(K) for K in self.K_of], np.int64)
        self.C = len(self.cK)
        self.rec_max = max(self.crec)
        self.rec = self.rec_max                               # staging / ring / bounce row size

        LE = self.L * self.E
        self.key_cls = np.repeat(cls_of_layer, self.E)        # key -> class
        self.key_rec = self.rec_of[np.arange(LE) // self.E]
        self.key_off = self.off_of[np.arange(LE) // self.E] + (np.arange(LE) % self.E) * self.key_rec
        # split a byte budget over the classes so every MoE layer can hold the same fraction of its experts
        nl = [int((cls_of_layer[:self.n_moe] == c).sum()) for c in range(self.C)]

        # a class used only by the MTP layer (2.51 bpw: K=4) gets a small fixed pool when speculative decoding is on
        mtp_on = int(os.environ.get("GLM_XPU_MTP", "0")) > 0
        fixed = [(96 if mtp_on else 0) if n == 0 else 0 for n in nl]

        from glm_xpu.tier_layout import split_slots

        def split(budget, minimum=None):
            return split_slots(budget, nl, self.crec, fixed, minimum)

        from glm_xpu.ops import s64
        self.s64 = s64
        # VRAM slots: global slot ids, class c owns ids [vbase[c], vbase[c] + n_c)
        main_cls = sorted(set(cls_of_layer[:self.n_moe].tolist()))
        self.stage_cls = max(main_cls, key=lambda c: self.crec[c])
        # A rare large-record class may receive fewer slots than one layer's staging needs. Reserve its minimum
        # inside the cache budget before distributing the rest, rather than allocating extra VRAM during prefill.
        minimum = [n_stage if c == self.stage_cls else 0 for c in range(self.C)]
        nv = split(vram_bytes, minimum)
        self.S = sum(nv)
        self._vram_t = []
        self.vrow, vaddr, vcls = [], [], []
        for c, n in enumerate(nv):
            t = torch.empty((max(n, 1), self.crec[c]), dtype=torch.uint8, device=self.dev)
            self._vram_t.append(t)
            self.vrow.extend(t[i] for i in range(n))
            vaddr.extend(s64(t.data_ptr() + i * self.crec[c]) for i in range(n))
            vcls.extend([c] * n)
        self.vrow64 = [r.view(torch.int64) for r in self.vrow]
        self.vaddr = np.array(vaddr, np.int64)
        self.vcls = np.array(vcls, np.int64)
        self.vram_of = np.full(LE, -1, np.int64)        # key -> slot
        self.vram_key = np.full(self.S, -1, np.int64)   # slot -> key
        # LRUs: tier_native.Lru (the OrderedDict subset used here, in C++ so the native decode step can touch and scan
        # them) unless GLM_XPU_NATIVE_CORE=0 or the module is not built
        try:
            from exl3xpu import tier_native as _tn
        except ImportError:
            _tn = None
        self._tn = _tn if os.environ.get("GLM_XPU_NATIVE_CORE", "1") == "1" else None
        _lru = (lambda: self._tn.Lru(LE)) if self._tn is not None else OrderedDict
        self.vram_lru = [_lru() for _ in range(self.C)]   # per class: resident key -> slot, LRU first
        self.free_vram = [[s for s in range(self.S - 1, -1, -1) if vcls[s] == c] for c in range(self.C)]
        # Prefill staging borrows the last n_stage slots of the largest main-model class (a smaller record uses a row's
        # prefix): a long prompt drops their experts for the duration of the forward, and the next decode step refills
        # them with the hottest non-resident experts. Decode keeps the whole pool.
        self.n_stage = n_stage
        self.stage_row = self.crec[self.stage_cls]
        n_c = nv[self.stage_cls]
        assert n_c >= n_stage, f"VRAM pool of K{self.cK[self.stage_cls]} ({n_c} slots) smaller than staging ({n_stage})"
        s0 = sum(nv[:self.stage_cls])
        self.stage_slots = list(range(s0 + n_c - n_stage, s0 + n_c))
        self.stage = self._vram_t[self.stage_cls][n_c - n_stage:]
        self.stage_borrowed = False
        self.need_refill = False
        self.warm_order = []
        # Layer-ahead staging (GLM_XPU_STAGE_AHEAD, decode): at a layer's routing sync, the next MoE layer's predicted
        # experts that sit in RAM are copied into reserved VRAM rows on a copy stream, overlapping this layer's work;
        # the next layer then reads the ones it picks from VRAM instead of zero-copy over PCIe (2 RAM picks made a
        # layer's MoE 0.22 -> 0.70 ms, tests/moebench.py; 89% of RAM picks were predicted a layer ahead). Two row sets
        # by layer parity; rows of the most common class only (sc_2.60: K2.5).
        # off by default: forced 200-token pass 25.7 s without, 31.2 s with 12 predicted per layer (~45% of staged rows
        # used: the extra PCIe traffic costs more than it hides), 27.4 s top-8, 28.1 s top-5 (2026-10-30)
        self.sa_on = os.environ.get("GLM_XPU_STAGE_AHEAD", "0") == "1"
        self.sa_rows = None
        self.sa_map = {}                     # tier layer -> (expert -> VRAM slot, copy event, parity)
        self.sa_cdone = [None, None]         # per parity: event after the last expert phase that read those rows
        if self.sa_on:
            main_counts = [int((cls_of_layer[:self.n_moe] == c).sum()) for c in range(self.C)]
            self.sa_cls = int(np.argmax(main_counts))
            n_sa = int(os.environ.get("GLM_XPU_STAGE_AHEAD_ROWS", "12"))
            st_set = set(self.stage_slots)
            fl = self.free_vram[self.sa_cls]
            take = [x for x in fl if x not in st_set][:2 * n_sa]
            if len(take) == 2 * n_sa:
                tk = set(take)
                fl[:] = [x for x in fl if x not in tk]
                self.sa_rows = [take[:n_sa], take[n_sa:]]
                self.sa_stream = torch.xpu.Stream(self.dev)
            else:
                self.sa_on = False
        # RAM tier: pinned (USM host) memory, so RAM -> VRAM copies are asynchronous DMA at full PCIe speed. O_DIRECT
        # reads into USM host memory fail (EFAULT), so NVMe reads land in a per-thread anonymous bounce and are copied
        # into the slot by the reader thread.
        # A single pinned allocation above ~4 GB silently fails on XPU (null pointer, not pinned), and the pinned
        # allocator rounds up to a power of two, so the tier is built from chunks just under 2 GiB; ram_t / ram_np are
        # per-slot row lists.
        nr = split(ram_bytes)
        self.R = sum(nr)
        self._ram_chunks = []
        self.ram_t, self.ram_np, rcls = [], [], []
        for c, n_c in enumerate(nr):
            per_chunk = (2**31 - 1) // self.crec[c]
            left = n_c
            while left > 0:
                n = min(per_chunk, left)
                ch = torch.empty((n, self.crec[c]), dtype=torch.uint8, pin_memory=True)
                assert ch.is_pinned() and ch.data_ptr() != 0, "pinned RAM-tier chunk allocation failed"
                self._ram_chunks.append(ch)
                cn = ch.numpy()
                self.ram_t.extend(ch[i] for i in range(n))
                self.ram_np.extend(cn[i] for i in range(n))
                rcls.extend([c] * n)
                left -= n
        # Write every page once before any read lands there. The fill's 16 readers faulting in fresh pinned pages
        # (alongside their O_DIRECT reads) filled 13.5 GB at 0.6-0.8 GB/s; pre-touched it fills at ~7 GB/s
        # (tests/fillbench.py, 2026-10-29). One thread per chunk, numpy releases the GIL.
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(len(self._ram_chunks) or 1) as ex:
            list(ex.map(lambda ch: ch.numpy().fill(0), self._ram_chunks))
        log(f"RAM tier: {sum(ch.numel() for ch in self._ram_chunks) / 2**30:.1f} GiB pinned and touched in "
            f"{time.perf_counter() - t0:.1f} s")
        self.rcls = np.array(rcls, np.int64)
        self.n_ram_cls = [int((self.rcls == c).sum()) for c in range(self.C)]
        self.ram_of = np.full(LE, -1, np.int64)
        self.ram_key = np.full(self.R, -1, np.int64)
        self.ram_lru = [_lru() for _ in range(self.C)]    # per class: RAM key -> slot, LRU first
        self.free_ram = [[s for s in range(self.R - 1, -1, -1) if rcls[s] == c] for c in range(self.C)]
        # NVMe ring for experts that bypass the RAM tier (prefill), pinned
        self.n_bounce = 32
        self.bounce_t = torch.empty((self.n_bounce, self.rec_max), dtype=torch.uint8, pin_memory=True)
        self.bounce_np = self.bounce_t.numpy()
        self._tl = threading.local()
        # pointer tables: one [E] int64 table per layer, device + host mirror
        self.ptrs = torch.zeros((self.L, self.E), dtype=torch.int64, device=self.dev)
        # host mirror in pinned memory so a layer's table uploads asynchronously (a pageable copy blocks the host
        # until every queued expert copy has run). A layer's row is only rewritten after the next layer's routing
        # sync, by which time its previous upload has completed.
        self.ptrs_pin = torch.zeros((self.L, self.E), dtype=torch.int64, pin_memory=True)
        self.ptrs_h = self.ptrs_pin.numpy()
        self.pool = cf.ThreadPoolExecutor(readers)
        self.vring_n = 48
        self.vring = torch.empty((self.vring_n, self.rec_max), dtype=torch.uint8, device=self.dev)
        self.vring_free = list(range(self.vring_n))
        self.vring64 = self.vring.view(torch.int64)
        self.flags = np.zeros(LE, np.uint8)             # per key: 1 read in flight, 2 write-back pending (_FlagDict)
        self.wb = _FlagDict(self.flags, 2)              # key -> (event, ring slot) of a write-back not yet landed
        self.side = torch.xpu.Stream(self.dev)
        self.n_pieces = 8                               # each record is read as 8 pieces in parallel
        self._pf_init(pf_slots)
        self.inflight = _FlagDict(self.flags, 1)        # key -> futures of a prefetch read into its RAM slot
        self._demand = set()                            # in-flight keys that begin_demand() read for the current layer
        self._sa_cur = None
        # residency bias for router substitution (block_graphs.SUBST_KEEP < 8): 0 for an expert in VRAM or RAM,
        # -1e4 otherwise. Host mirror in pinned memory, uploaded (async) before each layer's router when it changed.
        self.res_pin = torch.full((self.L, self.E), -1e4, dtype=torch.float32, pin_memory=True)
        self.res_h = self.res_pin.numpy()
        self.res_dev = torch.full((self.L, self.E), -1e4, dtype=torch.float32, device=self.dev)
        # 1.0 where the expert is in VRAM (no-sync router's RAM margin, block_graphs.set_margin), refreshed with ptrs
        self.vram_pin = torch.zeros((self.L, self.E), dtype=torch.float32, pin_memory=True)
        self.vram_dev = torch.zeros((self.L, self.E), dtype=torch.float32, device=self.dev)
        self.res_dirty = True
        self.upload_res = True                          # block_graphs turns it off when no router reads res_dev
        # router residency bonus (block_graphs RES_BONUS): added to an expert's selection score, bv for one in VRAM, br
        # for one in RAM (read over PCIe on every use, so it gets less), 0 on NVMe; rebuilt from res_h and vram_of at
        # each upload
        self.bonus_v = float(os.environ.get("GLM_XPU_RES_BONUS", "0"))
        self.bonus_r = float(os.environ.get("GLM_XPU_RES_BONUS_RAM", str(self.bonus_v)))
        self.bonus_pin = torch.zeros((self.L, self.E), dtype=torch.float32, pin_memory=True)
        self.bonus_h = self.bonus_pin.numpy().reshape(-1)
        self.bonus_dev = torch.zeros((self.L, self.E), dtype=torch.float32, device=self.dev)
        self.bonus_on = bool(self.bonus_v or self.bonus_r)
        # NVMe reads: the native reader (csrc/tier_native.cpp: C++ threads, no GIL) unless GLM_XPU_NATIVE_READER=0 or
        # it is not built; it marks a background read's expert resident itself and raises _ndirty for sync_res
        self._ndirty = np.zeros(1, np.uint8)
        self.native = None
        if os.environ.get("GLM_XPU_NATIVE_READER", "1") == "1":
            try:
                from exl3xpu import tier_native
                self.native = tier_native.Reader(
                    self.fd, readers, 4 << 20, self.res_h.ctypes.data, self.bonus_h.ctypes.data,
                    self.vram_of.ctypes.data, self.ram_of.ctypes.data, self._ndirty.ctypes.data,
                    self.bonus_v, self.bonus_r, self.bonus_on)
            except ImportError as e:
                log(f"native reader not built ({e}); Python reader threads")
        if self.native is not None:
            # drain and join the reader threads before interpreter teardown frees the buffers they write into
            import atexit
            atexit.register(self.native.stop)
        self.ram_addr = np.array([s64(t.data_ptr()) for t in self.ram_t], np.int64)
        self.zero_copy = True
        self.admit_min = float(os.environ.get("GLM_XPU_ADMIT_MIN", "4"))     # decayed pick count needed for VRAM admission
        self.admit_per_call = int(os.environ.get("GLM_XPU_ADMIT_PER_CALL", "2"))   # RAM -> VRAM copies per layer call
        self.freq = np.zeros(LE, np.float32)
        self.hits = np.zeros(LE, np.float64)   # undecayed decode pick counts, saved as the routing profile (warm order)
        self.freq_decay_every = 42 * 32 # halve every ~32 decode tokens
        self.tick = 0
        # the common part of ensure() in C++ (tier_native.Core.ensure_fast); the arrays it holds pointers to are never
        # reallocated
        self.core = None
        if self._tn is not None:
            self.vcls_a = np.ascontiguousarray(self.vcls, np.int64)
            self.rcls_a = np.ascontiguousarray(self.rcls, np.int64)
            self.key_cls_a = np.ascontiguousarray(self.key_cls, np.int64)
            for a in (self.vram_of, self.ram_of, self.vaddr, self.ram_addr):
                assert a.dtype == np.int64 and a.flags["C_CONTIGUOUS"]
            assert self.ptrs_h.flags["C_CONTIGUOUS"] and self.freq.dtype == np.float32 and self.hits.dtype == np.float64
            self.core = self._tn.Core(self.L, self.E, self.key_cls_a.ctypes.data, self.vcls_a.ctypes.data,
                                      self.rcls_a.ctypes.data, self.vram_of.ctypes.data, self.ram_of.ctypes.data,
                                      self.ptrs_h.ctypes.data, self.vaddr.ctypes.data, self.ram_addr.ctypes.data,
                                      self.freq.ctypes.data, self.hits.ctypes.data, self.flags.ctypes.data,
                                      self.vram_lru, self.ram_lru, self.admit_min, self.admit_per_call)
        self.stats = dict(calls=0, picks=0, vram_hit=0, ram_hit=0, nvme=0, prefill_stage=0, prefill_nvme=0,
                          t_nvme=0.0, t_h2d=0.0, writeback=0)
        cl = ", ".join(f"K{self.cK[c]}: {nv[c]} VRAM / {nr[c]} RAM ({nl[c]} layers)" for c in range(self.C))
        log(f"store {store_bin}: {self.L} layers x {self.E} experts; VRAM slots {self.S} "
            f"({sum(n * r for n, r in zip(nv, self.crec)) / 2**30:.1f} GiB), RAM slots {self.R} "
            f"({sum(n * r for n, r in zip(nr, self.crec)) / 2**30:.1f} GiB), staging {n_stage}; {cl}")
        self.min_resident = 0
        self.check_vram_aliasing()

    def check_vram_aliasing(self):
        """xe has been seen to map two pages of a large device allocation to the same memory (Strata: a 22 GiB arena,
        so some cache slots held another expert's bytes). Tag every 64 KiB of the slot pools and read the tags back."""
        bad = 0
        for t in self._vram_t:
            flat = t.view(-1)
            n = flat.numel() // 65536
            if n == 0:
                continue
            v = flat[:n * 65536].view(n, 65536)
            tag = torch.arange(n, dtype=torch.int64, device=self.dev) * 2654435761 + 12345
            v[:, :8].copy_(tag.view(torch.uint8).view(n, 8))       # all tags written before any is read back
            back = v[:, :8].contiguous().view(torch.int64).view(-1)
            bad += int((back != tag).sum())
        torch.xpu.synchronize(self.dev)
        if bad:
            raise RuntimeError(f"VRAM slot pool: {bad} 64 KiB pages alias other pages (xe mapping bug); restart")
        log("VRAM slot pools: no aliased 64 KiB pages")

    def rec_k(self, key):
        return int(self.key_rec[key])

    # ------------------------------------------------------------------ helpers
    def key(self, li, e):
        return li * self.E + e

    def _read_piece(self, key, off, n, dst_np):
        """O_DIRECT read of bytes [off, off+n) of record `key` (off, n 4K-aligned) via a thread-local anonymous bounce,
        then copy into dst_np[off:off+n]."""
        tl = self._tl
        if not hasattr(tl, "buf"):
            tl.buf = mmap.mmap(-1, self.rec, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            tl.np = np.frombuffer(tl.buf, dtype=np.uint8)
        mv = memoryview(tl.buf)
        got = 0
        base = int(self.key_off[key]) + off
        t0 = time.perf_counter()
        while got < n:
            r = os.preadv(self.fd, [mv[got:n]], base + got)
            if r <= 0:
                raise IOError(f"short read of expert record {key}")
            got += r
        t1 = time.perf_counter()
        np.copyto(dst_np[off:off + n], tl.np[:n])
        st = self.stats                                 # reader-thread seconds (summed over threads)
        st["t_pread"] = st.get("t_pread", 0.0) + t1 - t0
        st["t_copy"] = st.get("t_copy", 0.0) + time.perf_counter() - t1

    def _pieces(self, rec):
        if not hasattr(self, "_piece_list"):
            self._piece_list = {}
        pl = self._piece_list.get(rec)
        if pl is None:
            pages = rec // 4096
            per = -(-pages // self.n_pieces)
            pl = self._piece_list[rec] = [(p * 4096, min(per, pages - p) * 4096) for p in range(0, pages, per)]
        return pl

    def _submit(self, key, dst_np, res_slot=-1, urgent=False):
        """Queue the piece reads of one record (into dst_np[:record bytes]); returns the list of futures. urgent: the
        caller waits on it now (native reader: ahead of queued prefetches)."""
        if self.native is not None:
            assert dst_np.flags["C_CONTIGUOUS"] and dst_np.nbytes >= self.rec_k(key)
            return [_NativeRead(self.native, self.native.submit(int(key), int(self.key_off[key]), self.rec_k(key),
                                                                 dst_np.ctypes.data, self.n_pieces, res_slot,
                                                                 urgent))]
        return [self.pool.submit(self._read_piece, key, off, n, dst_np) for off, n in self._pieces(self.rec_k(key))]

    def _submit_bg(self, k, rs, urgent=False):
        """Background read of expert k into RAM slot rs: registered in-flight, and only marked resident for router
        substitution once every piece has landed (a substitute must never make the step wait)."""
        self._set_res(k, -1e4)
        if self.native is not None:
            return self._submit(k, self.ram_np[rs], res_slot=rs, urgent=urgent)   # the reader marks it resident
        fs = self._submit(k, self.ram_np[rs])
        left = [len(fs)]

        def done(_f, k=k, rs=rs):
            left[0] -= 1
            if left[0] == 0 and self.ram_of[k] == rs:
                self._set_res(k, 0.0)
        for f in fs:
            f.add_done_callback(done)
        return fs

    def _read(self, key, dst_np):
        for f in self._submit(key, dst_np):
            f.result()

    def _reap_inflight(self):
        for k in [k for k, fs in self.inflight.items() if all(f.done() for f in fs)]:
            for f in self.inflight.pop(k):
                f.result()                           # surface read errors

    def _take_ram_slot(self, protect, c, wait=True):
        """A free RAM slot of class c, else the least recently used one whose expert is not in `protect`, not in
        flight and not being written back; -1 if none. Required reads can wait for an unprotected pending read;
        speculative prefetch and write-back pass wait=False so they stay off the critical path."""
        if self.free_ram[c]:
            return self.free_ram[c].pop()
        for attempt in (0, 1):
            lru = self.ram_lru[c]
            for k in lru:
                if k not in protect and k not in self.inflight and k not in self.wb and not self.flags[k] & 4:
                    rs = lru.pop(k)
                    self.ram_of[k] = -1
                    self.ram_key[rs] = -1
                    if self.vram_of[k] < 0:
                        self._set_res(k, -1e4)
                    return rs
            if attempt or not self.inflight:
                break
            # every candidate was in flight: the start-up RAM fill registers ~3700 reads that are otherwise only
            # reaped every 64 ensure() calls, and a prompt right after start found no slot
            self._reap_inflight()
            if wait:
                pending = next((k for k in lru if k not in protect and k in self.inflight and k not in self.wb), None)
                if pending is not None:
                    # A small size class can be completely occupied by prefetches that are still running. They
                    # cannot be overwritten yet, but waiting for one makes its slot safe to evict on the next pass.
                    for f in self.inflight[pending]:
                        f.result()
                    self.inflight.pop(pending)
        return -1

    def stage_ahead(self, nli, pred):
        """Copy layer nli's predicted, RAM-resident (landed) experts into its parity's staging rows on the copy stream.
        Their RAM slots are held (flags bit 4) until stage_take(nli) has seen the copy finish."""
        if not self.sa_on or self.stage_borrowed or self.need_refill:
            return
        par = nli & 1
        rows = self.sa_rows[par]
        base = nli * self.E
        m = {}
        keys = []
        for e in dict.fromkeys(pred.tolist()):          # rank order, best first
            k = base + e
            if self.vram_of[k] >= 0 or self.ram_of[k] < 0 or self.flags[k] & 7 or self.key_cls[k] != self.sa_cls:
                continue
            m[e] = rows[len(m)]
            keys.append(k)
            if len(m) == len(rows):
                break
        if not m:
            return
        if self.sa_cdone[par] is not None:
            self.sa_stream.wait_event(self.sa_cdone[par])     # the rows' previous reader has finished
        with torch.xpu.stream(self.sa_stream):
            for (e, row), k in zip(m.items(), keys):
                self.vrow[row].copy_(self.ram_t[int(self.ram_of[k])], non_blocking=True)
                self.flags[k] |= 4
            ev = torch.xpu.Event()
            ev.record(self.sa_stream)
        self.sa_map[nli] = (m, ev, par)
        self.stats["staged"] = self.stats.get("staged", 0) + len(m)

    def stage_take(self, li):
        """Start of layer li's ensure(): finish its staging copies (normally done: they ran during the previous layer),
        release the RAM slots, and return expert -> staging slot (or None)."""
        ent = self.sa_map.pop(li, None)
        if ent is None:
            return None
        m, ev, par = ent
        ev.synchronize()
        base = li * self.E
        for e in m:
            self.flags[base + e] &= 0xFF & ~4
        return m

    def stage_apply(self, li, m):
        """After ensure(li): picks that ended up read zero-copy from RAM read the staged VRAM copy instead. Returns
        whether a pointer changed (the caller uploads the row)."""
        base = li * self.E
        row = self.ptrs_h[li]
        changed = False
        n = 0
        for e, slot in m.items():
            k = base + e
            rs = self.ram_of[k]
            if self.vram_of[k] < 0 and rs >= 0 and row[e] == self.ram_addr[rs]:
                row[e] = self.vaddr[slot]
                changed = True
                n += 1
        if n:
            self.stats["staged_used"] = self.stats.get("staged_used", 0) + n
            self.sa_used = li
        return changed

    def stage_done(self, li):
        """After layer li's expert phase is submitted: its parity's rows may be refilled once it has run."""
        if getattr(self, "sa_used", -1) == li:
            ev = torch.xpu.Event()
            ev.record()
            self.sa_cdone[li & 1] = ev
            self.sa_used = -1

    def begin_demand(self, li, ids):
        """Right after a layer's routing sync, before the next layer's prefetches are queued: start the reads this
        layer's ensure() will wait on (urgent, so they go ahead of queued prefetches) and promote queued prefetches of
        its picks. ensure() then finds them in flight. Native reader only; skipped around a prompt's staging."""
        if self.native is None or self.stage_borrowed or self.need_refill:
            return
        base = li * self.E
        keys = {base + e for e in set(ids.ravel().tolist())}
        for k in keys:
            if self.vram_of[k] >= 0:
                continue
            fs = self.inflight.get(k)
            if fs is not None:
                for f in fs:
                    if isinstance(f, _NativeRead):
                        f.r.promote(f.id)
                continue
            if self.ram_of[k] >= 0:
                continue
            rs = self._take_ram_slot(keys, self.key_cls[k])
            if rs < 0:
                continue                         # ensure() handles it (and raises if there is no slot at all)
            self._ram_put(k, rs)
            self.inflight[k] = self._submit_bg(k, rs, urgent=True)
            self._demand.add(k)

    def _stage_reset(self):
        """Drop pending staging (its copies have run: called after a device sync)."""
        for li, (m, ev, par) in self.sa_map.items():
            ev.synchronize()
            for e in m:
                self.flags[li * self.E + e] &= 0xFF & ~4
        self.sa_map.clear()

    def _set_res(self, k, v):
        self.res_h.flat[k] = v
        self._bonus_k(k)

    def _bonus_k(self, k):
        """Expert k's router bonus from its current residency (VRAM bv, landed in RAM br, else 0); marks the upload."""
        if self.bonus_on:
            self.bonus_h[k] = self.bonus_v if self.vram_of[k] >= 0 else (self.bonus_r if self.res_h.flat[k] > -1.0
                                                                          else 0.0)
        self.res_dirty = True

    def sync_res(self):
        """Upload the residency bias if it changed (stream-ordered; call right before a router that reads it)."""
        if self._ndirty[0]:
            self._ndirty[0] = 0                 # cleared before the copy: a read landing during it raises it again
            self.res_dirty = True
        if self.res_dirty:
            if self.upload_res:                 # res_dev: only the legacy substitution router reads it
                self.res_dev.copy_(self.res_pin, non_blocking=True)
            if self.bonus_on:
                self.bonus_dev.copy_(self.bonus_pin, non_blocking=True)     # kept per expert by _bonus_k
            self.res_dirty = False

    def _ram_put(self, k, rs):
        self._set_res(k, 0.0)
        self.ram_of[k] = rs
        self.ram_key[rs] = k
        lru = self.ram_lru[self.rcls[rs]]
        lru[k] = rs
        lru.move_to_end(k)

    def _set_ptr(self, li, e, addr):
        self.ptrs_h[li, e] = addr

    # ------------------------------------------------------------------ decode / small batches
    def ensure(self, li: int, ids: np.ndarray):
        """Make every expert in `ids` (layer li) VRAM-resident; update the layer's pointer table."""
        if self.stage_borrowed:
            self._end_large_forward()
        if self.need_refill:
            self._refill()
        if self.inflight and self.tick % 64 == 0:
            self._reap_inflight()
        self._reap_wb()
        self.tick += 1
        t = self.tick
        if self.native is not None and t % 42 == 0:
            self.stats.update(self.native.stats())              # reader-thread seconds (t_pread, t_copy)
        sa = self.stage_take(li) if self.sa_map else None
        self._sa_cur = sa
        if self.core is not None:
            # VRAM hits, zero-copy RAM hits and the admission decisions in C++; the rest below in Python
            changed, slow, adm, n_vh, n_zc, n_fix, n_u = self.core.ensure_fast(
                li, ids.reshape(-1), [len(f) for f in self.free_vram], self.zero_copy,
                self.tick % self.freq_decay_every == 0)          # decay after the increments, before the decisions
            st = self.stats
            st["picks"] += n_u
            st["vram_hit"] += n_vh
            if n_fix:
                st["ptr_fix"] = st.get("ptr_fix", 0) + n_fix
            if n_zc:
                st["zero_copy"] = st.get("zero_copy", 0) + n_zc
                st["ram_hit"] += n_zc
            if not slow and not adm:
                if sa:
                    changed |= self.stage_apply(li, sa)
                if changed:
                    self.ptrs[li].copy_(self.ptrs_pin[li], non_blocking=True)
                return
            base = li * self.E
            protect = {base + e for e in set(ids.ravel().tolist())}
            self._ensure_slow(li, slow, adm, protect)
            return
        uniq = np.array(sorted(set(ids.ravel().tolist())), np.int64)   # (np.unique costs ~10 us on 8 picks)
        keys = uniq + li * self.E
        protect = set(keys.tolist())
        self.freq[keys] += 1.0
        self.hits[keys] += 1.0
        if self.tick % self.freq_decay_every == 0:
            self.freq *= 0.5
        misses = []
        stale = False
        for e, k in zip(uniq.tolist(), keys.tolist()):
            s = self.vram_of[k]
            if s >= 0:
                self.vram_lru[self.vcls[s]].move_to_end(k)
                self.stats["vram_hit"] += 1
                a = int(self.vaddr[s])
                if self.ptrs_h[li, e] != a:
                    # moved to VRAM without its pointer (a path that skips it would make the kernel read address 0,
                    # an xe CAT fault, or a freed RAM slot's new expert)
                    self.ptrs_h[li, e] = a
                    stale = True
                    self.stats["ptr_fix"] = self.stats.get("ptr_fix", 0) + 1
            else:
                misses.append((e, k))
        self.stats["picks"] += len(uniq)
        if not misses:
            if stale:
                self.ptrs[li].copy_(self.ptrs_pin[li], non_blocking=True)
            return
        self._ensure_slow(li, misses, [], protect)

    def _ensure_slow(self, li, misses, admit, protect):
        """ensure() for picks that are not plain VRAM / zero-copy RAM hits: `misses` (in flight, NVMe-only, write-back
        pending, or everything when the native step is off) are read if needed and then admitted or read zero-copy;
        `admit` are RAM hits the native step already chose for VRAM admission (they count against admit_per_call)."""
        # NVMe reads first (parallel), into RAM slots
        t0 = time.perf_counter()
        waits = []
        n_pf = 0
        n_dem = 0
        for e, k in misses:
            if k in self.inflight:
                fs = self.inflight.pop(k)
                for f in fs:
                    if isinstance(f, _NativeRead):
                        f.r.promote(f.id)          # a prefetch the step now waits on goes ahead of the others
                waits += fs
                if k in self._demand:              # a demand read begin_demand() started, not a prefetch hit
                    self._demand.discard(k)
                    n_dem += 1
                else:
                    n_pf += 1
        need_read = [(e, k) for e, k in misses if self.ram_of[k] < 0]
        for e, k in need_read:
            rs = self._take_ram_slot(protect, self.key_cls[k])
            if rs < 0:
                raise RuntimeError("RAM tier too small for one layer's picks")
            self._ram_put(k, rs)
            waits += self._submit(k, self.ram_np[rs], urgent=True)
        for f in waits:
            f.result()
        if waits:
            self.stats["t_nvme"] += time.perf_counter() - t0
        self.stats["nvme"] += len(need_read) + n_dem
        self.stats["prefetch_used"] = self.stats.get("prefetch_used", 0) + n_pf
        self.stats["ram_hit"] += len(misses) - len(need_read) - n_pf - n_dem
        # VRAM admission, or zero-copy: the MoE kernel reads a RAM-tier expert in place over PCIe (pinned USM host
        # memory is device-addressable), which moves the same bytes as an admission without the VRAM write, the victim
        # eviction and its write-back. An expert is admitted only when it has been picked often recently (decayed
        # count) and the LRU victim has been picked less, at most admit_per_call per layer call.
        t0 = time.perf_counter()
        n_adm = len(admit)
        self.stats["ram_hit"] += len(admit)
        if admit:
            self.stats["admit"] = self.stats.get("admit", 0) + len(admit)
        for e, k, chosen in [(e, k, True) for e, k in admit] + [(e, k, False) for e, k in misses]:
            c = self.key_cls[k]
            if self.zero_copy and not chosen:
                adm = False
                if n_adm < self.admit_per_call and self.freq[k] >= self.admit_min:
                    if self.free_vram[c]:
                        adm = True
                    else:
                        v = next((kk for kk in self.vram_lru[c] if kk not in protect), None)
                        adm = v is not None and self.freq[v] < self.freq[k]
                if not adm:
                    self._ram_ready(k)
                    self.ram_lru[c].move_to_end(k)
                    self._set_ptr(li, e, int(self.ram_addr[int(self.ram_of[k])]))
                    self.stats["zero_copy"] = self.stats.get("zero_copy", 0) + 1
                    continue
                n_adm += 1
                self.stats["admit"] = self.stats.get("admit", 0) + 1
            s = self._take_vram_slot(protect, c)
            rs = int(self.ram_of[k])
            self._ram_ready(k)
            sa = self._sa_cur
            e_ = k - li * self.E
            if sa and e_ in sa:
                self.vrow[s].copy_(self.vrow[sa[e_]], non_blocking=True)   # staged copy: VRAM to VRAM, no PCIe
            else:
                self.vrow[s].copy_(self.ram_t[rs], non_blocking=True)
            # exclusive: the RAM slot is free once the copy above has run (stream order covers GPU-side reuse; host
            # writers only reuse it after the next layer's routing sync)
            self.ram_of[k] = -1
            self.ram_key[rs] = -1
            self.ram_lru[c].pop(k, None)
            self.free_ram[c].append(rs)
            self.vram_of[k] = s
            self.vram_key[s] = k
            self.vram_lru[c][k] = s
            self._set_res(k, 0.0)
            self._set_ptr(li, e, int(self.vaddr[s]))
        self.stats["t_h2d"] += time.perf_counter() - t0
        if self._sa_cur:
            self.stage_apply(li, self._sa_cur)
        self.ptrs[li].copy_(self.ptrs_pin[li], non_blocking=True)

    def prefetch(self, li: int, ids: np.ndarray, protect=()):
        """Start NVMe reads (into RAM slots) for predicted picks of layer li that are in neither VRAM nor RAM. The next
        ensure() of that layer waits on them; nothing waits here."""
        uniq = np.unique(ids)
        n = 0
        for e in uniq.tolist():
            k = li * self.E + e
            if self.vram_of[k] >= 0 or self.ram_of[k] >= 0 or k in self.inflight:
                continue
            rs = self._take_ram_slot(set(protect), self.key_cls[k], wait=False)
            if rs < 0:
                continue
            self._ram_put(k, rs)
            self.inflight[k] = self._submit_bg(k, rs)
            n += 1
        self.stats["prefetch"] = self.stats.get("prefetch", 0) + n

    # ------------------------------------------------------------------ no-sync decode
    # The decode step runs without any host sync inside the token: the router picks only experts whose row in the
    # pointer table is non-zero (VRAM slot, or RAM-tier address read zero-copy), so a pick that is only on NVMe is
    # replaced by the next-best resident expert. Once per token (at the start of the next step, when the GPU is idle
    # anyway) step_nosync() takes the previous token's exact picks, starts NVMe reads for the missed ones, admits hot
    # RAM experts to VRAM, and uploads the whole pointer table.

    def refresh_ptrs(self):
        """Rebuild every pointer row: VRAM slot address, else RAM-tier address once its bytes have landed, else 0.
        The upload is stream-ordered; the host mirror is only rewritten after a sync (step start)."""
        vo, ro = self.vram_of, self.ram_of
        t = np.zeros(vo.shape, np.int64)
        inv = vo >= 0
        t[inv] = self.vaddr[vo[inv]]
        inr = ~inv & (ro >= 0)
        t[inr] = self.ram_addr[ro[inr]]
        for k in list(self.inflight) + list(self.wb):
            if vo[k] < 0:
                t[k] = 0
        tl = t.reshape(self.L, self.E)
        self.ptrs_h[...] = tl
        self.min_resident = int((tl[:self.n_moe] != 0).sum(1).min())
        self.ptrs.copy_(self.ptrs_pin, non_blocking=True)
        self.vram_pin.numpy()[...] = inv.reshape(self.L, self.E)
        self.vram_dev.copy_(self.vram_pin, non_blocking=True)

    def _admit(self, k, protect):
        """RAM-tier expert k -> a VRAM slot (LRU victim of its class written back to RAM)."""
        c = self.key_cls[k]
        s = self._take_vram_slot(protect, c)
        rs = int(self.ram_of[k])
        self._ram_ready(k)
        self.vrow[s].copy_(self.ram_t[rs], non_blocking=True)
        self.ram_of[k] = -1
        self.ram_key[rs] = -1
        self.ram_lru[c].pop(k, None)
        self.free_ram[c].append(rs)
        self.vram_of[k] = s
        self.vram_key[s] = k
        self.vram_lru[c][k] = s
        li, e = divmod(k, self.E)
        self.ptrs_h[li, e] = int(self.vaddr[s])     # ensure() trusts a VRAM hit's pointer row
        self._bonus_k(k)                            # the VRAM bonus moves with it

    def _evict_vram(self, v, protect=()):
        """Evict VRAM-resident expert v (written back to RAM when a slot is free there); its slot joins the free list."""
        s = int(self.vram_of[v])
        c = self.vcls[s]
        self.vram_lru[c].pop(v, None)
        li, e = divmod(v, self.E)
        self._write_back(v, s, set(protect))
        self.ptrs_h[li, e] = 0
        self.vram_of[v] = -1
        self.vram_key[s] = -1
        if self.ram_of[v] < 0:
            self._set_res(v, -1e4)
        else:
            self._bonus_k(v)
        self.free_vram[c].append(s)

    def curate(self, stop=lambda: False, max_moves=None, margin=1.0):
        """Idle-time cache curation (between requests, GPU idle): VRAM takes the RAM experts used most recently (decayed
        pick counts, freq) in place of the least-used VRAM ones, and RAM reads the most-used NVMe-only experts in place
        of its least-used ones. Decode admits at most admit_per_call per layer, so after a turn the cache lags what the
        conversation used. A move needs freq[candidate] > freq[victim] + margin. stop() is polled between moves (a new
        request). Returns (VRAM moves, NVMe reads started)."""
        if self.stage_borrowed:
            self._end_large_forward()
        if self.need_refill:
            self._refill()
        self._reap_inflight()
        self._reap_wb()
        max_moves = max_moves or int(os.environ.get("GLM_XPU_CURATE_MAX", "512"))
        t0 = time.perf_counter()
        LEm = self.n_moe * self.E
        f = self.freq[:LEm]
        vo, ro, fl, kc = self.vram_of[:LEm], self.ram_of[:LEm], self.flags[:LEm], self.key_cls[:LEm]
        n_v = n_r = 0
        for c in range(self.C):
            # RAM -> VRAM
            cand = np.nonzero((ro >= 0) & (vo < 0) & (kc == c) & (fl == 0))[0]
            cand = cand[np.argsort(-f[cand], kind="stable")]
            vict = np.nonzero((vo >= 0) & (kc == c))[0]
            vict = vict[np.argsort(f[vict], kind="stable")]
            vi = 0
            for k in cand.tolist():
                if stop() or n_v >= max_moves or f[k] <= 0 or self.flags[k] or self.ram_of[k] < 0:
                    break
                if not self.free_vram[c]:
                    while vi < len(vict) and self.vram_of[vict[vi]] < 0:
                        vi += 1
                    if vi >= len(vict) or f[k] <= f[vict[vi]] + margin:
                        break                        # sorted: no better pair left
                    self._evict_vram(int(vict[vi]), protect=(k,))
                    vi += 1
                if self.ram_of[k] < 0:               # its RAM slot went to the write-back
                    continue
                self._admit(int(k), set())
                n_v += 1
            # NVMe -> RAM
            cand = np.nonzero((vo < 0) & (ro < 0) & (kc == c) & (fl == 0))[0]
            cand = cand[np.argsort(-f[cand], kind="stable")]
            vict = np.nonzero((ro >= 0) & (vo < 0) & (kc == c) & (fl == 0))[0]
            vict = vict[np.argsort(f[vict], kind="stable")]
            vi = 0
            for k in cand.tolist():
                if stop() or n_r >= max_moves or f[k] <= 0:
                    break
                if self.free_ram[c]:
                    rs = self.free_ram[c].pop()
                else:
                    while vi < len(vict) and (self.ram_of[vict[vi]] < 0 or self.flags[vict[vi]] or
                                              self.vram_of[vict[vi]] >= 0):
                        vi += 1
                    if vi >= len(vict) or f[k] <= f[vict[vi]] + margin:
                        break
                    v = int(vict[vi])
                    vi += 1
                    rs = int(self.ram_of[v])
                    self.ram_lru[self.rcls[rs]].pop(v, None)
                    self.ram_of[v] = -1
                    self.ram_key[rs] = -1
                    self._set_res(v, -1e4)
                self._ram_put(int(k), rs)
                self.inflight[int(k)] = self._submit_bg(int(k), rs)
                n_r += 1
        self.stats["curated_vram"] = self.stats.get("curated_vram", 0) + n_v
        self.stats["curated_nvme"] = self.stats.get("curated_nvme", 0) + n_r
        if n_v or n_r:
            log(f"curate: {n_v} RAM -> VRAM, {n_r} NVMe -> RAM reads started, {time.perf_counter() - t0:.2f} s")
        return n_v, n_r

    def step_nosync(self, picks, admit_per_token: int = 8):
        """Per-token bookkeeping for no-sync decode. picks: int64 [n_moe, k] exact router picks of the previous token
        (None after a prefill). Call with the GPU idle (the previous token's work complete)."""
        if self.stage_borrowed:
            self._end_large_forward()
        for k in [k for k, fs in self.inflight.items() if all(f.done() for f in fs)]:
            for f in self.inflight.pop(k):
                f.result()                       # surface read errors
        self._reap_wb()
        if picks is not None:
            self.tick += 1
            keys = np.unique((picks + (np.arange(picks.shape[0]) * self.E)[:, None]).ravel())
            protect = set(keys.tolist())
            self.freq[keys] += 1.0
            self.hits[keys] += 1.0
            if self.tick % 32 == 0:
                self.freq *= 0.5
            self.stats["picks"] += len(keys)
            busy = self.inflight.keys() | self.wb.keys()
            cand = []
            for k in keys.tolist():
                s = self.vram_of[k]
                if s >= 0:
                    self.vram_lru[self.vcls[s]].move_to_end(k)
                    self.stats["vram_hit"] += 1
                    continue
                rs = self.ram_of[k]
                if rs >= 0:
                    self.ram_lru[self.rcls[rs]].move_to_end(k)
                    if k in self.inflight:
                        self.stats["nosync_miss"] = self.stats.get("nosync_miss", 0) + 1
                        self.stats["nosync_inflight"] = self.stats.get("nosync_inflight", 0) + 1
                    else:
                        self.stats["ram_hit"] += 1
                        if self.freq[k] >= self.admit_min:
                            cand.append(k)
                    continue
                self.stats["nosync_miss"] = self.stats.get("nosync_miss", 0) + 1
                rs = self._take_ram_slot(protect, self.key_cls[k], wait=False)
                if rs < 0:
                    self.stats["nosync_noslot"] = self.stats.get("nosync_noslot", 0) + 1   # not loaded at all
                    continue
                self._ram_put(k, rs)
                self.inflight[k] = self._submit_bg(k, rs)
                self.stats["nvme"] += 1
            # admissions after the reads above took their RAM slots: a slot freed here is only reused after the next
            # step-start sync, by which time its RAM -> VRAM copy has run
            cand.sort(key=lambda k: -self.freq[k])
            n = 0
            for k in cand:
                if n >= admit_per_token:
                    break
                c = self.key_cls[k]
                if not self.free_vram[c]:
                    v = next((kk for kk in self.vram_lru[c] if kk not in protect), None)
                    if v is None or self.freq[v] >= self.freq[k]:
                        continue
                self._admit(k, protect)
                n += 1
            self.stats["admit"] = self.stats.get("admit", 0) + n
        if self.need_refill:
            self._refill()                       # after this step's reads took their RAM slots (see above)
        self.refresh_ptrs()

    # ------------------------------------------------------------------ staging slots borrowed from the VRAM pool
    def _borrow_stage(self):
        """Take the staging slots out of the VRAM pool for a large forward; their experts are dropped (still in RAM if
        they were there, else only on NVMe)."""
        for s in self.stage_slots:
            k = int(self.vram_key[s])
            if k >= 0:
                li, e = divmod(k, self.E)
                self.vram_lru[self.vcls[s]].pop(k, None)
                self.vram_of[k] = -1
                self.vram_key[s] = -1
                self.ptrs_h[li, e] = 0
                if self.ram_of[k] < 0:
                    self._set_res(k, -1e4)
                else:
                    self._bonus_k(k)
        st = set(self.stage_slots)
        fl = self.free_vram[self.stage_cls]
        fl[:] = [s for s in fl if s not in st]
        self.stage_borrowed = True

    def _end_large_forward(self):
        """First small call after a large forward. A prefill stops after the last layer's attention (exllamav3
        prefill_ls), so the last MoE layer never runs and cannot end the stream itself."""
        torch.xpu.synchronize(self.dev)
        self._stage_reset()
        self._pf_stop()
        self._return_stage()
        # the prompt's activations stay reserved in torch's caching allocator; the Level Zero runtime needs free
        # device memory of its own (a graph replay after a 4K-token prompt failed with OUT_OF_DEVICE_MEMORY)
        torch.xpu.empty_cache()

    def _return_stage(self):
        """Large forward done (its MoE calls have synchronised): staging slots go back to the pool, refilled at the
        next decode step."""
        if self.stage_borrowed:
            self.free_vram[self.stage_cls].extend(self.stage_slots)
            self.stage_borrowed = False
            self.need_refill = True

    def _refill(self):
        """Fill the free slots of the staging class with the hottest experts (warm order) not in VRAM: from RAM by
        admission, else from NVMe through the bounce buffers. Ends with a device sync, so the RAM slots it frees can be
        reused at once."""
        self.need_refill = False
        c = self.stage_cls
        n_free = len(self.free_vram[c])
        if not n_free:
            return
        t0 = time.perf_counter()
        busy = self.inflight.keys() | self.wb.keys()
        keys = []
        for li, e in self.warm_order:
            k = li * self.E + e
            if self.key_cls[k] == c and self.vram_of[k] < 0 and k not in busy:
                keys.append(k)
                if len(keys) >= n_free:
                    break
        n_ram = 0
        for k in keys:
            if self.ram_of[k] >= 0:
                self._admit(k, set())
                self._set_res(k, 0.0)
                n_ram += 1
        # NVMe-only experts are left to admission: reading them here held the first decode step ~0.5 s after every
        # prompt (GLM_XPU_REFILL_NVME=1 restores it)
        nvme = [k for k in keys if self.vram_of[k] < 0] if self.refill_nvme else []
        nb = self.n_bounce
        for c0 in range(0, len(nvme), nb):
            chunk = nvme[c0:c0 + nb]
            torch.xpu.synchronize(self.dev)                   # the bounce rows are free again
            for f in [f for j, k in enumerate(chunk) for f in self._submit(k, self.bounce_np[j])]:
                f.result()
            for j, k in enumerate(chunk):
                s = self.free_vram[c].pop()
                self.vrow[s].copy_(self.bounce_t[j][:self.rec_k(k)], non_blocking=True)
                self.vram_of[k] = s
                self.vram_key[s] = k
                self.vram_lru[c][k] = s
                self._set_res(k, 0.0)
                li, e = divmod(k, self.E)
                self.ptrs_h[li, e] = int(self.vaddr[s])
        self.ptrs.copy_(self.ptrs_pin, non_blocking=True)
        torch.xpu.synchronize(self.dev)
        self.stats["refill"] = self.stats.get("refill", 0) + len(keys)
        log(f"refilled {len(keys)} staging slots ({n_ram} from RAM, {len(nvme)} from NVMe) in "
            f"{time.perf_counter() - t0:.2f} s")

    def _take_vram_slot(self, protect, c):
        if self.free_vram[c]:
            return self.free_vram[c].pop()
        lru = self.vram_lru[c]
        for k in lru:
            if k in protect:
                continue
            s = lru.pop(k)
            li, e = divmod(k, self.E)
            self._write_back(k, s, protect)
            self.ptrs_h[li, e] = 0          # device rows of other layers go stale; only picked rows are ever read
            self.vram_of[k] = -1
            self.vram_key[s] = -1
            if self.ram_of[k] < 0:
                self._set_res(k, -1e4)
            else:
                self._bonus_k(k)            # its VRAM bonus goes (RAM bonus if the write-back kept it)
            return s
        raise RuntimeError("VRAM expert cache too small for one layer's picks")

    def _write_back(self, k, s, protect):
        """Victim k in VRAM slot s -> victim ring (device copy now) -> a RAM slot (side stream)."""
        if not self.R:
            return
        if not self.vring_free:
            self._reap_wb()
            if not self.vring_free:
                self.stats["wb_dropped"] = self.stats.get("wb_dropped", 0) + 1
                return
        rs = self._take_ram_slot(set(protect), self.key_cls[k], wait=False)
        if rs < 0:
            return
        j = self.vring_free.pop()
        rec = self.rec_k(k)
        # an elementwise kernel, not copy_: a device-to-device copy_ goes to a copy engine and blocks the host ~100 us
        torch.bitwise_or(self.vrow64[s], 0, out=self.vring64[j][:rec // 8])
        ev = torch.xpu.Event()
        ev.record()
        with torch.xpu.stream(self.side):
            self.side.wait_event(ev)
            self.ram_t[rs].copy_(self.vring[j][:rec], non_blocking=True)
            ev2 = torch.xpu.Event()
            ev2.record(self.side)
        self._ram_put(k, rs)
        self.wb[k] = (ev2, j)
        self.stats["writeback"] += 1

    def _reap_wb(self):
        for k in [k for k, (ev, _) in self.wb.items() if ev.query()]:
            self.vring_free.append(self.wb.pop(k)[1])

    def _ram_ready(self, k):
        """Make the compute stream wait for k's RAM write-back if it has not landed yet."""
        w = self.wb.get(k)
        if w is not None:
            torch.xpu.current_stream(self.dev).wait_event(w[0])

    # ------------------------------------------------------------------ prefill
    # ------------------------------------------------------------------ prefill streaming
    # A reader thread works ahead through the MoE layers of a large forward, reading every expert that is in neither
    # VRAM nor RAM into a pinned host ring (prompt-sized forwards route to nearly all 288 experts of every layer).
    # stage_layer() then only issues host -> staging copies. The ring bounds the read-ahead; a layer's ring slots are
    # released after its MoE call has synchronised.

    def _pf_init(self, n_slots):
        import queue
        import threading
        self.pf_n = n_slots
        self._pf_chunks, self.pf_t, self.pf_np = [], [], []
        left = n_slots
        while left > 0:
            n = min(226, left)
            c = torch.empty((n, self.rec), dtype=torch.uint8, pin_memory=True)
            assert c.is_pinned() and c.data_ptr() != 0
            self._pf_chunks.append(c)
            cn = c.numpy()
            self.pf_t.extend(c[i] for i in range(n))
            self.pf_np.extend(cn[i] for i in range(n))
            left -= n
        # Touch every page now (as the RAM tier): the first long prompt otherwise faulted in these ~5 GB of pinned
        # pages from the reader threads' copies, 92 thread-seconds of copy instead of 29 (4.9k-token prompt 70 s
        # instead of 29 s, tests/startbench.py --prefill, 2026-10-30)
        import concurrent.futures as _cf
        t0 = time.perf_counter()
        with _cf.ThreadPoolExecutor(len(self._pf_chunks) + 1) as ex:
            list(ex.map(lambda c: c.numpy().fill(0), self._pf_chunks + [self.bounce_t]))
        log(f"prompt read-ahead ring: {sum(c.numel() for c in self._pf_chunks) / 2**30:.1f} GiB pinned and touched in "
            f"{time.perf_counter() - t0:.1f} s")
        self.pf_free = queue.Queue()
        for i in range(n_slots):
            self.pf_free.put(i)
        self.pf_lock = threading.Lock()
        self.pf_map = {}            # key -> (ring slot, futures)
        self.pf_layer = {}          # li -> list of keys read for it
        self.pf_epoch = 0
        self.pf_thread = None
        self.pf_next = -1           # layer the stream is expected to serve next

    def _pf_start(self, li):
        import threading
        self._pf_stop()
        self.pf_epoch += 1
        ep = self.pf_epoch
        self.pf_next = li

        def run():
            for l in range(li, self.n_moe):
                if self.pf_epoch != ep:
                    return
                keys = []
                for e in range(self.E):
                    k = l * self.E + e
                    if self.vram_of[k] >= 0 or self.ram_of[k] >= 0:
                        continue
                    slot = self.pf_free.get()
                    if slot < 0 or self.pf_epoch != ep:
                        if slot >= 0:
                            self.pf_free.put(slot)
                        return
                    with self.pf_lock:
                        self.pf_map[k] = (slot, self._submit(k, self.pf_np[slot]))
                    keys.append(k)
                with self.pf_lock:
                    self.pf_layer[l] = keys
        self.pf_thread = threading.Thread(target=run, daemon=True)
        self.pf_thread.start()

    def _pf_release(self, li):
        with self.pf_lock:
            keys = self.pf_layer.pop(li, [])
            items = [self.pf_map.pop(k) for k in keys if k in self.pf_map]
        for slot, futs in items:
            for f in futs:
                f.result()
            self.pf_free.put(slot)

    def _pf_stop(self):
        if self.pf_thread is None:
            return
        self.pf_epoch += 1
        for _ in range(4):                          # unblock a reader waiting for a ring slot
            self.pf_free.put(-1)
        self.pf_thread.join()
        with self.pf_free.mutex:                    # drop the unused sentinels again
            q = self.pf_free.queue
            for _ in range(len(q)):
                v = q.popleft()
                if v >= 0:
                    q.append(v)
        for li in list(self.pf_layer):
            self._pf_release(li)
        with self.pf_lock:
            rest = list(self.pf_map.values())
            self.pf_map.clear()
        for slot, futs in rest:
            for f in futs:
                f.result()
            self.pf_free.put(slot)
        self.pf_thread = None

    def stage_layer(self, li: int, ids: np.ndarray) -> torch.Tensor:
        """Pointer table for a large forward: VRAM-resident experts in place, others copied into staging."""
        if self.pf_next != li:
            if not self.stage_borrowed:
                self._borrow_stage()                    # before the reader decides what is resident
            # The stream reads every non-resident expert of the layers ahead (their routing is not known yet): right
            # for long prompts, which use nearly all of them, but a short one (a chat follow-up) needs ~10 per layer
            # and waited on ~125 per layer of reads. Short prompts read just their picks, in parallel, per layer
            self.pf_demand = ids.size < self.pf_min_picks
            if self.pf_demand:
                self._pf_stop()
            else:
                self._pf_start(li)                      # a new forward (or an out-of-order layer): restart the stream
        self.pf_next = li + 1
        uniq = np.unique(ids)
        table = self.ptrs_h[li].copy()
        stage, row = self.stage, self.stage_row
        if int(self.rec_of[li]) > row:                  # a record larger than the staging rows (MTP layer, K=4)
            row = int(self.rec_of[li])
            # kept on self: the kernel reads it after we return, and a freed block would be handed to the next
            # allocation on the queue before that read
            big = self.__dict__.get("_big_stage")
            if big is None or big.shape[0] < len(uniq) or big.shape[1] != row:
                big = self._big_stage = torch.empty((max(len(uniq), self.n_stage), row), dtype=torch.uint8,
                                                    device=self.dev)
            stage = big
        stage_base = stage.data_ptr()
        need = []
        nxt = 0
        t0 = time.perf_counter()
        n_nvme = 0
        for e in uniq.tolist():
            k = li * self.E + e
            s = self.vram_of[k]
            if s >= 0:
                self.vram_lru[self.vcls[s]].move_to_end(k)
                continue
            j = nxt
            nxt += 1
            rec = self.rec_k(k)
            rs = self.ram_of[k]
            if rs >= 0:
                self.ram_lru[self.rcls[rs]].move_to_end(k)
                if k in self.inflight:
                    for f in self.inflight.pop(k):
                        f.result()
                self._ram_ready(k)
                stage[j][:rec].copy_(self.ram_t[rs], non_blocking=True)
            else:
                while True:
                    with self.pf_lock:
                        got = self.pf_map.get(k)
                    if got is not None or self.pf_thread is None or not self.pf_thread.is_alive():
                        break
                    time.sleep(0.0002)
                if got is None:                         # not streamed (demand mode, or resident when the reader
                    need.append((k, j, rec))            # passed): read below, in parallel
                else:
                    for f in got[1]:
                        f.result()
                    stage[j][:rec].copy_(self.pf_t[got[0]][:rec], non_blocking=True)
                n_nvme += 1
            table[e] = self.s64(stage_base + j * row)
        nb = self.n_bounce
        for c0 in range(0, len(need), nb):
            chunk = need[c0:c0 + nb]
            if c0:
                torch.xpu.synchronize(self.dev)         # the bounce rows of the previous batch are free again
            for f in [f for b, (k, j, rec) in enumerate(chunk) for f in self._submit(k, self.bounce_np[b])]:
                f.result()
            for b, (k, j, rec) in enumerate(chunk):
                stage[j][:rec].copy_(self.bounce_t[b][:rec], non_blocking=True)
        self.stats["prefill_stage"] += nxt
        self.stats["prefill_nvme"] += n_nvme
        self.stats["t_nvme"] += time.perf_counter() - t0
        return torch.from_numpy(table).to(self.dev)

    # ------------------------------------------------------------------ forward
    def moe(self, li: int, x: torch.Tensor, sel: torch.Tensor, w: torch.Tensor, ids_np=None) -> torch.Tensor:
        """Routed-expert sum for layer li. x fp16 [M, H], sel int64 [M, k], w fp16 [M, k] -> fp16 [M, H]."""
        self.stats["calls"] += 1
        if ids_np is None:
            ids_np = sel.cpu().numpy()
        n_picks = ids_np.size
        ids = sel.to(torch.int32).contiguous()
        wf = w.to(torch.float32).contiguous()
        # ensure() needs a RAM slot of the layer's class for every pick not in VRAM at once; a small class (sc_2.60:
        # the one 3.5-bit layer has 44 RAM slots, and its VRAM pool is the prompt staging area, emptied by the last
        # long prompt) cannot take a short prompt's ~100 distinct picks, which raised "RAM tier too small". Such a call
        # goes through the staging path, which takes any number of experts.
        large = n_picks > self.admit_max
        if not large:
            base = li * self.E
            miss = sum(1 for e in set(ids_np.ravel().tolist()) if self.vram_of[base + e] < 0)
            large = miss > self.n_ram_cls[self.key_cls[base]] // 2
        if not large:
            self.ensure(li, ids_np)
            ptrs = self.ptrs[li]
        else:
            ptrs = self.stage_layer(li, ids_np)
        out = self.X.moe_forward(x.contiguous(), ids, wf, ptrs, I, self.K_of[li], self.E)
        if large:
            torch.xpu.synchronize(self.dev)     # staging is reused by the next layer
            self._pf_release(li)                # its ring slots are free once the copies above have run
            if li >= self.n_moe - 1:
                self._pf_stop()
                self._return_stage()
        return out

    def warm(self, order):
        """Fill VRAM, then RAM, with experts in `order` (list of (li, e), hottest first). Parallel reads."""
        t0 = time.perf_counter()
        self.warm_order = list(order)
        vram_jobs, ram_jobs = [], []
        nv = [len(f) for f in self.free_vram]
        nr = [len(f) for f in self.free_ram]
        for li, e in order:
            k = li * self.E + e
            c = self.key_cls[k]
            if nv[c] > 0:
                vram_jobs.append((li, e, k))
                nv[c] -= 1
            elif nr[c] > 0:
                ram_jobs.append(k)
                nr[c] -= 1
        nb = self.n_bounce
        for c0 in range(0, len(vram_jobs), nb):
            chunk = vram_jobs[c0:c0 + nb]
            torch.xpu.synchronize(self.dev)
            for f in [f for j, (_, _, k) in enumerate(chunk) for f in self._submit(k, self.bounce_np[j])]:
                f.result()
            for j, (li, e, k) in enumerate(chunk):
                c = self.key_cls[k]
                s = self.free_vram[c].pop()
                self.vrow[s].copy_(self.bounce_t[j][:self.rec_k(k)], non_blocking=True)
                self.vram_of[k] = s
                self.vram_key[s] = k
                self.vram_lru[c][k] = s
                self._set_res(k, 0.0)
                self.ptrs_h[li, e] = int(self.vaddr[s])
        slots = [self.free_ram[self.key_cls[k]].pop() for k in ram_jobs]
        # RAM fill, waited for here: with the tier pre-touched it reads ~7 GB/s (13 GB in ~2 s), but run in the
        # background under the launcher's warm-up request the two slowed each other (the request's reads queued behind
        # ~10k fill pieces, each needing the GIL the request's thread held): fill and request both took ~45 s
        # (tests/startbench.py, 2026-10-29). _submit_bg still marks each expert resident as it lands.
        t1 = time.perf_counter()
        fs = []
        for k, rs in zip(ram_jobs, slots):
            self._ram_put(k, rs)
            fs += self._submit_bg(k, rs)
        for f in fs:
            f.result()
        torch.xpu.synchronize(self.dev)
        self.ptrs.copy_(torch.from_numpy(self.ptrs_h))
        log(f"warm: {len(vram_jobs)} VRAM experts in {t1 - t0:.1f} s, {len(ram_jobs)} RAM experts in "
            f"{time.perf_counter() - t1:.1f} s")

    def save_profile(self, path):
        """Add this session's decode pick counts to the routing profile at `path` (.npy, [L * E] float64)."""
        if not self.hits.any():
            return
        tot = np.load(path) if os.path.exists(path) else np.zeros_like(self.hits)
        if tot.shape != self.hits.shape:
            tot = np.zeros_like(self.hits)
        np.save(path + ".tmp.npy", tot + self.hits)
        os.replace(path + ".tmp.npy", path)
        self.hits[:] = 0

    def summary(self):
        s = dict(self.stats)
        s["vram_resident"] = int((self.vram_key >= 0).sum())
        s["ram_resident"] = int((self.ram_key >= 0).sum())
        return s
