"""CPU-only equivalence tests of exl3xpu.tier_native (no GPU, no torch):
  - Lru vs collections.OrderedDict under random operations (incl. popping the current key while iterating);
  - Core.ensure_fast vs the Python ensure() logic it replaces, on random cache states.
  in_rootfs.sh python3 /w/xpu/test_native_core.py
"""
import random
import sys
import unittest
from collections import OrderedDict

import numpy as np

sys.path.insert(0, "/w/xpu")
from exl3xpu import tier_native as tn


class LruTest(unittest.TestCase):
    def test_matches_ordereddict(self):
        rnd = random.Random(7)
        n = 64
        a, b = tn.Lru(n), OrderedDict()
        for step in range(20000):
            op = rnd.randrange(7)
            k = rnd.randrange(n)
            if op == 0:
                v = rnd.randrange(1000)
                a[k] = v; b[k] = v
            elif op == 1 and k in b:
                a.move_to_end(k); b.move_to_end(k)
            elif op == 2:
                self.assertEqual(a.pop(k, None), b.pop(k, None))
            elif op == 3 and k in b:
                self.assertEqual(a.pop(k), b.pop(k))
            elif op == 4:
                # eviction scan: first key passing a filter is popped, iteration stops
                skip = {rnd.randrange(n) for _ in range(3)}
                ka = next((x for x in a if x not in skip), None)
                kb = next((x for x in b if x not in skip), None)
                self.assertEqual(ka, kb)
                if ka is not None:
                    self.assertEqual(a.pop(ka), b.pop(kb))
            elif op == 5:
                # pop the current key and keep iterating
                la, lb = [], []
                for x in a:
                    la.append(x)
                    if x % 3 == 0:
                        a.pop(x)
                for x in list(b):
                    lb.append(x)
                    if x % 3 == 0:
                        b.pop(x)
                self.assertEqual(la, lb)
            self.assertEqual(len(a), len(b))
            self.assertEqual(bool(a), bool(b))
            if step % 500 == 0:
                self.assertEqual(list(a), list(b))
                for x in b:
                    self.assertIn(x, a)
                    self.assertEqual(a[x], b[x])
        self.assertEqual(list(a), list(b))
        with self.assertRaises(KeyError):
            a.pop(n - 1) if (n - 1) not in b else a.pop(next(x for x in range(n) if x not in b))
        with self.assertRaises(IndexError):
            a[n] = 1


def make_state(rnd, L=4, E=32, S=40, R=30, C=2):
    LE = L * E
    key_cls = np.array([(k // E) % C for k in range(LE)], np.int64)
    vcls = np.array([s % C for s in range(S)], np.int64)
    rcls = np.array([s % C for s in range(R)], np.int64)
    vram_of = np.full(LE, -1, np.int64); ram_of = np.full(LE, -1, np.int64)
    vram_key = np.full(S, -1, np.int64); ram_key = np.full(R, -1, np.int64)
    keys = list(range(LE)); rnd.shuffle(keys)
    vfree = [[s for s in range(S) if vcls[s] == c] for c in range(C)]
    rfree = [[s for s in range(R) if rcls[s] == c] for c in range(C)]
    for c in range(C):
        rnd.shuffle(vfree[c]); rnd.shuffle(rfree[c])
    for k in keys:
        c = key_cls[k]
        x = rnd.random()
        if x < 0.35 and len(vfree[c]) > rnd.randrange(3):
            s = vfree[c].pop(); vram_of[k] = s; vram_key[s] = k
        elif x < 0.6 and rfree[c]:
            s = rfree[c].pop(); ram_of[k] = s; ram_key[s] = k
    flags = np.zeros(LE, np.uint8)
    for k in range(LE):
        if ram_of[k] >= 0 and rnd.random() < 0.15:
            flags[k] = rnd.choice([1, 2, 3])
    freq = np.array([rnd.choice([0.0, 1.0, 3.0, 5.0, 8.0, 12.0]) for _ in range(LE)], np.float32)
    hits = np.zeros(LE, np.float64)
    ptrs_h = np.zeros((L, E), np.int64)
    vaddr = np.arange(S, dtype=np.int64) * 1000 + 10 ** 9
    ram_addr = np.arange(R, dtype=np.int64) * 1000 + 2 * 10 ** 9
    for k in range(LE):
        li, e = divmod(k, E)
        if vram_of[k] >= 0:
            ptrs_h[li, e] = vaddr[vram_of[k]] if rnd.random() < 0.9 else 0      # some stale rows
        elif ram_of[k] >= 0:
            ptrs_h[li, e] = ram_addr[ram_of[k]] if rnd.random() < 0.5 else 0
    order_v = [k for k in keys if vram_of[k] >= 0]; rnd.shuffle(order_v)
    order_r = [k for k in keys if ram_of[k] >= 0]; rnd.shuffle(order_r)
    return dict(L=L, E=E, C=C, key_cls=key_cls, vcls=vcls, rcls=rcls, vram_of=vram_of, ram_of=ram_of, flags=flags,
                freq=freq, hits=hits, ptrs_h=ptrs_h, vaddr=vaddr, ram_addr=ram_addr, order_v=order_v, order_r=order_r,
                vfree=vfree)


def py_ensure(st, li, ids, admit_min=4.0, admit_per_call=2, decay=False):
    """The Python ensure() common path, as tier.py had it, on plain containers. Returns the same tuple as ensure_fast
    plus the LRUs."""
    E = st["E"]
    vlru = [OrderedDict() for _ in range(st["C"])]; rlru = [OrderedDict() for _ in range(st["C"])]
    for k in st["order_v"]:
        vlru[st["vcls"][st["vram_of"][k]]][k] = st["vram_of"][k]
    for k in st["order_r"]:
        rlru[st["rcls"][st["ram_of"][k]]][k] = st["ram_of"][k]
    uniq = sorted(set(ids.tolist()))
    keys = [li * E + e for e in uniq]
    protect = set(keys)
    for k in keys:
        st["freq"][k] += 1.0; st["hits"][k] += 1.0
    if decay:
        st["freq"] *= np.float32(0.5)
    changed, n_vh, n_zc, n_fix, n_adm = False, 0, 0, 0, 0
    # admissions are sequential: each one without a free slot evicts the first unprotected VRAM LRU entry
    # (_take_vram_slot), so later candidates compare against the next one; `sim` tracks that without touching vlru
    consumed = set()
    slow, admit = [], []
    row = st["ptrs_h"][li]
    nfree = [len(f) for f in st["vfree"]]
    for e, k in zip(uniq, keys):
        s = st["vram_of"][k]
        if s >= 0:
            vlru[st["vcls"][s]].move_to_end(k); n_vh += 1
            if row[e] != st["vaddr"][s]:
                row[e] = st["vaddr"][s]; changed = True; n_fix += 1
            continue
        rs = st["ram_of"][k]
        if rs < 0 or st["flags"][k] & 3:
            slow.append((e, k)); continue
        c = st["key_cls"][k]
        adm = False
        if n_adm < admit_per_call and st["freq"][k] >= admit_min:
            if nfree[c] > 0:
                adm = True
            else:
                v = next((kk for kk in vlru[c] if kk not in protect and kk not in consumed), None)
                adm = v is not None and st["freq"][v] < st["freq"][k]
                if adm:
                    consumed.add(v)
        if not adm:
            rlru[c].move_to_end(k)
            if row[e] != st["ram_addr"][rs]:
                row[e] = st["ram_addr"][rs]; changed = True
            n_zc += 1
            continue
        n_adm += 1
        if nfree[c] > 0:
            nfree[c] -= 1
        admit.append((e, k))
    return (changed, slow, admit, n_vh, n_zc, n_fix, len(uniq)), vlru, rlru


class CoreTest(unittest.TestCase):
    def test_matches_python_ensure(self):
        rnd = random.Random(11)
        for trial in range(2000):
            st = make_state(rnd)
            ref = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
            li = rnd.randrange(st["L"])
            ids = np.array([rnd.randrange(st["E"]) for _ in range(rnd.choice([8, 8, 16, 24]))], np.int64)
            vl = [tn.Lru(st["L"] * st["E"]) for _ in range(st["C"])]
            rl = [tn.Lru(st["L"] * st["E"]) for _ in range(st["C"])]
            for k in st["order_v"]:
                vl[st["vcls"][st["vram_of"][k]]][k] = int(st["vram_of"][k])
            for k in st["order_r"]:
                rl[st["rcls"][st["ram_of"][k]]][k] = int(st["ram_of"][k])
            core = tn.Core(st["L"], st["E"], st["key_cls"].ctypes.data, st["vcls"].ctypes.data, st["rcls"].ctypes.data,
                           st["vram_of"].ctypes.data, st["ram_of"].ctypes.data, st["ptrs_h"].ctypes.data,
                           st["vaddr"].ctypes.data, st["ram_addr"].ctypes.data, st["freq"].ctypes.data,
                           st["hits"].ctypes.data, st["flags"].ctypes.data, vl, rl, 4.0, 2)
            decay = rnd.random() < 0.1
            got = core.ensure_fast(li, ids, [len(f) for f in st["vfree"]], True, decay)
            exp, vlru, rlru = py_ensure(ref, li, ids, decay=decay)
            got = (got[0], [tuple(x) for x in got[1]], [tuple(x) for x in got[2]]) + tuple(got[3:])
            self.assertEqual(got, exp, f"trial {trial}")
            self.assertTrue(np.array_equal(st["ptrs_h"], ref["ptrs_h"]), f"ptrs trial {trial}")
            self.assertTrue(np.array_equal(st["freq"], ref["freq"]) and np.array_equal(st["hits"], ref["hits"]))
            for c in range(st["C"]):
                self.assertEqual(list(vl[c]), list(vlru[c]), f"VRAM LRU order trial {trial}")
                self.assertEqual(list(rl[c]), list(rlru[c]), f"RAM LRU order trial {trial}")

    def test_after_sync_matches_python(self):
        rnd = random.Random(5)
        for trial in range(2000):
            st = make_state(rnd)
            L, E = st["L"], st["E"]
            li = rnd.randrange(L - 1)
            nxt = li + 1 if rnd.random() < 0.8 else -1
            Q, k, w1 = rnd.choice([1, 1, 2, 3]), 8, 16
            miss_pf = rnd.choice([-1, 0, 1, 2])
            rows = []
            for t in range(Q):
                ex = rnd.sample(range(E), k)
                sel = ex[:] if rnd.random() < 0.3 else rnd.sample(ex, rnd.randrange(4, k + 1))
                while len(sel) < k:
                    e = rnd.randrange(E)
                    if e not in sel:
                        sel.append(e)
                rows.append((sel, ex))
            pn = [rnd.randrange(E) for _ in range(Q * w1)]
            picks = np.array([e for s_, _ in rows for e in s_] + [e for _, x in rows for e in x] + pn, np.int64)
            ref_freq = st["freq"].copy()
            # Python reference (block_graphs host step before after_sync, plus tier.prefetch's filter)
            n_sub, miss = 0, []
            for sel, ex in rows:
                ranked = [e for e in ex if e not in set(sel)]
                n_sub += len(ranked)
                miss += ranked[:miss_pf] if miss_pf >= 0 else ranked
                for e in ranked:
                    ref_freq[li * E + e] += 1.0

            def want(key):
                return st["vram_of"][key] < 0 and st["ram_of"][key] < 0 and not st["flags"][key] & 1
            exp_next = [e for e in sorted(set(pn)) if want(nxt * E + e)] if nxt >= 0 else []
            exp_this = [e for e in sorted(set(miss)) if want(li * E + e)]
            vl = [tn.Lru(L * E) for _ in range(st["C"])]
            core = tn.Core(L, E, st["key_cls"].ctypes.data, st["vcls"].ctypes.data, st["rcls"].ctypes.data,
                           st["vram_of"].ctypes.data, st["ram_of"].ctypes.data, st["ptrs_h"].ctypes.data,
                           st["vaddr"].ctypes.data, st["ram_addr"].ctypes.data, st["freq"].ctypes.data,
                           st["hits"].ctypes.data, st["flags"].ctypes.data, vl, vl, 4.0, 2)
            got = core.after_sync(li, nxt, picks, Q, k, w1, miss_pf)
            self.assertEqual((list(got[0]), list(got[1]), got[2], got[3]), (exp_next, exp_this, n_sub, Q * k),
                             f"trial {trial}")
            self.assertTrue(np.array_equal(st["freq"], ref_freq), f"freq trial {trial}")

    def test_bad_ids_raise(self):
        st = make_state(random.Random(1))
        vl = [tn.Lru(st["L"] * st["E"]) for _ in range(st["C"])]
        core = tn.Core(st["L"], st["E"], st["key_cls"].ctypes.data, st["vcls"].ctypes.data, st["rcls"].ctypes.data,
                       st["vram_of"].ctypes.data, st["ram_of"].ctypes.data, st["ptrs_h"].ctypes.data,
                       st["vaddr"].ctypes.data, st["ram_addr"].ctypes.data, st["freq"].ctypes.data,
                       st["hits"].ctypes.data, st["flags"].ctypes.data, vl, vl, 4.0, 2)
        with self.assertRaises(IndexError):
            core.ensure_fast(0, np.array([st["E"]], np.int64), [0, 0], True)
        with self.assertRaises(IndexError):
            core.ensure_fast(st["L"], np.array([0], np.int64), [0, 0], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
