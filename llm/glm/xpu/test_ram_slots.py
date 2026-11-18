"""CPU-only tests of Tier's actual slot-selection methods, isolated from XPU imports."""
import ast
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
import unittest
from unittest.mock import Mock


def slot_methods():
    path = Path(__file__).parent / "glm_xpu/tier.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Tier")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_take_ram_slot", "_reap_inflight")]
    namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return type("SlotMethods", (), namespace)


class PendingRead:
    def __init__(self):
        self.finished = False

    def done(self):
        return self.finished

    def result(self):
        self.finished = True


class RamSlotsTest(unittest.TestCase):
    def tier(self):
        t = slot_methods()()
        t.free_ram = [[]]
        t.ram_lru = [OrderedDict([(0, 0), (1, 1)])]
        t.ram_of = [0, 1]
        t.ram_key = [0, 1]
        t.vram_of = [-1, -1]
        t.inflight = {0: [PendingRead()], 1: [PendingRead()]}
        t.wb = {}
        t._set_res = Mock()
        return t

    def test_required_slot_waits_for_unprotected_read(self):
        t = self.tier()
        self.assertEqual(t._take_ram_slot({0}, 0), 1)
        self.assertEqual(t.ram_of, [0, -1])
        self.assertIn(0, t.inflight)
        self.assertNotIn(1, t.inflight)

    def test_never_evicts_protected_experts(self):
        t = self.tier()
        self.assertEqual(t._take_ram_slot({0, 1}, 0), -1)
        self.assertEqual(t.ram_of, [0, 1])

    def test_prefetch_does_not_wait(self):
        t = self.tier()
        self.assertEqual(t._take_ram_slot(set(), 0, wait=False), -1)
        self.assertFalse(t.inflight[0][0].finished)
        self.assertFalse(t.inflight[1][0].finished)

    def test_does_not_evict_pending_writeback(self):
        t = self.tier()
        t.wb = {1: object()}
        self.assertEqual(t._take_ram_slot({0}, 0), -1)
        self.assertEqual(t.ram_of, [0, 1])

    def test_completed_reads_are_reaped(self):
        t = self.tier()
        t.inflight[1][0].result()
        self.assertEqual(t._take_ram_slot({0}, 0), 1)

    def test_read_errors_surface(self):
        t = self.tier()
        bad = Future()
        bad.set_exception(IOError("read failed"))
        t.inflight[1] = [bad]
        with self.assertRaisesRegex(IOError, "read failed"):
            t._take_ram_slot({0}, 0)


if __name__ == "__main__":
    unittest.main()
