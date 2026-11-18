"""Exercise the adapter loader with fake modules, without importing XPU libraries."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


class Tensor:
    def float(self):
        return self

    def __mul__(self, value):
        return self


class Linear:
    def __init__(self, key):
        self.key = key


class LoraPathsTest(unittest.TestCase):
    def load(self):
        torch = types.ModuleType("torch")
        torch.inference_mode = lambda: lambda fn: fn
        modules = {"torch": torch}
        for name, cls in [("linear", Linear), ("mla_attn", type("MLAttention", (), {})),
                          ("mlp", type("GatedMLP", (), {}))]:
            module = types.ModuleType("exllamav3.modules." + name)
            setattr(module, {"linear": "Linear", "mla_attn": "MLAttention", "mlp": "GatedMLP"}[name], cls)
            modules[module.__name__] = module
        self.context = patch.dict(sys.modules, modules)
        self.context.start()
        self.addCleanup(self.context.stop)
        spec = importlib.util.spec_from_file_location("tested_lora", Path(__file__).parent / "glm_xpu/lora.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run_adapter(self, paths, keys):
        module = self.load()
        module._peft_tensors = lambda _: ({"r": 2, "lora_alpha": 2}, 1,
                                          {p: {"A": Tensor(), "B": Tensor()} for p in paths})
        module._set = Mock()
        module.apply([Linear(k) for k in keys], "/adapter", log=lambda _: None)
        return module._set

    def test_short_prefix_matches_language_model(self):
        setter = self.run_adapter(["model.layers.15.self_attn.o_proj"],
                                  ["model.language_model.layers.15.self_attn.o_proj"])
        self.assertEqual(setter.call_count, 1)

    def test_language_model_prefix_keeps_matching(self):
        setter = self.run_adapter(["model.language_model.layers.15.self_attn.o_proj"],
                                  ["model.language_model.layers.15.self_attn.o_proj"])
        self.assertEqual(setter.call_count, 1)

    def test_only_unloaded_targets_does_not_silently_succeed(self):
        with self.assertRaisesRegex(ValueError, "no targets applied"):
            self.run_adapter(["model.layers.45.self_attn.o_proj"],
                             ["model.language_model.layers.15.self_attn.o_proj"])

    def test_unknown_target_in_loaded_layer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "match no module"):
            self.run_adapter(["model.layers.15.self_attn.missing"],
                             ["model.language_model.layers.15.self_attn.o_proj"])


if __name__ == "__main__":
    unittest.main()
