"""Small-model checks for the real engine, with no Hub download."""

from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:  # The cockpit's mock mode does not require torch.
    torch = None

from sememe.engine.api import EngineError
from sememe.engine.torch_engine import TorchEngine


@unittest.skipIf(torch is None, "install sememe[torch] for engine tests")
class TorchEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Linear(2, 2, bias=False)
                self.register_buffer("marker", torch.tensor([0, 1]))
                self.config = types.SimpleNamespace(to_dict=lambda: {"hidden_size": 2})
                with torch.no_grad():
                    self.layer.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        self.model = TinyModel()
        self.engine = TorchEngine()
        fake_transformers = types.SimpleNamespace(
            AutoConfig=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: types.SimpleNamespace()),
            AutoModel=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: self.model),
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            self.info = self.engine.load(Path.cwd())

    def test_static_metadata_has_direct_tensors_and_correct_counts(self) -> None:
        self.assertEqual(self.info.class_name, "TinyModel")
        self.assertEqual([m.path for m in self.info.modules], ["", "layer"])
        self.assertEqual(self.info.param_count, 4)
        self.assertEqual(self.info.config, {"hidden_size": 2})

        root = self.info.module("")
        layer = self.info.module("layer")
        self.assertEqual(root.params, ())
        self.assertEqual(root.buffers[0].name, "marker")
        self.assertEqual(root.buffers[0].shape, (2,))
        self.assertEqual(layer.params[0].name, "weight")
        self.assertEqual(layer.params[0].shape, (2, 2))
        self.assertEqual(layer.params[0].bytes, 16)

    def test_stats_are_computed_only_when_requested(self) -> None:
        stats = self.engine.param_stats("layer", "weight")
        self.assertEqual((stats.min, stats.max, stats.mean), (1.0, 4.0, 2.5))
        self.assertAlmostEqual(stats.std, math.sqrt(1.25))
        self.assertAlmostEqual(stats.l2_norm, math.sqrt(30))
        self.assertEqual(sum(stats.histogram), 4)
        self.assertEqual(self.engine.param_stats("", "marker").zero_fraction, 0.5)

    def test_unknown_module_or_tensor_is_an_engine_error(self) -> None:
        with self.assertRaisesRegex(EngineError, "Unknown module"):
            self.engine.param_stats("absent", "weight")
        with self.assertRaisesRegex(EngineError, "no tensor"):
            self.engine.param_stats("layer", "bias")


if __name__ == "__main__":
    unittest.main()


def _cached_qwen() -> Path | None:
    from sememe.sources import scan_hub_cache
    return next((c.folder for c in scan_hub_cache() if c.label == "Qwen/Qwen3.5-0.8B" and c.loadable), None)


@unittest.skipIf(torch is None or _cached_qwen() is None, "needs sememe[torch] and a cached Qwen/Qwen3.5-0.8B")
class RealLoadEventTests(unittest.TestCase):
    """Against the real transformers loader: the counts it reports are its own."""

    def test_install_events_count_every_parameter_and_the_hook_is_restored(self) -> None:
        import transformers.core_model_loading as loading
        original = loading.tqdm
        events = []
        info = TorchEngine().load(_cached_qwen(), events.append)
        install = [e for e in events if e.phase == "install" and e.done is not None]
        self.assertGreater(len(install), 1)
        self.assertEqual([e.done for e in install], list(range(1, install[-1].total + 1)))
        self.assertTrue(install[-1].finished and install[-1].unit == "parameters")
        self.assertEqual([e.phase for e in events if e.finished],
                         ["import", "config", "install", "inspect"])
        starts = [e.phase for e in events if not e.finished and e.done is None]
        self.assertEqual(starts, ["import", "config", "install", "inspect"])  # every phase opens before it works
        self.assertIs(loading.tqdm, original)
        self.assertGreater(info.param_count, 0)

    def test_a_failed_load_still_restores_the_hook_and_names_its_phase(self) -> None:
        import transformers.core_model_loading as loading
        original = loading.tqdm
        engine = TorchEngine()
        with patch("transformers.AutoModel.from_pretrained", side_effect=RuntimeError("disk vanished")):
            with self.assertRaises(EngineError) as caught:
                engine.load(_cached_qwen())
        self.assertIn("install failed: disk vanished", str(caught.exception))
        self.assertIs(loading.tqdm, original)
