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
        self.assertEqual(layer.description, "in_features=2, out_features=2, bias=False")

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


@unittest.skipIf(torch is None, "install sememe[torch] for engine tests")
class NonFiniteScoreTests(unittest.TestCase):
    """A run whose final-position logits are not finite must fail in phase
    score, with an attempt record, never succeed with NaN probabilities."""

    def _engine(self, logits):
        import tempfile

        class Fixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))

            def forward(self, input_ids=None, attention_mask=None, use_cache=None):
                return types.SimpleNamespace(logits=torch.tensor([[logits]]))

        class OneToken:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[1]])}

            def decode(self, ids):
                return f"<{ids[0]}>"

        engine = TorchEngine()
        engine._full, engine._tokenizer, engine._torch, engine._identity = Fixed(), OneToken(), torch, {}
        self.runs = tempfile.TemporaryDirectory()
        self.addCleanup(self.runs.cleanup)
        env = patch.dict("os.environ", {"SEMEME_RUNS_DIR": self.runs.name})
        env.start()
        self.addCleanup(env.stop)
        return engine

    def test_nan_logits_fail_in_score_and_are_recorded(self) -> None:
        import json
        from sememe.engine.api import RunFailed, RunSettings
        engine = self._engine([float("nan"), 1.0, 2.0])
        with self.assertRaises(RunFailed) as caught:
            engine.run("x", RunSettings(top_k=1))
        self.assertEqual(caught.exception.status, "failed")
        self.assertIn("non-finite logits", str(caught.exception))
        record = json.loads(Path(caught.exception.record_path).read_text())
        self.assertEqual((record["status"], record["phase"]), ("failed", "score"))
        self.assertEqual(record["candidates"], [])

    def test_infinite_logits_fail_the_same_way(self) -> None:
        from sememe.engine.api import RunFailed, RunSettings
        with self.assertRaises(RunFailed) as caught:
            self._engine([float("inf"), 1.0, 2.0]).run("x", RunSettings(top_k=1))
        self.assertIn("non-finite", str(caught.exception))

    def test_finite_logits_still_succeed(self) -> None:
        from sememe.engine.api import RunSettings
        result = self._engine([0.0, 1.0, 2.0]).run("x", RunSettings(top_k=1))
        self.assertEqual(result.candidates[0].id, 2)
        self.assertIsNotNone(result.record_path)


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
                         ["import", "config", "install", "tokenizer", "inspect"])
        starts = [e.phase for e in events if not e.finished and e.done is None]
        self.assertEqual(starts, ["import", "config", "install", "tokenizer", "inspect"])  # every phase opens before it works
        self.assertIs(loading.tqdm, original)
        self.assertGreater(info.param_count, 0)

    def test_a_failed_load_still_restores_the_hook_and_names_its_phase(self) -> None:
        import transformers.core_model_loading as loading
        original = loading.tqdm
        engine = TorchEngine()
        class Vanishing:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("disk vanished")

        with patch("sememe.engine.torch_engine._task_class", return_value=Vanishing):
            with self.assertRaises(EngineError) as caught:
                engine.load(_cached_qwen())
        self.assertIn("install failed: disk vanished", str(caught.exception))
        self.assertIs(loading.tqdm, original)


@unittest.skipIf(torch is None or _cached_qwen() is None, "needs sememe[torch] and a cached Qwen/Qwen3.5-0.8B")
class RealRunTests(unittest.TestCase):
    """A run must equal a direct transformers call on the same tokens."""

    @classmethod
    def setUpClass(cls) -> None:
        import tempfile
        cls.runs = tempfile.TemporaryDirectory()
        cls.env = patch.dict("os.environ", {"SEMEME_RUNS_DIR": cls.runs.name})
        cls.env.start()
        cls.engine = TorchEngine()
        cls.engine.load(_cached_qwen())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.stop()
        cls.runs.cleanup()

    def test_top_candidates_equal_a_direct_call_over_the_full_vocabulary(self) -> None:
        import transformers
        from sememe.engine.api import RunSettings
        prompt = "The capital of France is"
        result = self.engine.run(prompt, RunSettings(top_k=5))
        cls = getattr(transformers, transformers.AutoConfig.from_pretrained(_cached_qwen()).architectures[0])
        model = cls.from_pretrained(_cached_qwen(), dtype="auto").eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(_cached_qwen())
        ids = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            probs = model(**ids).logits[0, -1].float().softmax(-1)
        top_p, top_i = probs.topk(5)
        self.assertEqual([t.id for t in result.tokens], ids["input_ids"][0].tolist())
        self.assertEqual([c.id for c in result.candidates], top_i.tolist())
        for c, p in zip(result.candidates, top_p.tolist()):
            self.assertAlmostEqual(c.probability, p, places=6)
        self.assertEqual(result.candidates[0].text, " Paris")
        self.assertEqual(result.used["vocab_size"], probs.shape[-1])

    def test_a_run_writes_one_bounded_record(self) -> None:
        import json
        from sememe.engine.api import RunSettings
        result = self.engine.run("Hello world", RunSettings(top_k=3))
        record = json.loads(Path(result.record_path).read_text())
        self.assertEqual(record["schema"], "sememe.run/1")
        self.assertEqual(record["run_id"], result.run_id)
        self.assertEqual(len(record["candidates"]), 3)
        self.assertEqual(record["model"]["class"], "Qwen3_5ForConditionalGeneration")
        self.assertEqual(len(record["model"]["config_sha256"]), 64)
        self.assertNotIn("logits", record)  # no tensors in a record
        self.assertLess(len(json.dumps(record)), 8_000)  # bounded: tokens, top-k and metadata only

    def test_whitespace_and_unicode_tokens_keep_their_exact_text(self) -> None:
        from sememe.engine.api import RunSettings
        result = self.engine.run("Café 東京\n", RunSettings(top_k=1))
        self.assertEqual("".join(t.text for t in result.tokens), "Café 東京\n")

    def test_an_engine_closed_after_loading_refuses_to_run(self) -> None:
        from sememe.engine.api import RunFailed, RunSettings
        engine = TorchEngine()
        engine.load(_cached_qwen())
        engine.close()
        with self.assertRaises(RunFailed) as caught:
            engine.run("x", RunSettings())
        self.assertEqual(caught.exception.status, "rejected")

    def test_only_the_last_position_is_computed_and_no_cache_is_kept(self) -> None:
        from sememe.engine.api import RunSettings
        used = self.engine.run("one two three four five six", RunSettings(top_k=1)).used
        self.assertEqual((used["use_cache"], used["logits_to_keep"], used["logits_positions"]), (False, 1, 1))

    def test_over_budget_prompts_and_bad_top_k_are_rejected_and_recorded_without_a_forward(self) -> None:
        import json
        from sememe.engine.api import RunFailed, RunSettings
        with self.assertRaises(RunFailed) as caught:
            self.engine.run("The capital of France is", RunSettings(top_k=3, max_input_tokens=2))
        failure = caught.exception
        self.assertEqual(failure.status, "rejected")
        self.assertIn("limit for one run is 2", str(failure))
        record = json.loads(Path(failure.record_path).read_text())
        self.assertEqual((record["status"], record["phase"]), ("rejected", "input"))
        self.assertNotIn("forward", record["timing"])
        self.assertEqual(len(record["tokens"]), 5)
        with self.assertRaises(RunFailed) as caught:
            self.engine.run("x", RunSettings(top_k=0))
        self.assertIn("between 1 and 100", str(caught.exception))

    def test_rejections_are_bounded_and_never_pretend_to_keep_the_exact_input(self) -> None:
        import json
        from sememe.engine.api import MAX_INPUT_TOKENS_CEILING, MAX_PROMPT_CHARS, RunFailed, RunSettings
        with self.assertRaises(RunFailed) as caught:
            self.engine.run("x", RunSettings(max_input_tokens=MAX_INPUT_TOKENS_CEILING + 1))
        self.assertIn("max_input_tokens must be between", str(caught.exception))
        huge = "word " * (MAX_PROMPT_CHARS // 5 + 10)
        with self.assertRaises(RunFailed) as caught:
            self.engine.run(huge, RunSettings())
        record = json.loads(Path(caught.exception.record_path).read_text())
        self.assertNotIn("tokenize", record["timing"])  # rejected before tokenizing
        self.assertTrue(record["prompt_truncated"])
        self.assertEqual(record["prompt_chars"], len(huge))
        self.assertLessEqual(len(record["prompt"]), 2_000)
        self.assertEqual(len(record["prompt_sha256"]), 64)
        long = "token " * 3_000  # under the character guard, over a small token budget
        with self.assertRaises(RunFailed) as caught:
            self.engine.run(long, RunSettings(max_input_tokens=100))
        record = json.loads(Path(caught.exception.record_path).read_text())
        self.assertTrue(record["tokens_truncated"])
        self.assertLessEqual(len(record["tokens"]), 256)
        self.assertGreater(record["token_count"], 100)
        self.assertLess(len(json.dumps(record)), 64_000)

    def test_a_cancelled_attempt_is_recorded_as_cancelled(self) -> None:
        import json
        from sememe.engine.api import RunFailed, RunSettings
        calls = []

        def cancelled() -> bool:
            calls.append(1)
            return len(calls) >= 2  # let tokenizing start, stop before the forward

        with self.assertRaises(RunFailed) as caught:
            self.engine.run("Hello", RunSettings(), cancelled=cancelled)
        record = json.loads(Path(caught.exception.record_path).read_text())
        self.assertEqual((record["status"], record["phase"]), ("cancelled", "forward"))

    def test_a_record_that_cannot_be_written_is_not_reported_as_saved(self) -> None:
        import tempfile
        from sememe.engine.api import RunSettings
        with tempfile.NamedTemporaryFile() as blocker, patch.dict("os.environ", {"SEMEME_RUNS_DIR": blocker.name}):
            result = self.engine.run("Hello", RunSettings(top_k=1))
        self.assertIsNone(result.record_path)
        self.assertIn("record not saved", result.record_error)

    def test_identity_names_the_snapshot_revision_and_scopes_the_config_hash(self) -> None:
        from sememe.engine.api import RunSettings
        model = self.engine.run("Hello", RunSettings(top_k=1)).model
        self.assertEqual(model["hub_repo"], "Qwen/Qwen3.5-0.8B")
        self.assertEqual(model["revision"], Path(_cached_qwen()).name)
        self.assertIn("does not identify weights", model["config_sha256_scope"])


class TaskClassTests(unittest.TestCase):
    def test_only_next_token_heads_can_run(self) -> None:
        from sememe.engine.torch_engine import _task_class
        fake = types.SimpleNamespace(BertForSequenceClassification=object, BertForMaskedLM=object,
                                     LlamaForCausalLM=object)
        pick = lambda *arch: _task_class(fake, types.SimpleNamespace(architectures=list(arch)))  # noqa: E731
        self.assertIsNone(pick("BertForSequenceClassification"))
        self.assertIsNone(pick("BertForMaskedLM"))
        self.assertIs(pick("LlamaForCausalLM"), object)
