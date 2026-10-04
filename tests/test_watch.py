"""Capture integrity, bounded summaries and cleanup without a Hub download."""
import json
import types
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
from sememe.engine.api import RunFailed, RunSettings, WatchSpec, WatchBudget
from sememe.engine.torch_engine import TorchEngine
from sememe.engine.watch import WatchSession


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv('SEMEME_RUNS_DIR', str(tmp_path))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.node = torch.nn.Linear(2, 3, bias=False)
            with torch.no_grad():
                self.node.weight.copy_(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
            self.calls = 0
            self.explode = False
        def forward(self, input_ids, use_cache=False):
            self.calls += 1
            x = torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)
            output = self.node(x)
            if self.explode:
                raise RuntimeError('broken forward')
            return types.SimpleNamespace(logits=output)
    class Tokenizer:
        def __call__(self, text, return_tensors=None):
            return {'input_ids': torch.tensor([[1, 2, 3]])}
        def decode(self, ids):
            return str(ids[0])
    e = TorchEngine()
    e._full, e._torch, e._tokenizer = Tiny(), torch, Tokenizer()
    return e


def assert_clean(e):
    for node in e._full.modules():
        assert not node._forward_hooks
        assert not node._forward_pre_hooks


def test_observe_does_not_change_logits_and_records_exact_summaries(engine):
    baseline = engine.run('x', RunSettings())
    result = engine.run('x', RunSettings(), watches=(WatchSpec('node'), WatchSpec('node', 'input')))
    assert result.candidates == baseline.candidates
    output, inputs = result.captures
    assert output['run_id'] == result.run_id
    assert output['shape'] == [1, 3, 3]
    assert inputs['shape'] == [1, 3, 2]
    row = torch.tensor([5., 11., 17.])
    assert output['summaries'][0]['mean'] == float(row.mean())
    assert output['summaries'][0]['std'] == float(row.std(unbiased=False))
    assert output['summaries'][0]['l2_norm'] == float(row.norm())
    record = json.loads(Path(result.record_path).read_text())
    assert record['captures'] == list(result.captures)
    assert record['watches'] == [{'module':'node', 'where':'output'}, {'module':'node', 'where':'input'}]
    assert_clean(engine)


@pytest.mark.parametrize('watches,budget', [
    ((WatchSpec('absent'),), WatchBudget()),
    ((WatchSpec('node', 'head'),), WatchBudget()),
    ((WatchSpec('node'), WatchSpec('node')), WatchBudget()),
    ((WatchSpec('node'),), WatchBudget(max_tokens=129)),
    ((WatchSpec('node'),), WatchBudget(max_features=0)),
    (tuple(WatchSpec(str(i)) for i in range(9)), WatchBudget()),
])
def test_invalid_watch_rejected_before_forward_with_record(engine, watches, budget):
    with pytest.raises(RunFailed) as caught:
        engine.run('x', RunSettings(), watches=watches, watch_budget=budget)
    assert caught.value.status == 'rejected'
    assert json.loads(Path(caught.value.record_path).read_text())['phase'] == 'watch'
    assert engine._full.calls == 0
    assert_clean(engine)


def test_budgets_are_explicit_not_silent_samples(engine):
    r = engine.run('x', RunSettings(), watches=(WatchSpec('node'),), watch_budget=WatchBudget(max_tokens=2))
    c = r.captures[0]
    assert c['tokens_truncated'] and c['tokens_total'] == 3
    assert [x['token_index'] for x in c['summaries']] == [0, 1]
    r = engine.run('x', RunSettings(), watches=(WatchSpec('node'),), watch_budget=WatchBudget(max_features=2))
    assert r.captures[0]['status'] == 'budget_skipped'
    assert not r.captures[0]['summaries']
    assert_clean(engine)


def test_cleanup_on_failure_and_cancel_keeps_attempt_captures(engine):
    engine._full.explode = True
    with pytest.raises(RunFailed) as caught:
        engine.run('x', RunSettings(), watches=(WatchSpec('node'),))
    assert json.loads(Path(caught.value.record_path).read_text())['captures'][0]['status'] == 'ok'
    assert_clean(engine)
    engine._full.explode = False
    with pytest.raises(RunFailed) as caught:
        engine.run('x', RunSettings(), cancelled=lambda: engine._full.calls > 1, watches=(WatchSpec('node'),))
    assert caught.value.status == 'cancelled'
    assert_clean(engine)


def test_registration_failure_removes_earlier_hook(engine):
    with patch.object(engine._full.node, 'register_forward_pre_hook', side_effect=RuntimeError('registration failed')):
        with pytest.raises(RunFailed):
            engine.run('x', RunSettings(), watches=(WatchSpec('node'), WatchSpec('node', 'input')))
    assert_clean(engine)


def test_callback_failure_removes_hooks(engine):
    with patch.object(WatchSession, 'summarize', side_effect=RuntimeError('summary failed')):
        with pytest.raises(RunFailed):
            engine.run('x', RunSettings(), watches=(WatchSpec('node'),))
    assert_clean(engine)


def test_repeated_invocation_and_unsupported_and_nonfinite(engine):
    captures = []
    session = WatchSession(engine._full, (WatchSpec('node'),), WatchBudget(), 'run', torch, captures)
    with session.attached(3):
        engine._full.node(torch.ones(1, 3, 2))
        engine._full.node(torch.ones(1, 3, 2))
    assert captures[0]['invocations'] == 2
    assert captures[0]['repeated_invocations_skipped'] == 1
    assert len(captures[0]['summaries']) == 3
    for value in (torch.ones(3, 2), (torch.ones(1,3,2), torch.ones(1,3,2)), {'hidden':torch.ones(1,3,2)}):
        r = {'summaries': []}
        session.summarize(value, r, 3)
        assert r['status'] == 'unsupported'
    r = {'summaries': []}
    session.summarize(torch.full((1, 3, 2), float('nan')), r, 3)
    assert r['summaries'][0]['nonfinite'] == 2
    assert r['summaries'][0]['mean'] is None
    json.dumps(r, allow_nan=False)
    assert_clean(engine)


def test_cancel_before_forward_attaches_nothing(engine):
    with pytest.raises(RunFailed):
        engine.run('x', RunSettings(), cancelled=lambda: True, watches=(WatchSpec('node'),))
    assert engine._full.calls == 0
    assert_clean(engine)


def test_same_engine_concurrent_runs_do_not_capture_each_other(engine):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    entered, release = threading.Event(), threading.Event()
    original = engine._full.forward
    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    engine._full.forward = paused
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(engine.run, 'x', RunSettings(), watches=(WatchSpec('node'),))
        assert entered.wait(5)
        second = pool.submit(engine.run, 'y', RunSettings())
        release.set()
        a, b = first.result(), second.result()
    assert a.captures[0]['invocations'] == 1
    assert b.captures == ()
    assert_clean(engine)


def test_real_qwen_scores_equal_and_summary_matches_direct_hook(tmp_path, monkeypatch):
    from sememe.ui.picker import scan_hub_cache
    item = next((x for x in scan_hub_cache() if x.label == 'Qwen/Qwen3.5-0.8B' and x.loadable), None)
    if item is None:
        pytest.skip('requires cached Qwen/Qwen3.5-0.8B')
    monkeypatch.setenv('SEMEME_RUNS_DIR', str(tmp_path))
    e = TorchEngine()
    info = e.load(item.folder)
    path = next(m.path for m in info.modules if m.path.endswith('layers.0.mlp.down_proj'))
    reference = []
    module = dict(e._full.model.named_modules())[path]
    def direct(owner, args, output):
        # Independent direct reference for the final token, no retained activations.
        row = output[0, -1].float()
        reference.append((float(row.mean()), float(row.std(unbiased=False)), float(row.norm())))
    handle = module.register_forward_hook(direct)
    try:
        base = e.run('The capital of France is', RunSettings())
    finally:
        handle.remove()
    observed = e.run('The capital of France is', RunSettings(), watches=(WatchSpec(path),))
    assert observed.candidates == base.candidates
    c = observed.captures[0]
    assert c['status'] == 'ok'
    last = c['summaries'][-1]
    assert (last['mean'], last['std'], last['l2_norm']) == reference[0]
    assert_clean(e)
    e.close()
