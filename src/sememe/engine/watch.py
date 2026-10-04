"""Read-only, bounded module-boundary observations. No torch import at startup."""
from contextlib import contextmanager, ExitStack
import time

from .api import WatchBudget, WatchSpec


class WatchSession:
    def __init__(self, model, specs, budget, run_id, torch, captures):
        if not isinstance(budget, WatchBudget):
            raise ValueError('watch_budget must be WatchBudget')
        if any(not isinstance(spec, WatchSpec) for spec in specs):
            raise ValueError('Every watch must be WatchSpec')
        if any(not isinstance(spec.module, str) or len(spec.module) > 512 for spec in specs):
            raise ValueError('Watch module must be a string of at most 512 characters')
        limits = WatchBudget()
        for name in ('max_watches', 'max_tokens', 'max_features'):
            value = getattr(budget, name)
            if type(value) is not int or not 1 <= value <= getattr(limits, name):
                raise ValueError(f'{name} must be an integer between 1 and {getattr(limits, name)}')
        if len(specs) > budget.max_watches:
            raise ValueError('Watch count exceeds max_watches')
        if any(not isinstance(s.where, str) or s.where not in ('input', 'output') for s in specs):
            raise ValueError('Unsupported watch boundary: expected input or output')
        addresses = [(s.module, s.where) for s in specs]
        if len(set(addresses)) != len(addresses):
            raise ValueError('Duplicate watch addresses')
        modules = dict(model.named_modules())
        for spec in specs:
            if spec.where not in ('input', 'output'):
                raise ValueError(f'Unsupported watch boundary: {spec.where}')
            if spec.module not in modules:
                raise ValueError(f'Unknown watch module: {spec.module}')
        self.targets = [(spec, modules[spec.module]) for spec in specs]
        self.budget, self.run_id, self.torch, self.captures = budget, run_id, torch, captures

    @contextmanager
    def attached(self, token_count):
        # Register cleanup immediately: even a later registration failure removes
        # every earlier hook. Hooks never return replacement inputs or outputs.
        with ExitStack() as stack:
            for spec, module in self.targets:
                result = dict(run_id=self.run_id, module=spec.module, where=spec.where,
                              status='not_called', invocations=0, summaries=[])
                self.captures.append(result)
                def observe(value, result=result):
                    result['invocations'] += 1
                    if result['invocations'] > 1:
                        result['repeated_invocations_skipped'] = result['invocations'] - 1
                        return
                    start = time.monotonic()
                    try:
                        self.summarize(value, result, token_count)
                    finally:
                        result['summary_seconds'] = time.monotonic() - start
                if spec.where == 'output':
                    def hook(owner, args, output, observe=observe):
                        observe(output)
                    handle = module.register_forward_hook(hook)
                else:
                    def hook(owner, args, kwargs, observe=observe, result=result):
                        # The model's activation argument, not auxiliary masks/ids.
                        # This does not infer a head axis or select arbitrary kwargs.
                        if args and self.torch.is_tensor(args[0]):
                            result['argument'] = 'args[0]'
                            observe(args[0])
                        elif self.torch.is_tensor(kwargs.get('hidden_states')):
                            result['argument'] = 'hidden_states'
                            observe(kwargs['hidden_states'])
                        else:
                            observe(None)
                    handle = module.register_forward_pre_hook(hook, with_kwargs=True)
                stack.callback(handle.remove)
            yield

    def summarize(self, value, result, token_count):
        torch = self.torch
        if isinstance(value, (tuple, list)):
            tensors = [v for v in value if torch.is_tensor(v)]
            value = tensors[0] if len(tensors) == 1 else None
        if not torch.is_tensor(value):
            result.update(status='unsupported', reason='Boundary must contain exactly one direct tensor')
            return
        result.update(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))
        if (value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != token_count
                or value.is_complex() or value.is_sparse or not value.is_floating_point()):
            result.update(status='unsupported', reason='Requires a floating [1, input_tokens, features] tensor')
            return
        if not 0 < value.shape[2] <= self.budget.max_features:
            result.update(status='budget_skipped', reason='Feature count exceeds budget or is empty')
            return
        count = min(token_count, self.budget.max_tokens)
        result.update(status='ok', tokens_total=token_count, tokens_captured=count,
                      tokens_truncated=count < token_count, basis='all features per captured token; float32 population statistics')
        for index in range(count):
            row = value[0, index].detach().float()
            finite = torch.isfinite(row)
            bad = int((~finite).sum())
            stats = dict(token_index=index, features=row.numel(), nonfinite=bad)
            if bad:
                stats.update(mean=None, std=None, min=None, max=None, l2_norm=None)
            else:
                stats.update(mean=float(row.mean()), std=float(row.std(unbiased=False)),
                             min=float(row.min()), max=float(row.max()), l2_norm=float(row.norm()))
                # Reductions can overflow even when inputs are finite. JSON stays valid.
                import math
                for key in ('mean', 'std', 'min', 'max', 'l2_norm'):
                    if not math.isfinite(stats[key]):
                        stats[key] = None
                        stats['reduction_nonfinite'] = True
            result['summaries'].append(stats)
