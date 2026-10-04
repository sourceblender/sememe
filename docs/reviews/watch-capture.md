# Observe-only capture foundation

`Engine.run(..., watches=(WatchSpec(module, where),), watch_budget=WatchBudget())`
adds observations without changing the UI or the forward's outputs. Module
addresses use the same inspected core namespace as `model_info`; the task head
remains outside that namespace. `where` is `input` or `output`.

Each capture records its run id, module, boundary, shape, dtype, device and
invocation count. For a single floating tensor shaped `[1, input_tokens,
features]`, it reports mean, population standard deviation, min, max and L2
norm over all features for each captured token. Statistics use float32; their
cost is recorded as `summary_seconds` (including synchronization). These
numbers describe activations, not importance or causality.

Defaults are hard ceilings: eight addresses, 128 tokens per address, 65,536
features per token, one summarized invocation per address. Callers may lower
these limits. Later invocations are counted but skipped; excess tokens are
explicitly truncated; excess features skip the entire capture. No raw tensor
is retained or written. Temporary reduction storage is bounded to one token's
features. Run records store requested budgets and actual results. A rejected
watch list records its first eight specs and total requested count.

Unknown paths, duplicate addresses, unsupported boundaries and invalid budgets
reject the attempt before its forward. Runtime values which cannot be interpreted
as a single token tensor carry an `unsupported` status and reason. Tuples/lists
are accepted only when exactly one direct element is a tensor; mappings and
ambiguous tuples are unsupported. Nonfinite elements produce counts and null
statistics, never invalid JSON; reduction overflow is also explicit. A target
which was not called has `not_called` status.

All forward calls on one engine serialize, including unwatched calls, so hooks
cannot capture another run on that engine. Hooks leave pre-existing hooks alone
and are removed after success, exception or cancellation, including partial
registration failure. Stop is checked again after acquiring the lock and after
the forward; an executing forward is not interrupted. Summaries from a failed
or discarded forward remain in its attempt record.

This slice supplies no sidebar controls, interventions, full-tensor capture,
attention matrices or head-level targeting. A later Watch interaction can consume
these records without inventing observations from the mock engine (which explicitly
rejects watches).
