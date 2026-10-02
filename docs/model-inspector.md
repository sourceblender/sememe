# Inspecting a loaded model

The Architecture tab is an overview of the loaded module structure. Colours
identify component types; they do not indicate execution, importance or health.
The Qwen-style overview groups its language decoder and vision tower. Models
without that decoder pattern show their top-level components instead. The
Modules tab is the complete ownership hierarchy reported by the engine,
including weightless modules and components omitted from the overview. It is
not an observed execution graph.

Click a block or choose a module in Modules. The Inspect sidebar shows its
exact address and class, its directly owned and subtree parameter counts, and
any description reported by the backend module. Parent returns to its owner.
The child tree lets you drill down further.

The tensor list includes parameters and buffers throughout the selected
subtree. Select a row with the mouse or arrow keys and Enter. The inspector
shows that tensor's complete address, shape, dtype, device, number of values
and logical size in bytes. Logical size does not measure resident memory or
physical reads from disk. A module with one tensor selects it automatically.
The Tables tab can also select a tensor directly into the same inspector.

Measure distribution (or `s`) reads only the selected tensor's values, in a
worker. It shows min/max, mean, population standard deviation, L2 norm, zero
fraction and 16 equal-width histogram bins spanning min to max. No forward
pass runs. Non-finite or unsupported tensors produce an explicit error rather
than a distribution. A late result is discarded after selection changes,
model replacement or Close Model.

Normal model loads use real engine metadata and weight values. `--mock` uses
synthetic metadata and values, explicitly labelled MOCK. Decode is still a mock panel. The synthetic performance strip appears only
in mock mode; a real model has no execution measurements yet.

This slice establishes selection as a module/tensor address within the loaded
model. Watches, logs, hooks, activity/replay and saving configurations are
future tools to discuss from this view; they are not implemented here.
