"""The sememe cockpit: a Textual app over any `Engine`.

The UI never touches torch. It asks the engine through `sememe.engine.api` and
makes every blocking call from a thread worker, so the screen stays live while
a model loads. Panels whose data is not real yet say MOCK in their title.
"""

from __future__ import annotations

import argparse
import json
from typing import Callable
import random
from collections import deque
from dataclasses import dataclass

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
    Tree,
)

from sememe.engine.api import (Engine, EngineError, LoadEvent, ModelInfo, ModuleInfo, RunFailed, RunResult,
                               RunSettings, TensorInfo)
from sememe.sources import ModelChoice, label_for
from sememe.ui.picker import CLOSE_MODEL, LOAD_DISK, LOAD_HF, ModelMenu, DiskPicker, HubPicker, MenuBar, describe

# Component kinds, each with one colour across the whole UI.
# Fixed colours, so every terminal and theme draws a kind the same way.
KIND_STYLE = {
    "embed": "#1e1e1e on #e5c07b",
    "linear_attn": "#1e1e1e on #56b6c2",
    "full_attn": "#1e1e1e on #c678dd",
    "mlp": "#1e1e1e on #98c379",
    "norm": "#1e1e1e on #abb2bf",
    "head": "#1e1e1e on #d19a66",
    "vision": "#1e1e1e on #61afef",
    "other": "#d0d0d0 on #3e4451",
}
# Every row padded to one width: the off screen centres line by line, so rows
# of different lengths would slide against each other and scramble the letters.
_TITLE_ROWS = [
    "███████ ███████ ███    ███ ███████ ███    ███ ███████",
    "██      ██      ████  ████ ██      ████  ████ ██     ",
    "███████ █████   ██ ████ ██ █████   ██ ████ ██ █████  ",
    "     ██ ██      ██  ██  ██ ██      ██  ██  ██ ██     ",
    "███████ ███████ ██      ██ ███████ ██      ██ ███████",
]
TITLE_WIDTH = max(len(row) for row in _TITLE_ROWS)
TITLE_ART = "\n".join(row.ljust(TITLE_WIDTH) for row in _TITLE_ROWS)
OFF_ART = Text.assemble(
    (TITLE_ART + "\n", "bold #56b6c2"),
    ("\na debugger for a model's forward pass\n\n", "dim"),
    ("Model ▾", "bold"), ("  (F10)  →  Load Hugging Face Model…  or  Load Model from Disk…", ""),
)
MONITOR = {  # name: (label, typical value for the synthetic series)
    "prefill": ("prefill tok/s", 2400.0),
    "decode": ("decode tok/s", 48.0),
    "latency": ("step ms", 21.0),
    "memory": ("memory GB", 1.7),
}


def kind_of(module: ModuleInfo) -> str:
    """Classify a module for colouring, from its path and class."""
    leaf = module.path.rsplit(".", 1)[-1]
    cls = module.class_name.lower()
    if module.path.startswith("visual"):
        return "vision"
    if "embed" in leaf:
        return "embed"
    if leaf == "linear_attn" or "deltanet" in cls:
        return "linear_attn"
    if leaf == "self_attn" or cls.endswith("attention"):
        return "full_attn"
    if leaf == "mlp" or cls.endswith("mlp"):
        return "mlp"
    if "norm" in leaf or "norm" in cls:
        return "norm"
    if leaf == "lm_head":
        return "head"
    return "other"


def human(n: int) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(n)


@dataclass(frozen=True)
class LayerView:
    """One decoder layer as the block map draws it."""

    index: int
    path: str
    attention: ModuleInfo | None
    mlp: ModuleInfo | None


def decoder_layers(info: ModelInfo) -> list[LayerView]:
    """The repeated decoder layers: children of the first `*.layers` ModuleList
    outside the vision tower, in index order."""
    holder = next((m for m in info.modules if m.path.endswith("layers") and m.class_name == "ModuleList"
                   and not m.path.startswith("visual")), None)
    if holder is None:
        return []
    layers = sorted(info.children(holder.path), key=lambda m: int(m.path.rsplit(".", 1)[-1]))
    views = []
    for layer in layers:
        kids = {m.path.rsplit(".", 1)[-1]: m for m in info.children(layer.path)}
        attention = kids.get("self_attn") or kids.get("linear_attn") or next(
            (m for m in kids.values() if kind_of(m) in ("linear_attn", "full_attn")), None)
        views.append(LayerView(int(layer.path.rsplit(".", 1)[-1]), layer.path, attention, kids.get("mlp")))
    return views


class Cell(Static):
    """A clickable, focusable block standing for one real module path."""

    can_focus = True

    class Selected(Message):
        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    def __init__(self, label: str, module: ModuleInfo, **kwargs) -> None:
        super().__init__(Text(label, style=KIND_STYLE[kind_of(module)]), **kwargs)
        self.module_path = module.path
        self.tooltip = f"{module.path}\n{module.class_name}"

    def on_click(self) -> None:
        self.focus()
        self.post_message(self.Selected(self.module_path))

    def on_focus(self) -> None:
        self.post_message(self.Selected(self.module_path))


class BlockMap(VerticalScroll):
    """The model drawn as an architecture: embedding, decoder layers, norm."""

    def on_resize(self, event) -> None:
        for grid in self.query(Grid):
            grid.styles.grid_size_columns = max(1, min(6, (event.size.width - 2) // 14))

    def show(self, info: ModelInfo) -> None:
        self.remove_children()
        widgets: list = [Static(Text("Structure overview · colours identify module types, not activity", style="dim"))]
        if not decoder_layers(info):
            widgets.append(Label("model components · complete hierarchy in Modules", classes="section"))
            widgets.extend(Cell(f" {m.path} · {m.class_name} ", m) for m in info.children(""))
            self.mount_all(widgets)
            return
        vision = info.module("visual")
        if vision is not None:
            widgets.append(Label("vision tower", classes="section"))
            widgets.append(Cell(f" {vision.class_name} · {human(info.subtree_param_count('visual'))} ▸ ", vision))
        widgets.append(Label("language model · overview; all modules in Modules", classes="section"))
        embed = next((m for m in info.modules if kind_of(m) == "embed" and not m.path.startswith("visual")), None)
        if embed is not None:
            widgets.append(Cell(f" embed_tokens {'×'.join(map(str, embed.params[0].shape))} ", embed))
        layers = decoder_layers(info)
        grid = Grid(classes="layers")
        cards = []
        for layer in layers:
            attn_kind = kind_of(layer.attention) if layer.attention else "other"
            title = f"L{layer.index:02d} {'full' if attn_kind == 'full_attn' else 'lin'}"
            row = []
            if layer.attention is not None:
                row.append(Cell(" attn ", layer.attention, classes="cell"))
            if layer.mlp is not None:
                row.append(Cell(" mlp ", layer.mlp, classes="cell"))
            cards.append(Vertical(Label(title, classes="layer-title"), Horizontal(*row, classes="layer-row"),
                                  classes="layer"))
        widgets.append(Label(f"{len(layers)} decoder layers", classes="section"))
        self.mount_all(widgets)
        self.mount(grid)
        grid.mount_all(cards)
        tail = [m for m in info.modules if m.path.count(".") == 1 and kind_of(m) in ("norm", "head")
                and not m.path.startswith("visual")]
        for module in tail:
            self.mount(Cell(f" {module.path.rsplit('.', 1)[-1]} ", module))
        legend = Text()
        for kind, label in (("embed", "embed"), ("linear_attn", "linear attn"), ("full_attn", "full attn"),
                            ("mlp", "mlp"), ("norm", "norm"), ("vision", "vision")):
            legend.append(f" {label} ", style=KIND_STYLE[kind])
            legend.append(" ")
        self.mount(Static(legend, classes="legend"))


class Sidebar(VerticalScroll):
    """What the selected module is, what it owns, and its subtree."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="inspector-nav"):
            yield Button("↑ Parent", id="side-parent", disabled=True)
            yield Label("Inspect", id="inspector-label")
        yield Static("select a block", id="side-title")
        yield DataTable(id="side-params", cursor_type="row")
        yield Static("select a tensor", id="side-tensor")
        yield Button("Measure distribution (s)", id="side-measure", disabled=True)
        yield Static("", id="side-stats")
        yield Tree("subtree", id="side-tree")

    def on_mount(self) -> None:
        table = self.query_one("#side-params", DataTable)
        table.add_column("tensor", width=20)
        table.add_column("kind", width=6)
        table.add_column("shape", width=12)


class Cockpit(App):
    # Search starts hidden; it should only take focus when explicitly opened.
    AUTO_FOCUS = None

    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; }
    BlockMap { width: 1fr; padding: 0 1; }
    Sidebar { width: 56; border-left: solid $accent; padding: 0 1; }
    #cockpit { height: 1fr; }
    #inspector-nav { height: 1; }
    #inspector-nav Button { height: 1; min-width: 0; border: none; padding: 0 1; }
    #inspector-label { width: 1fr; text-align: right; color: $text-muted; }
    #side-tensor { height: auto; margin-top: 1; }
    #side-measure { height: 1; min-width: 0; border: none; padding: 0 1; margin-top: 1; }
    #model-tree { height: 1fr; }
    #model-summary { height: auto; margin: 0 1 1 1; }
    .selected-block { text-style: bold reverse; }
    .section { color: $text-muted; margin-top: 1; }
    .layers { grid-size: 6; grid-gutter: 0 1; height: auto; }
    .layer { height: 4; border: round $panel-lighten-2; }
    .layer-title { color: $text-muted; }
    .layer-row { height: 1; }
    .layer-row Cell { margin-right: 1; }
    #tabs { height: 1fr; width: 1fr; }
    Cell { width: auto; height: 1; }
    Cell:focus { text-style: reverse; }
    .legend { margin-top: 1; }
    #side-title { height: auto; margin-bottom: 1; }
    #side-params { height: auto; max-height: 8; }
    #side-stats { height: auto; margin: 1 0; }
    #side-tree { height: 12; }
    #run-panel { height: 1fr; padding: 0 1; }
    #run-bar { height: 3; }
    #run-prompt { width: 1fr; }
    #run-status { height: auto; margin-top: 1; }
    #run-tokens { height: auto; margin-top: 1; }
    #run-candidates { height: auto; margin-top: 1; }
    #run-meta { height: auto; margin-top: 1; }
    #search { display: none; dock: top; }
    #search.open { display: block; }
    #monitor { height: 6; border-top: solid $accent; }
    #monitor Vertical { width: 1fr; padding: 0 1; }
    #monitor Sparkline { height: 2; }
    MenuBar { height: 1; background: $panel; }
    MenuBar Button.menu { height: 1; min-width: 0; border: none; padding: 0 1; background: $panel; }
    MenuBar Button.menu:hover { background: $accent; }
    MenuBar Button.menu:disabled { color: $text-disabled; }
    #menu-spacer { width: 1fr; }
    #menu-selection { width: auto; padding: 0 1; }
    #stage { height: 1fr; }
    #off { width: 1fr; height: 1fr; content-align: center middle; text-align: center; }
    #loading { padding: 1 2; }
    #load-title { margin-bottom: 1; }
    #load-bar { margin-bottom: 1; }
    #load-basis { height: 2; margin-bottom: 1; }
    #load-log { height: 1fr; border: round $panel-lighten-2; }
    """
    BINDINGS = [
        Binding("f10", "menu", "menu"),
        Binding("slash", "search", "search"),
        Binding("s", "stats", "weight stats"),
        Binding("q", "quit", "quit"),
    ]
    TITLE = "sememe"

    def __init__(self, engine: Engine | None = None, model: str | None = None, mock: bool = False,
                 engine_factory: Callable[[], Engine] | None = None) -> None:
        """`model` given: load it at startup. Otherwise start on the Off screen.
        `engine_factory` makes a fresh engine for each load from the Model menu."""
        super().__init__()
        self.mock = mock
        self.startup_engine = engine
        self.startup_model = model
        self.engine_factory = engine_factory or (_fake_factory if mock else _torch_factory)
        self.engine: Engine | None = None  # the engine whose model is loaded
        self.model_ref: str | None = None
        self.info: ModelInfo | None = None
        self.selected_tensor: tuple[str, str] | None = None
        self.inspection_token = 0
        self.tensor_rows: dict[str, tuple[str, str, TensorInfo]] = {}
        self.selected: str | None = None
        self.choice: ModelChoice | None = None  # what the Model menu picked
        self.state = "off"  # off | loading | loaded | failed
        self.load_token = 0  # bumped by every load and close; stale results are dropped
        self.run_token = 0  # bumped by every run; a superseded run's result is dropped
        self.run_cancel = False  # Stop requested for the current run
        self.last_event: LoadEvent | None = None
        self.series = {name: deque([base] * 60, maxlen=60) for name, (_, base) in MONITOR.items()}

    def compose(self) -> ComposeResult:
        yield Header()
        yield MenuBar(id="menubar")
        yield Input(placeholder="jump to module path…", id="search")
        with ContentSwitcher(initial="off", id="stage"):
            yield Static(OFF_ART, id="off")
            with Vertical(id="loading"):
                yield Label("", id="load-title")
                yield ProgressBar(id="load-bar", show_eta=False)
                yield Static("", id="load-basis")
                yield RichLog(id="load-log", markup=False, wrap=True)
            with Horizontal(id="cockpit"):
                with TabbedContent(id="tabs"):
                    with TabPane("Architecture", id="tab-arch"):
                        yield BlockMap(id="blockmap")
                    with TabPane("Modules", id="tab-modules"):
                        yield Static("", id="model-summary")
                        yield Tree("model", id="model-tree")
                    with TabPane("Tables", id="tab-tables"):
                        yield DataTable(id="all-params", cursor_type="row", zebra_stripes=True)
                    with TabPane("Run", id="tab-run"):
                        with VerticalScroll(id="run-panel"):
                            with Horizontal(id="run-bar"):
                                yield Input(placeholder="type a prompt, Enter to run it", id="run-prompt")
                                yield Button("▶ Run", id="run-go", disabled=True)
                            yield Static("", id="run-status")
                            yield Static("", id="run-tokens")
                            yield DataTable(id="run-candidates", cursor_type="row", zebra_stripes=True)
                            yield Static("", id="run-meta")
                yield Sidebar(id="sidebar")
        with Horizontal(id="monitor"):
            for name, (label, _) in MONITOR.items():
                with Vertical():
                    yield Label(f"{label} — MOCK", id=f"mon-{name}-label")
                    yield Sparkline(list(self.series[name]), id=f"mon-{name}")
        yield Footer()

    def on_resize(self, event) -> None:
        self.query_one(Sidebar).styles.width = 42 if event.size.width < 110 else 56

    def on_mount(self) -> None:
        # Held, not queried per tick: a tick that lands while the app is shutting
        # down would otherwise find no widgets and raise.
        self.monitor_widgets = {name: (self.query_one(f"#mon-{name}", Sparkline),
                                       self.query_one(f"#mon-{name}-label", Label)) for name in MONITOR}
        self.set_interval(0.5, self.tick_monitor)
        # The loading widgets are held for the same reason: a load worker can
        # still report while the app shuts down.
        self.w_title = self.query_one("#load-title", Label)
        self.w_bar = self.query_one("#load-bar", ProgressBar)
        self.w_basis = self.query_one("#load-basis", Static)
        self.w_log = self.query_one("#load-log", RichLog)
        self.reset_run_panel()
        self.show_state("off")
        if self.startup_model:
            self.start_load(self.startup_model, label_for(self.startup_model), engine=self.startup_engine)

    # ---- the load lifecycle -------------------------------------------------

    def show_state(self, state: str) -> None:
        self.state = state
        self.query_one("#stage", ContentSwitcher).current = {"off": "off", "loaded": "cockpit",
                                                             "running": "cockpit"}.get(state, "loading")
        self.query_one("#monitor").display = state == "loaded" and self.mock
        self.query_one("#menu-selection", Static).update(describe(self.model_ref, state))
        self.query_one("#menu-play", Button).disabled = state != "loaded"
        self.query_one("#run-go", Button).disabled = state != "loaded"
        self.query_one("#menu-stop", Button).disabled = state != "running" or self.run_cancel
        if state == "off":
            self.sub_title = ""

    def start_load(self, ref: str, label: str, engine: Engine | None = None) -> None:
        self.load_token += 1
        # Release the current model before its replacement allocates, so two
        # models' weights are not held at once, and nothing of it stays on screen.
        self.release_model()
        self.model_ref = label
        self.last_event = None
        self.w_title.update(Text.assemble(("loading ", "dim"), (label, "bold")))
        bar = self.w_bar
        bar.update(total=None, progress=0)
        self.w_basis.update(Text("waiting for the first event…", style="dim"))
        self.w_log.clear()
        self.sub_title = f"loading {label}…"
        self.show_state("loading")
        self.load_model(engine, ref, self.load_token)

    @work(thread=True, group="load")
    def load_model(self, engine: Engine | None, ref: str, token: int) -> None:
        def progress(event: LoadEvent) -> None:
            self.call_from_thread(self.on_load_event, token, event)
        try:
            engine = engine or self.engine_factory()  # in the worker: a missing torch is a load failure
            info = engine.load(ref, progress, cancelled=lambda: token != self.load_token)
        except Exception as exc:  # the message is shown, never swallowed
            self.call_from_thread(self.load_failed, token, str(exc))
            return
        self.call_from_thread(self.loaded, token, engine, info)

    def on_load_event(self, token: int, event: LoadEvent) -> None:
        if token != self.load_token:
            return  # a superseded load still reporting; its numbers are not this one's
        previous, self.last_event = self.last_event, event
        bar, basis = self.w_bar, self.w_basis
        if event.total:
            bar.update(total=event.total, progress=event.done or 0)
            basis.update(Text.assemble((f"{event.message}  {event.done:,} / {event.total:,} {event.unit}", "bold"),
                                       ("\n" + event.item if event.item else "", "dim")))
        else:
            bar.update(total=None)  # indeterminate: this phase exposes no count
            basis.update(Text(event.message, style="bold"))
        # The log keeps phase boundaries, not every step.
        first_of_phase = previous is None or previous.phase != event.phase
        if first_of_phase or event.finished:
            count = f"  {event.done:,}/{event.total:,} {event.unit}" if event.total else ""
            self.w_log.write(
                Text.assemble((f"{event.elapsed:7.2f}s  ", "dim"), (f"{event.phase:<8}", "cyan"),
                              f"{event.message}{count}"))

    def load_failed(self, token: int, message: str) -> None:
        if token != self.load_token:
            return
        where = ""
        if self.last_event is not None:
            e = self.last_event
            at = f" at {e.done:,}/{e.total:,} {e.unit}" if e.total else ""
            where = f" during {e.phase}{at}" + (f" ({e.item})" if e.item else "")
        # Everything stays on screen: the bar where it stopped, the log, the error.
        self.w_log.write(Text(f"FAILED{where}: {message}", style="bold red"))
        self.w_title.update(Text.assemble(("load failed: ", "red"), (self.model_ref or "", "bold")))
        self.sub_title = "load failed"
        self.show_state("failed")

    def loaded(self, token: int, engine: Engine, info: ModelInfo) -> None:
        if token != self.load_token:
            _close(engine)  # superseded or closed while loading: never published
            return
        self.engine = engine
        self.info = info
        self.selected = None
        tag = " — MOCK engine" if self.mock else ""
        self.sub_title = f"{info.class_name} · {len(info.modules)} modules · {human(info.param_count)} params{tag}"
        self.query_one(BlockMap).show(info)
        self.reset_sidebar()
        self.show_hierarchy(info)
        table = self.query_one("#all-params", DataTable)
        table.clear(columns=True)
        table.add_columns("module", "tensor", "kind", "shape", "dtype", "count", "bytes")
        self.all_tensor_rows = {}
        for module in info.modules:
            for kind, tensors in (("param", module.params), ("buffer", module.buffers)):
                for t in tensors:
                    table.add_row(module.path or "<root>", t.name, kind, "×".join(map(str, t.shape)),
                                  t.dtype.removeprefix("torch."), f"{t.numel:,}", human(t.bytes),
                                  key=str(len(self.all_tensor_rows)))
                    self.all_tensor_rows[str(len(self.all_tensor_rows))] = (module.path, t.name)
        self.show_state("loaded")
        self.select("")

    def show_hierarchy(self, info: ModelInfo) -> None:
        tag = "MOCK" if self.mock else "Loaded model · no forward pass"
        self.query_one("#model-summary", Static).update(Text(
            f"{tag}\n{info.class_name} · {len(info.modules):,} modules · {info.param_count:,} parameter entries\nModule ownership hierarchy; not an execution trace.", style="dim"))
        tree = self.query_one("#model-tree", Tree)
        tree.clear()
        tree.root.set_label(info.class_name)
        tree.root.data = ""
        nodes = {"": tree.root}
        for module in info.modules:
            if not module.path:
                continue
            parent = module.path.rpartition(".")[0]
            label = Text.assemble((module.path.rsplit(".", 1)[-1], "bold"),
                                  f" · {module.class_name} · {human(info.subtree_param_count(module.path))} params")
            nodes[module.path] = nodes[parent].add(label, data=module.path, allow_expand=bool(info.children(module.path)))
        tree.root.expand()

    def close_model(self) -> None:
        """Unload: invalidate any load or stats in flight, drop every reference, go Off."""
        self.load_token += 1
        self.release_model()
        self.choice = None
        self.model_ref = None
        self.show_state("off")

    def release_model(self) -> None:
        """Close the loaded engine and clear everything drawn from its model."""
        if self.engine is not None:
            _close(self.engine)
        self.engine = None
        self.info = None
        self.selected = None
        self.query_one(BlockMap).remove_children()
        self.reset_sidebar()
        self.query_one("#model-tree", Tree).clear()
        self.query_one("#model-summary", Static).update("")
        self.query_one("#all-params", DataTable).clear(columns=True)
        self.reset_run_panel()

    def reset_sidebar(self) -> None:
        self.inspection_token += 1
        self.selected_tensor = None
        self.tensor_rows = {}
        side = self.query_one(Sidebar)
        side.query_one("#side-parent", Button).disabled = True
        side.query_one("#side-measure", Button).disabled = True
        side.query_one("#side-tensor", Static).update("select a tensor")
        side.query_one("#side-title", Static).update("select a block")
        side.query_one("#side-params", DataTable).clear()
        side.query_one("#side-stats", Static).update("")
        side.query_one("#side-tree", Tree).clear()

    @on(Cell.Selected)
    def select_cell(self, event: Cell.Selected) -> None:
        self.select(event.path)

    def select(self, path: str) -> None:
        if self.info is None:
            return
        module = self.info.module(path)
        if module is None:
            return
        self.selected = path
        self.inspection_token += 1
        self.selected_tensor = None
        self.tensor_rows = {}
        side = self.query_one(Sidebar)
        side.scroll_home(animate=False)
        side.query_one("#side-parent", Button).disabled = path == ""
        side.query_one("#side-measure", Button).disabled = True
        tag = "MOCK" if self.mock else "loaded metadata"
        side.query_one("#side-title", Static).update(Text.assemble(
            (f"{path or '<root>'}\n", "bold"), (f"{module.class_name}\n", KIND_STYLE[kind_of(module)]),
            f"own {module.own_param_count:,} · subtree {self.info.subtree_param_count(path):,} params\n",
            (f"{tag} · no forward pass", "dim"),
            (f"\n{module.description}" if module.description else "", "")))
        table = side.query_one("#side-params", DataTable)
        table.clear(columns=True)
        table.add_column("tensor", width=24 if side.size.width >= 50 else 14)
        table.add_column("kind", width=6)
        table.add_column("shape", width=12)
        prefix = path + "." if path else ""
        owners = [m for m in self.info.modules if m.path == path or m.path.startswith(prefix)]
        for owner in owners:
            for kind, tensors in (("parameter", owner.params), ("buffer", owner.buffers)):
                for tensor in tensors:
                    relative = owner.path[len(prefix):] if owner.path != path else ""
                    address = f"{relative}.{tensor.name}" if relative else tensor.name
                    key = str(len(self.tensor_rows))
                    self.tensor_rows[key] = (owner.path, kind, tensor)
                    table.add_row(address, "param" if kind == "parameter" else "buffer", "×".join(map(str, tensor.shape)) or "scalar", key=key)
        side.query_one("#side-tensor", Static).update(Text("Choose a tensor above to inspect its metadata.", style="dim"))
        side.query_one("#side-stats", Static).update("")
        for cell in self.query(Cell):
            cell.set_class(cell.module_path == path, "selected-block")
        # A leaf has one obvious weight; preserve search → s while parents require a choice.
        if len(self.tensor_rows) == 1:
            self.select_tensor("0")
        tree = side.query_one("#side-tree", Tree)
        tree.clear()
        tree.root.set_label(path.rsplit(".", 1)[-1] or "<root>")
        tree.root.data = path
        self._grow(tree.root, path)
        tree.root.expand()

    def _grow(self, node, path: str) -> None:
        assert self.info is not None
        for child in self.info.children(path):
            leaf = child.path.rsplit(".", 1)[-1]
            label = Text.assemble((f" {leaf} ", KIND_STYLE[kind_of(child)]), f" {child.class_name}")
            kids = self.info.children(child.path)
            branch = node.add(label, data=child.path) if kids else node.add_leaf(label, data=child.path)
            if kids:
                self._grow(branch, child.path)

    @on(Tree.NodeSelected, "#side-tree")
    def drill(self, event: Tree.NodeSelected) -> None:
        if event.node.data is not None and event.node.data != self.selected:
            self.select(event.node.data)

    def action_search(self) -> None:
        box = self.query_one("#search", Input)
        box.add_class("open")
        box.value = ""
        box.focus()

    @on(Input.Submitted, "#search")
    def jump(self, event: Input.Submitted) -> None:
        box = self.query_one("#search", Input)
        box.remove_class("open")
        # A hidden box keeping focus would swallow the next keys (e.g. `s` for stats).
        self.set_focus(None)
        # The "/" that opened the box can arrive in it as text; it is never part of a path.
        needle = event.value.strip().lstrip("/").strip()
        if not needle or self.info is None:
            return
        hit = (self.info.module(needle)
               or next((m for m in self.info.modules if m.path.endswith(needle)), None)
               or next((m for m in self.info.modules if needle in m.path), None))
        if hit is None:
            self.notify(f"no module matches {needle!r}", severity="warning")
            return
        self.query_one("#tabs", TabbedContent).active = "tab-arch"
        self.select(hit.path)

    @on(Tree.NodeSelected, "#model-tree")
    def hierarchy_selected(self, event: Tree.NodeSelected) -> None:
        if event.node.data is not None:
            self.select(event.node.data)

    @on(Button.Pressed, "#side-parent")
    def parent_selected(self) -> None:
        if self.selected:
            self.select(self.selected.rpartition(".")[0])

    @on(DataTable.RowSelected, "#side-params")
    def tensor_selected(self, event: DataTable.RowSelected) -> None:
        self.select_tensor(str(event.row_key.value))

    @on(DataTable.RowSelected, "#all-params")
    def table_selected(self, event: DataTable.RowSelected) -> None:
        owner, name = self.all_tensor_rows[str(event.row_key.value)]
        self.select(owner)
        key = next(k for k, (path, _, tensor) in self.tensor_rows.items() if path == owner and tensor.name == name)
        self.select_tensor(key)

    def select_tensor(self, key: str) -> None:
        if key not in self.tensor_rows:
            return
        path, kind, tensor = self.tensor_rows[key]
        self.inspection_token += 1
        self.selected_tensor = (path, tensor.name)
        self.query_one("#side-params", DataTable).move_cursor(row=int(key))
        address = f"{path}.{tensor.name}" if path else tensor.name
        tag = "MOCK metadata" if self.mock else "loaded tensor metadata"
        self.query_one("#side-tensor", Static).update(Text.assemble(
            (address + "\n", "bold"),
            f"{kind} · {' × '.join(map(str, tensor.shape)) or 'scalar'}\n",
            f"{tensor.dtype.removeprefix('torch.')} · {tensor.device}\n",
            f"{tensor.numel:,} values · {tensor.bytes:,} bytes\n",
            (f"{tag}\nLogical tensor size; not resident memory.", "dim")))
        self.query_one("#side-stats", Static).update("")
        self.query_one("#side-measure", Button).disabled = tensor.numel == 0 or tensor.device == "meta"

    @on(Button.Pressed, "#side-measure")
    def measure_selected(self) -> None:
        self.action_stats()

    def action_stats(self) -> None:
        if self.selected_tensor is None or self.engine is None:
            self.notify("Choose a tensor in the inspector first.")
            return
        self.query_one("#side-stats", Static).update(Text("reading selected tensor values…", style="dim"))
        self.fetch_stats(self.engine, self.load_token, self.inspection_token, *self.selected_tensor)

    @work(thread=True, exclusive=True, group="stats")
    def fetch_stats(self, engine: Engine, token: int, inspection: int, path: str, tensor: str) -> None:
        """Stats for one tensor of `engine`'s model. The answer is published only
        if, by the time it arrives, that engine is still the loaded one (same
        load token) and the same tensor selection is still current: a slow answer from a
        replaced or closed model must never land in the new model's sidebar."""
        try:
            stats = engine.param_stats(path, tensor)
        except (EngineError, Exception) as exc:
            self.call_from_thread(self.publish_stats, engine, token, inspection, path, tensor, Text(str(exc), style="red"))
            return
        bars = "▁▂▃▄▅▆▇█"
        top = max(stats.histogram) or 1
        hist = "".join(bars[min(7, int(7 * h / top))] for h in stats.histogram)
        tag = "  MOCK" if self.mock else ""
        basis = "Synthetic values — MOCK" if self.mock else "Measured values; no forward pass."
        text = Text.assemble((f"{tensor}{tag}\n", "bold"),
                             f"μ {stats.mean:+.4f}  σ {stats.std:.4f}  min {stats.min:+.3f}  max {stats.max:+.3f}\n",
                             f"‖·‖₂ {stats.l2_norm:.2f}  zeros {stats.zero_fraction:.1%}\n{hist}\n16 equal-width bins over [min, max]\n{basis}")
        self.call_from_thread(self.publish_stats, engine, token, inspection, path, tensor, text)

    def publish_stats(self, engine: Engine, token: int, inspection: int, path: str, tensor: str, text: Text) -> None:
        if (token != self.load_token or engine is not self.engine or inspection != self.inspection_token
                or (path, tensor) != self.selected_tensor):
            return
        self.query_one("#side-stats", Static).update(text)

    def action_menu(self) -> None:
        self.push_screen(ModelMenu(can_close=self.state != "off"), self.menu_chosen)

    @on(Button.Pressed, "#menu-model")
    def model_pressed(self) -> None:
        self.action_menu()

    def menu_chosen(self, item: str | None) -> None:
        if item == LOAD_HF:
            self.push_screen(HubPicker(), self.model_chosen)
        elif item == LOAD_DISK:
            self.push_screen(DiskPicker(), self.model_chosen)
        elif item == CLOSE_MODEL:
            self.close_model()

    def model_chosen(self, choice: ModelChoice | None) -> None:
        """Picking a model loads its weights (never runs it). Cancel keeps
        whatever was there before."""
        if choice is None:
            return
        self.choice = choice
        rev = f" @ {choice.revision[:7]}" if choice.revision else ""
        self.start_load(str(choice.folder), f"{choice.label}{rev}")

    # ---- runs ----------------------------------------------------------------

    def reset_run_panel(self) -> None:
        self.query_one("#run-status", Static).update(Text(
            "One forward pass over the prompt, scored at its last token. No generation.", style="dim"))
        self.query_one("#run-tokens", Static).update("")
        self.query_one("#run-candidates", DataTable).clear(columns=True)
        self.query_one("#run-meta", Static).update("")

    @on(Button.Pressed, "#menu-play")
    @on(Button.Pressed, "#run-go")
    @on(Input.Submitted, "#run-prompt")
    def play_pressed(self) -> None:
        self.start_run()

    def start_run(self) -> None:
        if self.state != "loaded" or self.engine is None:
            return
        prompt = self.query_one("#run-prompt", Input).value
        self.query_one("#tabs", TabbedContent).active = "tab-run"
        if not prompt:
            self.query_one("#run-prompt", Input).focus()
            self.notify("Type a prompt in the Run tab, then press Enter or Play.")
            return
        self.run_token += 1
        self.run_cancel = False
        self.reset_run_panel()  # nothing from the previous run is shown as if it belonged to this one
        self.query_one("#run-status", Static).update(Text("running one forward pass…", style="yellow"))
        self.show_state("running")
        self.run_model(self.engine, prompt, RunSettings(), self.load_token, self.run_token)

    @work(thread=True, group="run")
    def run_model(self, engine: Engine, prompt: str, settings: RunSettings, load_token: int, run_token: int) -> None:
        stale = lambda: self.run_cancel or run_token != self.run_token or load_token != self.load_token  # noqa: E731
        try:
            result = engine.run(prompt, settings, cancelled=stale)
        except RunFailed as exc:  # the attempt has an id and, usually, a record
            where = (f"record: {exc.record_path}" if exc.record_path else exc.record_error or "record not saved")
            self.call_from_thread(self.run_failed, load_token, run_token, f"{exc} ({exc.status}, run {exc.run_id}; "
                                  f"{where})", exc.status == "cancelled")
            return
        except Exception as exc:  # shown, never swallowed
            self.call_from_thread(self.run_failed, load_token, run_token, str(exc), False)
            return
        self.call_from_thread(self.run_finished, load_token, run_token, result)

    @on(Button.Pressed, "#menu-stop")
    def stop_pressed(self) -> None:
        if self.state != "running":
            return
        self.run_cancel = True
        self.query_one("#menu-stop", Button).disabled = True
        self.query_one("#run-status", Static).update(Text(
            "Stop requested. A forward pass already in progress can't be interrupted; "
            "it finishes and its result is discarded.", style="yellow"))

    def run_failed(self, load_token: int, run_token: int, message: str, cancelled: bool) -> None:
        if load_token != self.load_token or run_token != self.run_token:
            return
        stopped = cancelled
        self.query_one("#run-status", Static).update(
            Text(f"Stopped: {message}", style="yellow") if stopped else Text(f"Run failed: {message}", style="bold red"))
        self.show_state("loaded")

    def run_finished(self, load_token: int, run_token: int, result: RunResult) -> None:
        if load_token != self.load_token or run_token != self.run_token:
            return  # a run superseded by a newer run, a new model or Close
        if self.run_cancel:
            # Stop was requested after the forward finished but before its result
            # was shown: honour the Stop rather than present a result the user
            # asked to discard.
            self.query_one("#run-status", Static).update(Text(
                "Stopped: the forward pass had already finished; its result was discarded.", style="yellow"))
            self.show_state("loaded")
            return
        mock = result.record_path is None and self.mock
        # The result names the exact prompt it belongs to: the input box stays
        # editable, so its current text can differ from what produced this.
        shown = json.dumps(result.prompt, ensure_ascii=False)
        if len(shown) > 240:
            shown = f"{shown[:240]}… ({len(result.prompt):,} characters; the record holds it exactly)"
        self.query_one("#run-status", Static).update(Text.assemble(
            ("next-token candidates", "bold"), ("  ·  MOCK" if mock else "", "yellow"),
            (f"\nrun {result.run_id} · prompt ", "dim"), shown))
        tokens = Text()
        for token in result.tokens:
            tokens.append(f"[{token.id}]", style="dim")
            tokens.append(json.dumps(token.text, ensure_ascii=False) + "  ")
        self.query_one("#run-tokens", Static).update(Text.assemble(
            (f"{len(result.tokens)} input tokens (exact tokenizer output)\n", "dim"), tokens))
        table = self.query_one("#run-candidates", DataTable)
        table.clear(columns=True)
        table.add_columns("rank", "token", "id", "probability", "logit")
        for rank, c in enumerate(result.candidates, 1):
            table.add_row(str(rank), json.dumps(c.text, ensure_ascii=False), str(c.id), f"{c.probability:.4%}",
                          f"{c.logit:.3f}")
        used = result.used
        lines = []
        if mock:
            lines.append("MOCK engine: synthetic tokens and candidates; this run measures nothing and is not recorded.")
        else:
            lines.append(f"Softmax over the full vocabulary ({used.get('vocab_size', 0):,} tokens) at position "
                         f"{used.get('position')}, computed in {used.get('scored_dtype')}; probabilities are not "
                         "renormalised over the top candidates.")
            lines.append(f"Ran on {used.get('device')} in {used.get('dtype')} · tokenizer {used.get('tokenizer')} · "
                         f"special tokens {'added' if used.get('add_special_tokens') else 'not added'} · "
                         f"chat template {'applied' if used.get('chat_template') else 'not applied'}")
            lines.append("timing: " + "  ".join(f"{k} {v:.3f}s" for k, v in result.timing.items()))
            lines.append(f"record: {result.record_path}" if result.record_path
                         else f"NOT SAVED: {result.record_error}")
        self.query_one("#run-meta", Static).update(Text("\n".join(lines), style="dim"))
        self.show_state("loaded")

    def tick_monitor(self) -> None:
        """Synthetic series until real timing lands; every label says MOCK."""
        for name, series in self.series.items():
            label, base = MONITOR[name]
            nxt = series[-1] + random.uniform(-0.04, 0.04) * base
            series.append(min(max(nxt, 0.8 * base), 1.2 * base))
            spark, text = self.monitor_widgets[name]
            spark.data = list(series)
            text.update(f"{label}  {series[-1]:,.1f}  — MOCK")


def _fake_factory() -> Engine:
    from sememe.engine.fake import FakeEngine
    return FakeEngine(step_delay=0.004)  # slow enough to see the loading screen


def _torch_factory() -> Engine:
    try:
        from sememe.engine.torch_engine import TorchEngine
    except ImportError as exc:  # shown as a load failure, not a crash
        raise EngineError(f"the real engine needs the torch extra ({exc}); try --mock") from exc
    return TorchEngine()


def _close(engine: Engine) -> None:
    close = getattr(engine, "close", None)
    if callable(close):
        close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sememe", description="A debugger for a model's forward pass.")
    parser.add_argument("model", nargs="?", help="load this model snapshot directory or Hub id at startup")
    parser.add_argument("--mock", action="store_true", help="use the built-in fake engine, no torch needed")
    args = parser.parse_args(argv)
    if args.mock:
        Cockpit(model=args.model or "Qwen3.5-0.8B (fake)", mock=True).run()
    else:
        Cockpit(model=args.model).run()  # no model: start on the Off screen


if __name__ == "__main__":
    main()
