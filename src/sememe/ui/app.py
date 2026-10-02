"""The sememe cockpit: a Textual app over any `Engine`.

The UI never touches torch. It asks the engine through `sememe.engine.api` and
makes every blocking call from a thread worker, so the screen stays live while
a model loads. Panels whose data is not real yet say MOCK in their title.
"""

from __future__ import annotations

import argparse
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
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
    Tree,
)

from sememe.engine.api import Engine, EngineError, ModelInfo, ModuleInfo
from sememe.sources import ModelChoice
from sememe.ui.picker import LOAD_DISK, LOAD_HF, ActivityMenu, DiskPicker, HubPicker, MenuBar, describe

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

    def show(self, info: ModelInfo) -> None:
        self.remove_children()
        widgets: list = []
        vision = info.module("visual")
        if vision is not None:
            widgets.append(Label("vision tower", classes="section"))
            widgets.append(Cell(f" {vision.class_name} · {human(info.subtree_param_count('visual'))} ▸ ", vision))
        widgets.append(Label("language model", classes="section"))
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


class Sidebar(Vertical):
    """What the selected module is, what it owns, and its subtree."""

    def compose(self) -> ComposeResult:
        yield Static("select a block", id="side-title")
        yield DataTable(id="side-params", cursor_type="row")
        yield Static("", id="side-stats")
        yield Tree("subtree", id="side-tree")

    def on_mount(self) -> None:
        table = self.query_one("#side-params", DataTable)
        table.add_columns("tensor", "shape", "dtype", "count")


class Cockpit(App):
    # Search starts hidden; it should only take focus when explicitly opened.
    AUTO_FOCUS = None

    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; }
    BlockMap { width: 1fr; padding: 0 1; }
    Sidebar { width: 56; border-left: solid $accent; padding: 0 1; }
    .section { color: $text-muted; margin-top: 1; }
    .layers { grid-size: 6; grid-gutter: 0 1; height: auto; }
    .layer { height: 4; border: round $panel-lighten-2; }
    .layer-title { color: $text-muted; }
    .layer-row { height: 1; }
    .layer-row Cell { margin-right: 1; }
    #tabs { height: 1fr; }
    Cell { width: auto; height: 1; }
    Cell:focus { text-style: reverse; }
    .legend { margin-top: 1; }
    #side-title { height: auto; margin-bottom: 1; }
    #side-params { height: auto; max-height: 12; }
    #side-stats { height: auto; margin: 1 0; }
    #side-tree { height: 1fr; }
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
    """
    BINDINGS = [
        Binding("f10", "menu", "menu"),
        Binding("slash", "search", "search"),
        Binding("s", "stats", "weight stats"),
        Binding("q", "quit", "quit"),
    ]
    TITLE = "sememe"

    def __init__(self, engine: Engine, model: str, mock: bool) -> None:
        super().__init__()
        self.engine = engine
        self.model_ref = model
        self.mock = mock
        self.info: ModelInfo | None = None
        self.selected: str | None = None
        self.choice: ModelChoice | None = None  # picked in the menu, not loaded
        self.series = {name: deque([base] * 60, maxlen=60) for name, (_, base) in MONITOR.items()}

    def compose(self) -> ComposeResult:
        yield Header()
        yield MenuBar(id="menubar")
        yield Input(placeholder="jump to module path…", id="search")
        with TabbedContent(id="tabs"):
            with TabPane("Architecture", id="tab-arch"):
                with Horizontal(id="main"):
                    yield BlockMap(id="blockmap")
                    yield Sidebar(id="sidebar")
            with TabPane("Tables", id="tab-tables"):
                yield DataTable(id="all-params", cursor_type="row", zebra_stripes=True)
            with TabPane("Decode — MOCK", id="tab-decode"):
                yield Static(id="decode")
        with Horizontal(id="monitor"):
            for name, (label, _) in MONITOR.items():
                with Vertical():
                    yield Label(f"{label} — MOCK", id=f"mon-{name}-label")
                    yield Sparkline(list(self.series[name]), id=f"mon-{name}")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"loading {self.model_ref}…"
        self.load_model()
        # Held, not queried per tick: a tick that lands while the app is shutting
        # down would otherwise find no widgets and raise.
        self.monitor_widgets = {name: (self.query_one(f"#mon-{name}", Sparkline),
                                       self.query_one(f"#mon-{name}-label", Label)) for name in MONITOR}
        self.set_interval(0.5, self.tick_monitor)
        self.query_one("#decode", Static).update(Text.assemble(
            ("Decode — MOCK\n\n", "bold"), ("> The cat sat on the\n\n", ""),
            ("next token   mat 0.61 · floor 0.12 · couch 0.07\n", "green"),
            ("\nA real prompt run arrives in a later slice.", "dim")))

    @work(thread=True, exclusive=True)
    def load_model(self) -> None:
        try:
            info = self.engine.load(self.model_ref)
        except Exception as exc:  # the message is shown, never swallowed
            self.call_from_thread(self.load_failed, str(exc))
            return
        self.call_from_thread(self.loaded, info)

    def load_failed(self, message: str) -> None:
        self.sub_title = f"load failed: {message}"
        self.notify(message, title="load failed", severity="error", timeout=30)

    def loaded(self, info: ModelInfo) -> None:
        self.info = info
        tag = " — MOCK engine" if self.mock else ""
        self.sub_title = f"{info.class_name} · {len(info.modules)} modules · {human(info.param_count)} params{tag}"
        self.query_one(BlockMap).show(info)
        table = self.query_one("#all-params", DataTable)
        table.add_columns("module", "tensor", "kind", "shape", "dtype", "count", "bytes")
        for module in info.modules:
            for kind, tensors in (("param", module.params), ("buffer", module.buffers)):
                for t in tensors:
                    table.add_row(module.path or "<root>", t.name, kind, "×".join(map(str, t.shape)),
                                  t.dtype.removeprefix("torch."), f"{t.numel:,}", human(t.bytes))

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
        side = self.query_one(Sidebar)
        side.query_one("#side-title", Static).update(Text.assemble(
            (f"{path or '<root>'}\n", "bold"), (f"{module.class_name}", KIND_STYLE[kind_of(module)]),
            f"  ·  own {human(module.own_param_count)}  ·  subtree {human(self.info.subtree_param_count(path))}"))
        table = side.query_one("#side-params", DataTable)
        table.clear()
        for t in module.params + module.buffers:
            table.add_row(t.name, "×".join(map(str, t.shape)), t.dtype.removeprefix("torch."), human(t.numel))
        side.query_one("#side-stats", Static).update(
            Text("press s for weight stats" if module.params else "no parameters of its own", style="dim"))
        tree = side.query_one("#side-tree", Tree)
        tree.clear()
        tree.root.set_label(path.rsplit(".", 1)[-1] or "<root>")
        tree.root.data = path
        self._grow(tree.root, path, depth=0)
        tree.root.expand()

    def _grow(self, node, path: str, depth: int) -> None:
        assert self.info is not None
        for child in self.info.children(path):
            leaf = child.path.rsplit(".", 1)[-1]
            label = Text.assemble((f" {leaf} ", KIND_STYLE[kind_of(child)]), f" {child.class_name}")
            kids = self.info.children(child.path)
            branch = node.add(label, data=child.path) if kids else node.add_leaf(label, data=child.path)
            if kids and depth < 2:
                self._grow(branch, child.path, depth + 1)

    @on(Tree.NodeSelected, "#side-tree")
    def drill(self, event: Tree.NodeSelected) -> None:
        if event.node.data and event.node.data != self.selected:
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

    def action_stats(self) -> None:
        if self.selected is None or self.info is None:
            return
        module = self.info.module(self.selected)
        if module is None or not module.params:
            return
        self.query_one("#side-stats", Static).update(Text("reading weights…", style="dim"))
        self.fetch_stats(module.path, module.params[0].name)

    @work(thread=True, exclusive=True, group="stats")
    def fetch_stats(self, path: str, tensor: str) -> None:
        try:
            stats = self.engine.param_stats(path, tensor)
        except (EngineError, Exception) as exc:
            self.call_from_thread(self.query_one("#side-stats", Static).update, Text(str(exc), style="red"))
            return
        bars = "▁▂▃▄▅▆▇█"
        top = max(stats.histogram) or 1
        hist = "".join(bars[min(7, int(7 * h / top))] for h in stats.histogram)
        tag = "  MOCK" if self.mock else ""
        text = Text.assemble((f"{tensor}{tag}\n", "bold"),
                             f"μ {stats.mean:+.4f}  σ {stats.std:.4f}  min {stats.min:+.3f}  max {stats.max:+.3f}\n",
                             f"‖·‖₂ {stats.l2_norm:.2f}  zeros {stats.zero_fraction:.1%}  {hist}")
        if path == self.selected:
            self.call_from_thread(self.query_one("#side-stats", Static).update, text)

    def action_menu(self) -> None:
        self.push_screen(ActivityMenu(), self.menu_chosen)

    @on(Button.Pressed, "#menu-activity")
    def activity_pressed(self) -> None:
        self.action_menu()

    def menu_chosen(self, item: str | None) -> None:
        if item == LOAD_HF:
            self.push_screen(HubPicker(), self.model_chosen)
        elif item == LOAD_DISK:
            self.push_screen(DiskPicker(), self.model_chosen)

    def model_chosen(self, choice: ModelChoice | None) -> None:
        """Record the pick. Cancel (None) keeps whatever was selected before."""
        if choice is None:
            return
        self.choice = choice
        self.query_one("#menu-selection", Static).update(describe(choice))
        self.query_one("#menu-play", Button).disabled = False

    @on(Button.Pressed, "#menu-play")
    def play_pressed(self) -> None:
        self.notify("Starting a model arrives in the next slice. Nothing is loaded; the cockpit still shows the mock.",
                    title="Play", timeout=6)

    def tick_monitor(self) -> None:
        """Synthetic series until real timing lands; every label says MOCK."""
        for name, series in self.series.items():
            label, base = MONITOR[name]
            nxt = series[-1] + random.uniform(-0.04, 0.04) * base
            series.append(min(max(nxt, 0.8 * base), 1.2 * base))
            spark, text = self.monitor_widgets[name]
            spark.data = list(series)
            text.update(f"{label}  {series[-1]:,.1f}  — MOCK")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sememe", description="A debugger for a model's forward pass.")
    parser.add_argument("model", nargs="?", help="model snapshot directory or Hub id")
    parser.add_argument("--mock", action="store_true", help="run on the built-in fake engine, no torch needed")
    args = parser.parse_args(argv)
    if args.mock or not args.model:
        from sememe.engine.fake import FakeEngine
        engine: Engine = FakeEngine()
        model, mock = args.model or "Qwen3.5-0.8B (fake)", True
    else:
        try:
            from sememe.engine.torch_engine import TorchEngine
        except ImportError as exc:
            parser.error(f"the real engine needs the torch extra ({exc}); try --mock")
        engine, model, mock = TorchEngine(), args.model, False
    Cockpit(engine, model, mock).run()


if __name__ == "__main__":
    main()
