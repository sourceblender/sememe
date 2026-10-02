"""The menu row and the two model pickers.

Picking a model only records the choice. Play is what will start it, in a
later slice; until then the cockpit keeps showing the mock behind the dialogs.
Every dialog is a modal screen: Escape or Cancel dismisses it with None, and
Textual hands focus back to whatever had it before the dialog opened.
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, DirectoryTree, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from sememe.sources import ModelChoice, choice_for_path, human_bytes, hub_cache_dir, listing_problem, scan_hub_cache

LOAD_HF = "load-hf"
LOAD_DISK = "load-disk"


def describe(choice: ModelChoice | None) -> Text:
    """The one line the menu row shows for the current selection."""
    if choice is None:
        return Text("no model selected", style="dim")
    rev = f" @ {choice.revision[:7]}" if choice.revision else ""
    return Text.assemble(("selected: ", "dim"), (f"{choice.label}{rev}", "bold"), ("  ·  stopped", "dim"))


class MenuBar(Horizontal):
    """One row under the title: the Model menu on the left, the selected
    model and its Play / Stop controls on the right."""

    def compose(self) -> ComposeResult:
        yield Button("Model ▾", id="menu-model", classes="menu")
        yield Static("", id="menu-spacer")
        yield Static(describe(None), id="menu-selection")
        yield Button("▶ Play", id="menu-play", classes="menu", disabled=True)
        yield Button("■ Stop", id="menu-stop", classes="menu", disabled=True)


class ModelMenu(ModalScreen[str | None]):
    """The dropdown under `Model ▾`."""

    BINDINGS = [Binding("escape", "dismiss(None)", "close")]
    DEFAULT_CSS = """
    ModelMenu { align: left top; background: transparent; }
    ModelMenu OptionList { width: 32; height: auto; margin: 2 0 0 0; border: round $accent; }
    """

    def compose(self) -> ComposeResult:
        yield OptionList(Option("Load Hugging Face Model…", id=LOAD_HF),
                         Option("Load Model from Disk…", id=LOAD_DISK), id="model-options")

    def on_mount(self) -> None:
        # Opened by keyboard, Enter must act on something: start on the first entry.
        options = self.query_one(OptionList)
        options.highlighted = 0
        options.focus()

    @on(OptionList.OptionSelected)
    def chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def on_click(self, event) -> None:
        # A click outside the list closes the menu, like any dropdown.
        if getattr(event, "widget", None) is self:
            self.dismiss(None)


DIALOG_CSS = """
{name} {{ align: center middle; }}
{name} > Vertical {{ width: 94%; max-width: 110; height: 94%; max-height: 34; border: round $accent;
                    background: $surface; padding: 0 1; }}
{name} .dialog-title {{ text-style: bold; }}
{name} .dialog-note {{ color: $text-muted; }}
{name} .dialog-detail {{ height: 3; }}
{name} .dialog-buttons {{ height: 3; align: right middle; }}
{name} .dialog-buttons Button {{ margin-left: 1; }}
"""


class HubPicker(ModalScreen[ModelChoice | None]):
    """Every model snapshot already in the local Hugging Face cache."""

    BINDINGS = [Binding("escape", "dismiss(None)", "cancel")]
    DEFAULT_CSS = DIALOG_CSS.format(name="HubPicker") + """
    HubPicker DataTable { height: 1fr; }
    """

    def __init__(self, hub: Path | None = None) -> None:
        super().__init__()
        self.hub = hub or hub_cache_dir()
        self.choices = scan_hub_cache(self.hub)
        self.shown: list[ModelChoice] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Load Hugging Face Model", classes="dialog-title")
            yield Static(f"cache: {self.hub}", classes="dialog-note")
            yield Input(placeholder="filter by name…", id="hub-filter")
            yield DataTable(id="hub-models", cursor_type="row", zebra_stripes=True)
            yield Static("", id="hub-detail", classes="dialog-detail")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Select", id="hub-select", variant="primary", disabled=True)
                yield Button("Cancel", id="hub-cancel")

    def on_mount(self) -> None:
        table = self.query_one("#hub-models", DataTable)
        table.add_columns("model", "revision", "refs", "size", "status")
        self.fill("")
        table.focus()

    def fill(self, needle: str) -> None:
        table = self.query_one("#hub-models", DataTable)
        table.clear()
        self.shown = [c for c in self.choices if needle.lower() in c.label.lower()]
        for c in self.shown:
            style = "" if c.loadable else "dim"
            table.add_row(Text(c.label, style=style), Text((c.revision or "—")[:7], style=style),
                          Text(", ".join(c.refs), style=style), Text(human_bytes(c.size_bytes), style=style),
                          Text("files present" if c.loadable else c.problem or "", style="green" if c.loadable else "dim"))
        if not self.choices:
            self.show_detail(None, f"No models in the Hugging Face cache at {self.hub}.")
        elif not self.shown:
            self.show_detail(None, f"No cached model matches {needle!r}.")
        else:
            self.show_detail(self.shown[0])

    def show_detail(self, choice: ModelChoice | None, message: str = "") -> None:
        detail = self.query_one("#hub-detail", Static)
        self.query_one("#hub-select", Button).disabled = choice is None or not choice.loadable
        if choice is None:
            detail.update(Text(message, style="dim"))
            return
        status = (Text("Transformers files present; whether it loads is checked when you press Play", style="green")
                  if choice.loadable else Text(f"can't be loaded: {choice.problem}", style="yellow"))
        # Status before path: on a narrow terminal the long snapshot path wraps,
        # and the reason a model can or can't load must never be what gets clipped.
        rev = f"  @ {choice.revision[:12]}" if choice.revision else ""
        refs = f"  ({', '.join(choice.refs)})" if choice.refs else ""
        detail.update(Text.assemble((choice.label, "bold"), (f"{rev}{refs}", "dim"), "\n", status, "\n",
                                    (str(choice.folder), "dim")))

    def highlighted(self) -> ModelChoice | None:
        row = self.query_one("#hub-models", DataTable).cursor_row
        return self.shown[row] if 0 <= row < len(self.shown) else None

    @on(Input.Changed, "#hub-filter")
    def filter_changed(self, event: Input.Changed) -> None:
        self.fill(event.value)

    @on(Input.Submitted, "#hub-filter")
    def filter_done(self) -> None:
        self.query_one("#hub-models", DataTable).focus()

    @on(DataTable.RowHighlighted, "#hub-models")
    def row_moved(self) -> None:
        self.show_detail(self.highlighted())

    @on(DataTable.RowSelected, "#hub-models")
    def row_chosen(self) -> None:
        self.pick()

    @on(Button.Pressed, "#hub-select")
    def select_pressed(self) -> None:
        self.pick()

    @on(Button.Pressed, "#hub-cancel")
    def cancel_pressed(self) -> None:
        self.dismiss(None)

    def pick(self) -> None:
        choice = self.highlighted()
        if choice is None:
            return
        if not choice.loadable:
            self.notify(f"{choice.label} can't be loaded: {choice.problem}", severity="warning")
            return
        self.dismiss(choice)


class DiskPicker(ModalScreen[ModelChoice | None]):
    """Browse the disk for a model folder. A file stands for its folder."""

    BINDINGS = [Binding("escape", "dismiss(None)", "cancel")]
    DEFAULT_CSS = DIALOG_CSS.format(name="DiskPicker") + """
    DiskPicker .disk-nav { height: 3; }
    DiskPicker .disk-nav Input { width: 1fr; }
    DiskPicker DirectoryTree { height: 1fr; }
    """

    def __init__(self, start: Path | None = None) -> None:
        super().__init__()
        self.start = (start or Path.home()).expanduser()
        self.current: ModelChoice | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Load Model from Disk", classes="dialog-title")
            yield Static("A model folder holds config.json and weights; a file picks its folder.",
                         classes="dialog-note")
            with Horizontal(classes="disk-nav"):
                yield Button("↑ Up", id="disk-up")
                yield Button("⌂ Home", id="disk-home")
                yield Input(str(self.start), id="disk-path")
            yield DirectoryTree(self.start, id="disk-tree")
            yield Static("", id="disk-detail", classes="dialog-detail")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Select", id="disk-select", variant="primary", disabled=True)
                yield Button("Cancel", id="disk-cancel")

    def on_mount(self) -> None:
        self.inspect(self.start)
        self.query_one("#disk-tree", DirectoryTree).focus()

    def go(self, path: Path) -> None:
        path = path.expanduser()
        problem = listing_problem(path)
        if problem is not None:
            # Stay where we are and say why, in the dialog itself.
            self.query_one("#disk-detail", Static).update(
                Text.assemble(("can't open ", "dim"), (str(path), "bold"), "\n", Text(problem, style="yellow")))
            return
        self.query_one("#disk-tree", DirectoryTree).path = path
        self.query_one("#disk-path", Input).value = str(path)
        self.inspect(path)

    def inspect(self, path: Path) -> None:
        self.current = choice_for_path(path)
        detail = self.query_one("#disk-detail", Static)
        status = (Text("Transformers files present; whether it loads is checked at Play", style="green") if self.current.loadable
                  else Text(f"not a model folder: {self.current.problem}", style="dim"))
        detail.update(Text.assemble(("selecting ", "dim"), (str(self.current.folder), "bold"), "\n", status))
        self.query_one("#disk-select", Button).disabled = not self.current.loadable

    @on(DirectoryTree.NodeHighlighted, "#disk-tree")
    def moved(self, event: DirectoryTree.NodeHighlighted) -> None:
        entry = event.node.data
        if entry is not None:
            self.inspect(Path(entry.path))

    @on(DirectoryTree.FileSelected, "#disk-tree")
    def file_chosen(self, event: DirectoryTree.FileSelected) -> None:
        self.inspect(event.path)
        self.pick()

    @on(Button.Pressed, "#disk-up")
    def up(self) -> None:
        self.go(Path(self.query_one("#disk-tree", DirectoryTree).path).parent)

    @on(Button.Pressed, "#disk-home")
    def home(self) -> None:
        self.go(Path.home())

    @on(Input.Submitted, "#disk-path")
    def typed(self, event: Input.Submitted) -> None:
        self.go(Path(event.value))

    @on(Button.Pressed, "#disk-select")
    def select_pressed(self) -> None:
        self.pick()

    @on(Button.Pressed, "#disk-cancel")
    def cancel_pressed(self) -> None:
        self.dismiss(None)

    def pick(self) -> None:
        if self.current is None:
            return
        if not self.current.loadable:
            self.notify(f"{self.current.folder} isn't a model folder: {self.current.problem}", severity="warning")
            return
        self.dismiss(self.current)
