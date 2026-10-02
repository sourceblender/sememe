"""The Model menu and both model pickers, on a fake cache: no torch, no hub."""

from pathlib import Path

import pytest
from textual.widgets import Button, DataTable, Static

from sememe.engine.fake import FakeEngine
from sememe.sources import choice_for_path, hub_cache_dir, scan_hub_cache
from sememe.ui.app import Cockpit
from sememe.ui.picker import DiskPicker, HubPicker, ModelMenu


def model_folder(folder: Path, config: bool = True, weights: bool = True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if config:
        (folder / "config.json").write_text('{"model_type": "qwen3_5"}')
    if weights:
        (folder / "model.safetensors").write_bytes(b"\0" * 16)
    return folder


def fake_hub(root: Path) -> Path:
    """Three repos: one loadable on main, one without config.json, one with no snapshot."""
    hub = root / "hub"
    qwen = hub / "models--Qwen--Qwen3.5-0.8B"
    model_folder(qwen / "snapshots" / "abc1234def")
    (qwen / "refs").mkdir(parents=True)
    (qwen / "refs" / "main").write_text("abc1234def")
    (qwen / "blobs").mkdir()
    (qwen / "blobs" / "x").write_bytes(b"\0" * 2048)
    model_folder(hub / "models--BAAI--bge-m3" / "snapshots" / "fff0001", config=False)
    (hub / "models--empty--repo" / "blobs").mkdir(parents=True)
    return hub


def test_the_cache_lists_every_snapshot_loadable_first(tmp_path):
    choices = scan_hub_cache(fake_hub(tmp_path))
    assert [c.label for c in choices] == ["Qwen/Qwen3.5-0.8B", "BAAI/bge-m3", "empty/repo"]
    qwen, bge, empty = choices
    assert qwen.loadable and qwen.revision == "abc1234def" and qwen.refs == ("main",) and qwen.size_bytes == 2048
    assert bge.problem == "no config.json"
    assert empty.problem == "no snapshot downloaded"


def test_a_missing_cache_is_empty_not_an_error(tmp_path):
    assert scan_hub_cache(tmp_path / "nowhere") == []


def test_the_cache_location_follows_hf_hub_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "custom"))
    assert hub_cache_dir() == tmp_path / "custom"


def test_picking_a_file_picks_its_model_folder(tmp_path):
    folder = model_folder(tmp_path / "my-model")
    choice = choice_for_path(folder / "model.safetensors")
    assert choice.folder == folder and choice.loadable
    assert choice_for_path(tmp_path).problem == "no config.json"


async def cockpit_with(pilot_size, tmp_path, monkeypatch, factory=FakeEngine):
    """A cockpit started with no model (the Off screen), loading through `factory`."""
    monkeypatch.setenv("HF_HUB_CACHE", str(fake_hub(tmp_path)))
    monkeypatch.setenv("HOME", str(tmp_path))
    return Cockpit(mock=True, engine_factory=factory)


async def until(pilot, check, seconds=3.0):
    for _ in range(int(seconds / 0.05)):
        if check():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition never became true")


def log_text(app) -> str:
    return "\n".join("".join(seg.text for seg in line) for line in app.w_log.lines)


@pytest.mark.asyncio
async def test_f10_opens_the_menu_and_escape_gives_focus_back(tmp_path, monkeypatch):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        before = app.focused
        await pilot.press("f10")
        await pilot.pause(0.1)
        assert isinstance(app.screen, ModelMenu)
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not isinstance(app.screen, ModelMenu)
        assert app.focused is before


@pytest.mark.asyncio
async def test_clicking_model_opens_the_menu(tmp_path, monkeypatch):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.click("#menu-model")
        await pilot.pause(0.1)
        assert isinstance(app.screen, ModelMenu)


@pytest.mark.asyncio
async def test_choosing_a_cached_model_loads_it_with_measured_progress(tmp_path, monkeypatch):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        assert app.state == "off" and app.query_one("#menu-play", Button).disabled
        assert not app.query_one("#monitor").display
        await pilot.press("f10", "enter")  # Load Hugging Face Model…
        await pilot.pause(0.2)
        assert isinstance(app.screen, HubPicker)
        await pilot.press("enter")  # the loadable Qwen row
        await until(pilot, lambda: app.state == "loaded")
        assert app.choice.label == "Qwen/Qwen3.5-0.8B" and app.info is not None
        shown = str(app.query_one("#menu-selection", Static).render())
        assert "Qwen/Qwen3.5-0.8B @ abc1234" in shown and "loaded · stopped" in shown
        assert not app.query_one("#menu-play", Button).disabled
        assert app.query_one("#menu-stop", Button).disabled
        log = log_text(app)
        assert "install" in log and "467/467 parameters" in log and "modules enumerated" in log


@pytest.mark.asyncio
async def test_an_unloadable_entry_explains_itself_and_stays_open(tmp_path, monkeypatch):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("f10", "enter")
        await pilot.pause(0.2)
        await pilot.press("down", "enter")  # bge-m3, no config.json
        await pilot.pause(0.2)
        assert isinstance(app.screen, HubPicker)
        assert app.screen.query_one("#hub-select", Button).disabled
        assert app.choice is None


@pytest.mark.asyncio
async def test_cancel_keeps_the_previous_selection(tmp_path, monkeypatch):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("f10", "enter")
        await pilot.pause(0.2)
        await pilot.press("enter")
        await pilot.pause(0.2)
        first = app.choice
        await pilot.press("f10", "enter")
        await pilot.pause(0.2)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert app.choice is first


@pytest.mark.asyncio
async def test_an_empty_cache_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "nowhere"))
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("f10", "enter")
        await pilot.pause(0.2)
        detail = str(app.screen.query_one("#hub-detail", Static).render())
        assert "No models in the Hugging Face cache" in detail
        assert app.screen.query_one("#hub-select", Button).disabled


@pytest.mark.asyncio
async def test_the_disk_picker_selects_a_model_folder(tmp_path, monkeypatch):
    folder = model_folder(tmp_path / "models" / "my-model")
    app = await cockpit_with((160, 50), tmp_path, monkeypatch)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("f10", "down", "enter")  # Load Model from Disk…
        await pilot.pause(0.3)
        picker = app.screen
        assert isinstance(picker, DiskPicker)
        assert picker.query_one("#disk-select", Button).disabled  # home is not a model folder
        picker.go(folder)
        await pilot.pause(0.3)
        assert not picker.query_one("#disk-select", Button).disabled
        await pilot.click("#disk-select")
        await until(pilot, lambda: app.state == "loaded")
        assert app.choice is not None and app.choice.folder == folder and app.choice.source == "disk"


@pytest.mark.asyncio
async def test_up_goes_to_the_parent_folder(tmp_path, monkeypatch):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("f10", "down", "enter")
        await pilot.pause(0.3)
        picker = app.screen
        picker.go(tmp_path / "hub")
        await pilot.pause(0.2)
        await pilot.click("#disk-up")
        await pilot.pause(0.2)
        assert Path(picker.query_one("#disk-tree").path) == tmp_path


def on_screen(app, selector) -> bool:
    region = app.screen.query_one(selector).region
    width, height = app.size
    return region.width > 0 and region.height > 0 and region.right <= width and region.bottom <= height


@pytest.mark.asyncio
async def test_both_dialogs_fit_an_80_by_24_terminal(tmp_path, monkeypatch):
    app = await cockpit_with((80, 24), tmp_path, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("f10", "enter")
        await pilot.pause(0.2)
        assert all(on_screen(app, s) for s in ("#hub-filter", "#hub-models", "#hub-select", "#hub-cancel"))
        await pilot.press("escape", "f10", "down", "enter")
        await pilot.pause(0.3)
        assert all(on_screen(app, s) for s in ("#disk-up", "#disk-home", "#disk-path", "#disk-tree",
                                               "#disk-select", "#disk-cancel"))


@pytest.fixture
def locked(tmp_path):
    folder = model_folder(tmp_path / "locked")
    folder.chmod(0)
    yield folder
    folder.chmod(0o755)


def test_an_unreadable_folder_is_a_reason_not_a_crash(locked):
    from sememe.sources import model_problem
    assert model_problem(locked).startswith("can't read it")


def test_an_unreadable_snapshot_is_listed_with_its_reason(tmp_path):
    hub = tmp_path / "hub"
    snapshots = hub / "models--org--locked" / "snapshots"
    model_folder(snapshots / "rev1")
    snapshots.chmod(0)
    try:
        choices = scan_hub_cache(hub)
    finally:
        snapshots.chmod(0o755)
    assert [c.label for c in choices] == ["org/locked"]
    assert choices[0].problem.startswith("can't read it")


@pytest.mark.asyncio
async def test_browsing_into_an_unreadable_folder_explains_and_stays_put(tmp_path, monkeypatch, locked):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("f10", "down", "enter")
        await pilot.pause(0.3)
        picker = app.screen
        before = Path(picker.query_one("#disk-tree").path)
        picker.go(locked)
        await pilot.pause(0.2)
        assert Path(picker.query_one("#disk-tree").path) == before
        assert "can't read it" in str(picker.query_one("#disk-detail", Static).render())
        picker.go(tmp_path / "gone")  # a stale path
        await pilot.pause(0.2)
        assert "not a folder" in str(picker.query_one("#disk-detail", Static).render())


def test_a_ctranslate2_folder_is_named_not_offered(tmp_path):
    from sememe.sources import model_problem
    folder = tmp_path / "faster-whisper"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    (folder / "model.bin").write_bytes(b"\0")
    assert model_problem(folder) == "CTranslate2 format, not Transformers"


def test_weights_must_be_real_files(tmp_path):
    from sememe.sources import model_problem
    folder = model_folder(tmp_path / "m", weights=False)
    (folder / "model.safetensors").symlink_to(tmp_path / "missing")  # dangling
    (folder / "shard.safetensors").mkdir()  # a folder with a weight name
    assert model_problem(folder) == "no Transformers weight files"


def test_a_config_without_model_type_is_not_transformers(tmp_path):
    from sememe.sources import model_problem
    folder = model_folder(tmp_path / "m")
    (folder / "config.json").write_text('{"hidden": 1}')
    assert "no model_type" in model_problem(folder)


@pytest.mark.asyncio
async def test_the_cockpit_keeps_running_while_a_dialog_stays_open(tmp_path, monkeypatch):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("f10", "enter")
        await pilot.pause(1.6)  # several monitor ticks with the picker on top
        assert app.is_running and isinstance(app.screen, HubPicker)



@pytest.mark.asyncio
async def test_with_no_model_it_starts_off_with_the_title_art(tmp_path, monkeypatch):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.2)
        assert app.state == "off" and app.info is None
        assert app.query_one("#stage").current == "off"
        assert "debugger for a model" in str(app.query_one("#off", Static).render())
        await pilot.press("f10")
        await pilot.pause(0.1)
        close = app.screen.query_one("#model-options").get_option("close-model")
        assert close.disabled  # nothing to close


@pytest.mark.asyncio
async def test_a_failure_keeps_where_it_stopped_and_why(tmp_path, monkeypatch):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch, factory=lambda: FakeEngine(fail_at=120))
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("f10", "enter")
        await pilot.pause(0.2)
        await pilot.press("enter")
        await until(pilot, lambda: app.state == "failed")
        assert app.info is None and app.query_one("#menu-play", Button).disabled
        assert app.w_bar.progress == 119 and app.w_bar.total == 467  # the bar stays where it stopped
        log = log_text(app)
        assert "FAILED during install at 119/467 parameters" in log and "mock failure installing" in log
        assert "load failed" in str(app.query_one("#menu-selection", Static).render())


@pytest.mark.asyncio
async def test_close_unloads_and_returns_to_off(tmp_path, monkeypatch):
    engines = []

    def factory():
        engines.append(FakeEngine())
        return engines[-1]

    app = await cockpit_with((160, 50), tmp_path, monkeypatch, factory=factory)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("f10", "enter")
        await pilot.pause(0.2)
        await pilot.press("enter")
        await until(pilot, lambda: app.state == "loaded")
        await pilot.press("f10", "down", "down", "enter")  # Close Model
        await until(pilot, lambda: app.state == "off")
        assert app.engine is None and app.info is None and app.choice is None
        assert engines[0]._info is None  # the engine dropped its model too
        assert app.query_one("#stage").current == "off"
        assert app.query_one("#menu-play", Button).disabled


@pytest.mark.asyncio
async def test_a_superseded_load_never_publishes(tmp_path, monkeypatch):
    import threading

    release = threading.Event()

    class Slow(FakeEngine):
        def load(self, model, progress=None, cancelled=None):
            release.wait(5)
            return super().load(model, progress)  # deliberately ignores cancelled: the publish guard alone

    engines = []

    def factory():
        engines.append(Slow() if not engines else FakeEngine())
        return engines[-1]

    folder = model_folder(tmp_path / "models" / "second")
    app = await cockpit_with((160, 50), tmp_path, monkeypatch, factory=factory)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("f10", "enter")
        await pilot.pause(0.2)
        await pilot.press("enter")  # first load: blocks
        await until(pilot, lambda: app.state == "loading")
        from sememe.sources import choice_for_path
        app.model_chosen(choice_for_path(folder))  # a second pick while the first is still loading
        await until(pilot, lambda: app.state == "loaded")
        assert app.engine is engines[1]
        release.set()  # now let the first one finish
        await pilot.pause(0.5)
        assert app.engine is engines[1] and app.model_ref == "second"
        assert engines[0]._info is None  # its late result was closed, not published



def test_startup_labels_are_short():
    from sememe.sources import label_for
    snap = "/x/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f/"
    assert label_for(snap) == "Qwen/Qwen3.5-0.8B @ 2fc0636"
    assert label_for("/data/models/my-model") == "my-model"
    assert label_for("Qwen/Qwen3.5-0.8B") == "Qwen/Qwen3.5-0.8B"


async def pick_qwen(pilot):
    await pilot.press("f10", "enter")
    await pilot.pause(0.2)
    await pilot.press("enter")


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["config", "install"])
async def test_a_failure_is_blamed_on_the_phase_it_happened_in(tmp_path, monkeypatch, phase):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch, factory=lambda: FakeEngine(fail_in=phase))
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pick_qwen(pilot)
        await until(pilot, lambda: app.state == "failed")
        assert f"FAILED during {phase}: mock failure in {phase}" in log_text(app)
        assert app.w_bar.total is None  # no count was ever reported, so no fraction is shown


@pytest.mark.asyncio
async def test_a_slow_answer_from_a_replaced_model_never_reaches_the_new_sidebar(tmp_path, monkeypatch):
    import threading
    from sememe.engine.api import TensorStats

    release = threading.Event()

    class SlowStats(FakeEngine):
        def param_stats(self, module, tensor):
            release.wait(5)
            return TensorStats(min=0, max=1, mean=111.0, std=1, l2_norm=1, zero_fraction=0, histogram=(1,) * 16)

    engines = []

    def factory():
        engines.append(SlowStats() if not engines else FakeEngine())
        return engines[-1]

    app = await cockpit_with((160, 50), tmp_path, monkeypatch, factory=factory)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pick_qwen(pilot)
        await until(pilot, lambda: app.state == "loaded")
        path = "language_model.layers.11.mlp.down_proj"
        app.select(path)
        await pilot.press("s")  # the old model's stats: blocked
        await pilot.pause(0.1)
        await pick_qwen(pilot)  # replace the model
        await until(pilot, lambda: app.state == "loaded" and app.engine is engines[1])
        app.select(path)  # same module path on the new model
        release.set()
        await pilot.pause(0.4)
        assert "111.0000" not in str(app.query_one("#side-stats", Static).render())


@pytest.mark.asyncio
async def test_the_old_model_is_released_before_its_replacement_loads(tmp_path, monkeypatch):
    import threading

    release = threading.Event()

    class Slow(FakeEngine):
        def load(self, model, progress=None, cancelled=None):
            release.wait(5)
            return super().load(model, progress, cancelled)

    engines = []

    def factory():
        engines.append(FakeEngine() if not engines else Slow())
        return engines[-1]

    app = await cockpit_with((160, 50), tmp_path, monkeypatch, factory=factory)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pick_qwen(pilot)
        await until(pilot, lambda: app.state == "loaded")
        await pick_qwen(pilot)  # second load blocks
        await until(pilot, lambda: app.state == "loading")
        assert engines[0]._info is None and app.engine is None and app.info is None  # released already
        release.set()
        await until(pilot, lambda: app.state == "loaded")
        assert app.engine is engines[1]


@pytest.mark.asyncio
async def test_a_superseded_load_stops_before_it_allocates(tmp_path, monkeypatch):
    import threading

    gate = threading.Event()
    reached = []

    class Waiting(FakeEngine):
        def load(self, model, progress=None, cancelled=None):
            gate.wait(5)  # like waiting behind the install lock
            try:
                return super().load(model, progress, cancelled)
            except Exception as exc:
                reached.append(str(exc))
                raise

    engines = []

    def factory():
        engines.append(Waiting() if not engines else FakeEngine())
        return engines[-1]

    folder = model_folder(tmp_path / "models" / "second")
    app = await cockpit_with((160, 50), tmp_path, monkeypatch, factory=factory)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        await pick_qwen(pilot)
        await until(pilot, lambda: app.state == "loading")
        from sememe.sources import choice_for_path
        app.model_chosen(choice_for_path(folder))
        await until(pilot, lambda: app.state == "loaded")
        gate.set()
        await pilot.pause(0.3)
        assert reached and reached[0].startswith("cancelled before config")
        assert engines[0]._info is None and app.engine is engines[1]


@pytest.mark.asyncio
async def test_the_title_art_renders_as_one_aligned_block(tmp_path, monkeypatch):
    """Rendered on screen, every art row keeps its own column offset: the rows
    move together or not at all. Centring rows of unequal width broke this."""
    from sememe.ui.app import _TITLE_ROWS
    app = await cockpit_with((100, 30), tmp_path, monkeypatch)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        off = app.query_one("#off")
        lines = ["".join(seg.text for seg in strip) for strip in off.render_lines(off.region.reset_offset)]
        rows = [line for line in lines if "█" in line]
        assert len(rows) == len(_TITLE_ROWS)
        indent = lambda text: len(text) - len(text.lstrip(" "))  # noqa: E731
        shifts = {indent(drawn) - indent(source) for drawn, source in zip(rows, _TITLE_ROWS)}
        assert len(shifts) == 1, f"rows shifted against each other by {sorted(shifts)}"
