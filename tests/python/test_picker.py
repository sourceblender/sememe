"""The Model menu and both model pickers, on a fake cache: no torch, no hub."""

from pathlib import Path

import pytest
from textual.widgets import Button, DataTable, Static

from sememe.engine.fake import FakeEngine
from sememe.sources import choice_for_path, hub_cache_dir, scan_hub_cache
from sememe.ui.app import Cockpit
from sememe.ui.picker import ModelMenu, DiskPicker, HubPicker


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


async def cockpit_with(pilot_size, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(fake_hub(tmp_path)))
    monkeypatch.setenv("HOME", str(tmp_path))
    return Cockpit(FakeEngine(), "fake", mock=True)


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
async def test_choosing_a_cached_model_records_it_without_loading(tmp_path, monkeypatch):
    app = await cockpit_with((160, 50), tmp_path, monkeypatch)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.3)
        assert app.query_one("#menu-play", Button).disabled
        await pilot.press("f10", "enter")  # first entry: Load Hugging Face Model…
        await pilot.pause(0.2)
        assert isinstance(app.screen, HubPicker)
        assert app.screen.query_one("#hub-models", DataTable).row_count == 3
        await pilot.press("enter")  # the highlighted, loadable Qwen row
        await pilot.pause(0.2)
        assert not isinstance(app.screen, HubPicker)
        assert app.choice is not None and app.choice.label == "Qwen/Qwen3.5-0.8B"
        shown = str(app.query_one("#menu-selection", Static).render())
        assert "Qwen/Qwen3.5-0.8B @ abc1234" in shown and "stopped" in shown
        assert not app.query_one("#menu-play", Button).disabled
        assert app.query_one("#menu-stop", Button).disabled
        assert app.model_ref == "fake"  # the mock is still what is loaded


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
        await pilot.pause(0.2)
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
