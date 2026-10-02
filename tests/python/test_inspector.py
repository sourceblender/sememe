"""Selection is a real module/tensor address, shared by every inspection view."""
import threading

import pytest
from textual.widgets import DataTable, Static, TabbedContent, Tree

from sememe.engine.api import ModelInfo, ModuleInfo, TensorInfo, TensorStats
from sememe.engine.fake import FakeEngine
from sememe.ui.app import Cockpit, Cell


async def loaded(pilot, app):
    for _ in range(100):
        if app.state == "loaded":
            await pilot.pause(0.1)
            return
        await pilot.pause(0.05)
    raise AssertionError("not loaded")


@pytest.mark.asyncio
async def test_parent_block_exposes_child_tensors_and_exact_selection():
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await loaded(pilot, app)
        parent = "language_model.layers.0.mlp"
        app.select(parent)
        assert app.query_one("#side-params", DataTable).row_count == 3
        assert app.selected_tensor is None
        key = next(k for k, (owner, _, _) in app.tensor_rows.items() if owner.endswith("down_proj"))
        app.select_tensor(key)
        assert app.selected == parent
        assert app.selected_tensor == (parent + ".down_proj", "weight")
        text = str(app.query_one("#side-tensor", Static).render())
        assert "1024 × 3584" in text and "7,340,032 bytes" in text and "cpu" in text
        app.action_stats()
        await pilot.pause(0.3)
        assert "MOCK" in str(app.query_one("#side-stats", Static).render())
        assert next(c for c in app.query(Cell) if c.module_path == parent).has_class("selected-block")


@pytest.mark.asyncio
async def test_hierarchy_includes_weightless_modules_and_arbitrary_architectures():
    class Tiny(FakeEngine):
        def load(self, model, progress=None, cancelled=None):
            self._info = ModelInfo("CustomModel", modules=(
                ModuleInfo("", "CustomModel"), ModuleInfo("stage", "Sequential"),
                ModuleInfo("stage.activation", "ReLU"),
                ModuleInfo("stage.normalise", "BatchNorm", buffers=(
                    TensorInfo("running_mean", (4,), "torch.float32", 4, 16, "cpu"),)),))
            return self._info
    app = Cockpit(Tiny(), "tiny")
    async with app.run_test(size=(100, 30)) as pilot:
        await loaded(pilot, app)
        tree = app.query_one("#model-tree", Tree)
        def addresses(node):
            return [node.data] + [p for child in node.children for p in addresses(child)]
        assert set(addresses(tree.root)) == {m.path for m in app.info.modules}
        assert [c.module_path for c in app.query(Cell)] == ["stage"]
        app.select("stage.normalise")
        assert app.selected_tensor == ("stage.normalise", "running_mean")
        assert "buffer" in str(app.query_one("#side-tensor", Static).render())
        app.parent_selected()
        assert app.selected == "stage"
        app.parent_selected()
        assert app.selected == ""


@pytest.mark.asyncio
async def test_tensor_table_keyboard_selection_and_table_jump():
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await loaded(pilot, app)
        app.select("language_model.layers.0.mlp")
        table = app.query_one("#side-params", DataTable)
        table.focus()
        await pilot.press("down", "enter")
        assert app.selected_tensor == ("language_model.layers.0.mlp.up_proj", "weight")
        app.query_one("#tabs", TabbedContent).active = "tab-tables"
        await pilot.pause()
        all_table = app.query_one("#all-params", DataTable)
        row = next(int(k) for k, pair in app.all_tensor_rows.items() if pair == ("language_model.layers.0.linear_attn", "A_log"))
        all_table.move_cursor(row=row)
        all_table.focus()
        await pilot.press("enter")
        assert app.selected == "language_model.layers.0.linear_attn"
        assert app.selected_tensor == (app.selected, "A_log")


@pytest.mark.asyncio
async def test_late_stats_cannot_replace_another_tensor_in_same_parent():
    release = threading.Event()
    entered = threading.Event()
    class Slow(FakeEngine):
        def param_stats(self, module, tensor):
            entered.set()
            release.wait(3)
            return TensorStats(0, 1, 111, 1, 1, 0, (1,) * 16)
    app = Cockpit(Slow(), "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await loaded(pilot, app)
        app.select("language_model.layers.0.mlp")
        app.select_tensor("0")
        app.action_stats()
        for _ in range(50):
            if entered.is_set(): break
            await pilot.pause(0.01)
        assert entered.is_set()
        app.select_tensor("1")
        release.set()
        await pilot.pause(0.3)
        assert "111.0000" not in str(app.query_one("#side-stats", Static).render())
        app.close_model()
        assert app.selected_tensor is None


@pytest.mark.asyncio
async def test_narrow_terminal_keeps_both_map_and_inspector_accessible():
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(80, 24)) as pilot:
        await loaded(pilot, app)
        await pilot.pause(0.2)
        app.select("language_model.layers.0.mlp.down_proj")
        sidebar = app.query_one("#sidebar")
        assert sidebar.region.right <= 80
        grid = app.query_one(".layers")
        assert grid.styles.grid_size_columns == 2
        assert app.query_one("#side-params").region.width > 0
        assert sidebar.max_scroll_y > 0
