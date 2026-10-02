"""The cockpit on the fake engine: no torch, no terminal."""

import pytest

from sememe.engine.fake import FakeEngine, qwen_like_model_info
from sememe.ui.app import Cell, Cockpit, decoder_layers, kind_of


async def started(pilot, app):
    for _ in range(50):
        if app.info is not None:
            return
        await pilot.pause(0.05)
    raise AssertionError("the model never loaded")


@pytest.mark.asyncio
async def test_quit_works_immediately_after_startup():
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await started(pilot, app)
        await pilot.press("q")
        await pilot.pause(0.1)
        assert not app.is_running


def test_the_block_map_reads_the_hybrid_layers_from_the_model():
    layers = decoder_layers(qwen_like_model_info())
    assert len(layers) == 24
    full = [layer.index for layer in layers if kind_of(layer.attention) == "full_attn"]
    assert full == [3, 7, 11, 15, 19, 23]
    assert all(layer.mlp is not None for layer in layers)


@pytest.mark.asyncio
async def test_every_block_is_a_real_module_and_selecting_one_shows_it():
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await started(pilot, app)
        await pilot.pause(0.2)
        cells = list(app.query(Cell))
        assert len(cells) == 1 + 1 + 24 * 2 + 1  # vision, embed, attn+mlp per layer, final norm
        assert all(app.info.module(c.module_path) is not None for c in cells)
        mlp = next(c for c in cells if c.module_path == "language_model.layers.11.mlp")
        mlp.focus()
        await pilot.pause(0.1)
        assert app.selected == "language_model.layers.11.mlp"


@pytest.mark.asyncio
async def test_search_jumps_to_a_module_and_stats_load_for_it():
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await started(pilot, app)
        await pilot.press("slash", *"layers.11.mlp.down_proj", "enter")
        await pilot.pause(0.1)
        assert app.selected == "language_model.layers.11.mlp.down_proj"
        await pilot.press("s")
        await pilot.pause(0.3)
        text = str(app.query_one("#side-stats").render())
        assert "weight" in text and "MOCK" in text and "σ" in text


@pytest.mark.asyncio
async def test_mocked_panels_say_so_and_tables_list_every_tensor():
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await started(pilot, app)
        await pilot.pause(0.6)
        for name in ("prefill", "decode", "latency", "memory"):
            assert "MOCK" in str(app.query_one(f"#mon-{name}-label").render())
        assert "MOCK" in app.sub_title
        tensors = sum(len(m.params) + len(m.buffers) for m in app.info.modules)
        assert app.query_one("#all-params").row_count == tensors


@pytest.mark.asyncio
async def test_a_failed_load_is_shown_not_swallowed():
    class Broken(FakeEngine):
        def load(self, model):
            raise RuntimeError("no such snapshot")

    app = Cockpit(Broken(), "missing", mock=False)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        assert app.info is None
        assert "no such snapshot" in app.sub_title
