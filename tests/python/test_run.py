"""Controlled Play on the fake engine: the Run tab, Stop, and stale results."""

import threading

import pytest
from textual.widgets import Button, DataTable, Input, Static

from sememe.engine.fake import FakeEngine
from sememe.ui.app import Cockpit


async def until(pilot, check, seconds=3.0):
    for _ in range(int(seconds / 0.05)):
        if check():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition never became true")


def text(app, selector) -> str:
    return str(app.query_one(selector, Static).render())


class Gated(FakeEngine):
    """A fake whose run waits until the test releases it."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def run(self, prompt, settings, cancelled=None):
        self.release.wait(5)
        return super().run(prompt, settings, cancelled)


class PastTheLastCheck(Gated):
    """A run already past its last cancellation check: it returns a result no
    matter what. Only the UI's own guard can keep that result off the screen."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()

    def run(self, prompt, settings, cancelled=None):
        # Like TorchEngine, which holds local references to its model: the
        # forward completes even if the engine is closed while it runs.
        result = FakeEngine.run(self, prompt, settings, None)
        self.started.set()  # the forward has completed; the result is held
        self.release.wait(5)
        return result


@pytest.mark.asyncio
async def test_typing_q_s_and_slash_in_the_prompt_is_text_not_shortcuts():
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await until(pilot, lambda: app.state == "loaded")
        app.query_one("#tabs").active = "tab-run"
        app.query_one("#run-prompt", Input).focus()
        await pilot.press(*"quit s/q", "space", *"is")
        await pilot.pause(0.1)
        assert app.is_running
        assert app.query_one("#run-prompt", Input).value == "quit s/q is"
        assert "open" not in app.query_one("#search").classes  # / did not open search
        await pilot.press("enter")
        await until(pilot, lambda: app.query_one("#run-candidates", DataTable).row_count == 10)
        assert "MOCK" in text(app, "#run-status")
        assert "not recorded" in text(app, "#run-meta")
        assert app.state == "loaded"


@pytest.mark.asyncio
async def test_play_with_an_empty_prompt_asks_for_one_and_runs_nothing():
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await until(pilot, lambda: app.state == "loaded")
        await pilot.click("#menu-play")
        await pilot.pause(0.2)
        assert app.query_one("#tabs").active == "tab-run"
        assert app.focused is app.query_one("#run-prompt", Input)
        assert app.query_one("#run-candidates", DataTable).row_count == 0
        assert app.run_token == 0


@pytest.mark.asyncio
async def test_stop_says_the_forward_is_not_interrupted_and_discards_its_result():
    engine = Gated()
    app = Cockpit(engine, "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await until(pilot, lambda: app.state == "loaded")
        app.query_one("#run-prompt", Input).value = "The capital of France is"
        app.start_run()
        await until(pilot, lambda: app.state == "running")
        assert not app.query_one("#menu-stop", Button).disabled
        await pilot.click("#menu-stop")
        await pilot.pause(0.1)
        assert "can't be interrupted" in text(app, "#run-status")
        assert app.query_one("#menu-stop", Button).disabled  # requested once, not twice
        engine.release.set()
        await until(pilot, lambda: app.state == "loaded")
        assert text(app, "#run-status").startswith("Stopped")
        assert app.query_one("#run-candidates", DataTable).row_count == 0


@pytest.mark.asyncio
async def test_closing_the_model_mid_run_drops_its_result():
    engine = PastTheLastCheck()
    app = Cockpit(engine, "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await until(pilot, lambda: app.state == "loaded")
        app.query_one("#run-prompt", Input).value = "hello"
        app.start_run()
        await until(pilot, lambda: app.state == "running")
        await until(pilot, lambda: engine.started.is_set())
        app.close_model()
        engine.release.set()
        await pilot.pause(0.4)
        assert app.state == "off"
        assert app.query_one("#run-candidates", DataTable).row_count == 0


@pytest.mark.asyncio
async def test_a_new_model_mid_run_drops_the_old_runs_result():
    engine = PastTheLastCheck()
    app = Cockpit(engine, "fake", mock=True, engine_factory=FakeEngine)
    async with app.run_test(size=(160, 50)) as pilot:
        await until(pilot, lambda: app.state == "loaded")
        app.query_one("#run-prompt", Input).value = "hello"
        app.start_run()
        await until(pilot, lambda: app.state == "running")
        await until(pilot, lambda: engine.started.is_set())
        app.start_load("other", "other")
        await until(pilot, lambda: app.state == "loaded" and app.engine is not engine)
        engine.release.set()
        await pilot.pause(0.4)
        assert app.query_one("#run-candidates", DataTable).row_count == 0
        assert app.state == "loaded"


@pytest.mark.asyncio
async def test_the_run_tab_scrolls_in_an_80_by_24_terminal():
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(80, 24)) as pilot:
        await until(pilot, lambda: app.state == "loaded")
        app.query_one("#tabs").active = "tab-run"
        app.query_one("#run-prompt", Input).value = "The capital of France is"
        app.start_run()
        await until(pilot, lambda: app.query_one("#run-candidates", DataTable).row_count == 10)
        await pilot.pause(0.1)
        panel = app.query_one("#run-panel")
        width, height = app.size
        prompt = app.query_one("#run-prompt").region
        go = app.query_one("#run-go").region
        assert prompt.width > 0 and prompt.right <= width and go.width > 0 and go.right <= width
        assert panel.max_scroll_y > 0  # the content is taller than the screen, so it scrolls
        panel.scroll_end(animate=False)
        await pilot.pause(0.1)
        assert panel.scroll_y == panel.max_scroll_y



@pytest.mark.asyncio
async def test_stop_pressed_after_the_forward_finished_still_discards_the_result():
    engine = PastTheLastCheck()
    app = Cockpit(engine, "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await until(pilot, lambda: app.state == "loaded")
        app.query_one("#run-prompt", Input).value = "hello"
        app.start_run()
        await until(pilot, lambda: engine.started.is_set())  # the forward is done, result not yet shown
        await pilot.click("#menu-stop")
        engine.release.set()
        await until(pilot, lambda: app.state == "loaded")
        assert "result was discarded" in text(app, "#run-status")
        assert app.query_one("#run-candidates", DataTable).row_count == 0


@pytest.mark.asyncio
async def test_a_new_run_clears_the_previous_output_before_it_starts():
    engine = Gated()
    app = Cockpit(engine, "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await until(pilot, lambda: app.state == "loaded")
        engine.release.set()
        app.query_one("#run-prompt", Input).value = "first"
        app.start_run()
        await until(pilot, lambda: app.query_one("#run-candidates", DataTable).row_count == 10)
        engine.release.clear()
        app.query_one("#run-prompt", Input).value = "second"
        app.start_run()
        await until(pilot, lambda: app.state == "running")
        assert app.query_one("#run-candidates", DataTable).row_count == 0
        assert text(app, "#run-tokens") == ""
        engine.release.set()
        await until(pilot, lambda: app.state == "loaded")



@pytest.mark.asyncio
async def test_a_result_names_the_prompt_it_came_from_even_after_the_box_is_edited():
    app = Cockpit(FakeEngine(), "fake", mock=True)
    async with app.run_test(size=(160, 50)) as pilot:
        await until(pilot, lambda: app.state == "loaded")
        app.query_one("#run-prompt", Input).value = 'The "capital" of France is'
        app.start_run()
        await until(pilot, lambda: app.query_one("#run-candidates", DataTable).row_count == 10)
        app.query_one("#run-prompt", Input).value = "something else entirely"
        await pilot.pause(0.1)
        status = text(app, "#run-status")
        assert 'prompt "The \\"capital\\" of France is"' in status
        assert "run mock" in status
        assert "something else" not in status
