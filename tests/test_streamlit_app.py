"""
tests/test_streamlit_app.py - Streamlit AppTest automated tests for AnchorGuard AI Dashboard.
"""

import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str((Path(__file__).parent.parent / "app.py").resolve())


def test_app_clean_startup():
    """Verifies that app.py starts up cleanly without uncaught exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run(timeout=25)
    assert not at.exception, f"App startup encountered exception: {at.exception}"


def test_app_run_simulation_button():
    """Verifies clicking the Run Simulation button executes successfully."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run(timeout=25)
    # Find Run Simulation button
    run_buttons = [b for b in at.button if "Run Simulation" in b.label]
    assert len(run_buttons) > 0, "Could not find Run Simulation button"
    run_buttons[0].click()
    at.run(timeout=25)
    assert not at.exception, f"Run Simulation button caused exception: {at.exception}"


def test_app_preset_selection():
    """Verifies clicking scenario preset buttons updates state and runs cleanly."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run(timeout=25)

    # Click Blocked Signals preset button
    blocked_buttons = [b for b in at.button if "Blocked Signals" in b.label]
    assert len(blocked_buttons) > 0, "Could not find Blocked Signals preset button"
    blocked_buttons[0].click()
    at.run(timeout=25)
    assert not at.exception, f"Blocked Signals button caused exception: {at.exception}"

    # Click Normal Signals preset button
    normal_buttons = [b for b in at.button if "Normal Signals" in b.label]
    assert len(normal_buttons) > 0, "Could not find Normal Signals preset button"
    normal_buttons[0].click()
    at.run(timeout=25)
    assert not at.exception, f"Normal Signals button caused exception: {at.exception}"
