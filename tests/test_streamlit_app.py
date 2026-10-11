import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str((Path(__file__).parent.parent / "app.py").resolve())


def test_app_clean_startup():
    """Verifies that app.py starts up cleanly without uncaught exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App startup encountered exception: {at.exception}"


def test_navigation_sections():
    """Verifies that navigation to each of the 6 workbench sections runs cleanly."""
    sections = [
        "Overview",
        "Environment Studio",
        "Measurement Lab",
        "Algorithm Lab",
        "Evaluation Lab",
        "Data & Reports",
    ]

    for sec in sections:
        at = AppTest.from_file(APP_PATH, default_timeout=25)
        at.run(timeout=25)
        # Find sidebar radio and select section
        radio = at.sidebar.radio[0]
        radio.set_value(sec)
        at.run(timeout=25)
        assert not at.exception, f"Section '{sec}' failed with exception: {at.exception}"


def test_app_run_simulation_button():
    """Verifies clicking the Run Simulation button executes successfully."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run(timeout=25)
    # Button is in sidebar
    run_btn = at.sidebar.button[0]
    run_btn.click()
    at.run(timeout=25)
    assert not at.exception, f"Run Simulation button caused exception: {at.exception}"
