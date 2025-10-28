"""Fixture definitions for ml-logger tests."""

import pytest
from pathlib import Path

from ml_dash import ML_Logger


@pytest.fixture
def logger(tmp_path):
    """Fixture that creates a ML_Logger instance with a temporary directory."""
    return ML_Logger("../data")


@pytest.fixture
def fixtures_dir():
    """Fixture that returns the path to test fixtures directory."""
    return Path(__file__).parent
