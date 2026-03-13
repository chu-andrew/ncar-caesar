"""Shared fixtures for golden tests."""

import os

import pytest

from nc.loader import PROJECT_ROOT

GOLDEN_DIR = os.path.join(PROJECT_ROOT, "output")


@pytest.fixture
def golden_dir():
    """Path to the golden reference output directory."""
    return GOLDEN_DIR


@pytest.fixture
def tmp_output(tmp_path):
    """Provide a temporary output directory for test runs."""
    return tmp_path
