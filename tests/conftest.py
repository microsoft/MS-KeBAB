# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared fixtures for tests using the data directory."""

from pathlib import Path

import pytest


@pytest.fixture
def emails_exp_1_dir_path() -> Path:
    """Get the path to the emails experiment 1 directory."""
    return Path(__file__).parent / "data" / "emails_exp_1"
