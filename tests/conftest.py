# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared fixtures for tests using the data directory."""

import pathlib

import pytest


@pytest.fixture
def emails_exp_1_path() -> pathlib.Path:
    """Get the path to the emails experiment 1 directory."""
    return pathlib.Path(__file__).parent / "data" / "emails_exp_1"
