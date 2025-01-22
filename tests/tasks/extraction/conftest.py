# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Fixtures for extraction tests."""

import pathlib

import pytest

from kebab.contracts.entity import PropertySchema


@pytest.fixture
def property_schema() -> PropertySchema:
    """Load extraction PropertySchema for testing."""
    return PropertySchema.from_file(
        pathlib.Path(__file__).parent.parent.parent / "data" / "extraction" / "property_schema.json"
    )
