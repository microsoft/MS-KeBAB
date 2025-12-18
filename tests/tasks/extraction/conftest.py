# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Fixtures for extraction tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from kebab.contracts.entity import PropertySchema


@pytest.fixture
def property_schema() -> PropertySchema:
    """Load extraction PropertySchema for testing."""
    return PropertySchema.from_file(Path(__file__).parents[2] / "data" / "extraction" / "property_schema_metrics.json")

@pytest.fixture
def metrics_config() -> dict[str, Any]:
    """Load extraction metrics config for testing."""
    from kebab.utils.io_helpers import load_dict_from_json

    return load_dict_from_json(
        Path(__file__).parents[2] / "data" / "extraction" / "metrics_config.json"
    )