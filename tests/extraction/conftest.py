# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Fixtures for extraction tests."""

import pathlib

import pytest

from kebab.entity import PropertySchema


@pytest.fixture
def property_schema() -> PropertySchema:
    return PropertySchema.from_file(pathlib.Path(__file__).parent.parent / "data" / "extraction" / "property_schema.json")

