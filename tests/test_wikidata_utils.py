# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the wikidata_utils module."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kebab.utils.dataset.wikidata import wikidata_utils


@pytest.fixture
def wikidata_records_file_path() -> Path:
    """Return the path to a JSON file with a single Wikidata record that is a property definition."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "wikidata_prop_records.json"


@pytest.fixture
def temp_output_file(tmp_path: Path) -> Path:
    """Return the path to a temporary output file."""
    return tmp_path / "temp_output.json"


def test_extract_entities_from_dump(wikidata_records_file_path: Path) -> None:
    """Test the extraction of property labels from the Wikidata JSON dump."""
    entities = list(
        wikidata_utils.extract_simple_entities_from_dump(wikidata_json_dump_path=wikidata_records_file_path)
    )

    assert len(entities) == 1

    p = entities[0]
    assert p["id"] == "P19"
    assert p["name"] == "place of birth"


def test_query_entity_via_api_and_simplify(wikidata_records_file_path: Path) -> None:
    """Test the querying of a Wikidata entity via the API and simplification of the response."""
    with open(wikidata_records_file_path, encoding="utf-8") as f:
        entities = json.loads(f.read())

    entities_dict = {entity["id"]: entity for entity in entities}

    query_func = wikidata_utils._query_entities_via_api  # noqa: SLF001
    with patch(f"{query_func.__module__}.{query_func.__qualname__}") as mock_query_entity_via_api:
        mock_query_entity_via_api.return_value = entities_dict
        queried_entity = wikidata_utils.query_single_entity_via_api("P19")

    assert queried_entity == entities[0]
