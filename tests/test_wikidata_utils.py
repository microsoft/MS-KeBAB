# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the wikidata_utils module."""

import json
from pathlib import Path

import pytest
from kebab.utils.dataset.wikidata import wikidata_utils
from pytest_mock import MockerFixture


@pytest.fixture
def wikidata_dump_sample_prop_records_file_path() -> Path:
    """Return the path to a JSON file with a single Wikidata record that is a property definition."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "sample" / "wikidata_dump_prop_records.json"


@pytest.fixture
def type_hierarchy_path() -> Path:
    """Return the path to a JSON file with the type hierarchy of Wikidata."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "wikidata_type_hierarchy.jsonl"


@pytest.fixture
def properties_path() -> Path:
    """Return the path to a JSON file with the properties of Wikidata."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "wikidata_properties.json"


@pytest.fixture
def temp_output_file(tmp_path: Path) -> Path:
    """Return the path to a temporary output file."""
    return tmp_path / "temp_output.json"


def test_extract_properties_from_dump(wikidata_dump_sample_prop_records_file_path: Path) -> None:
    """Test the extraction of property labels from the Wikidata JSON dump."""
    entities = list(
        wikidata_utils.extract_wikidata_entities_from_dump(
            wikidata_json_dump_path=wikidata_dump_sample_prop_records_file_path,
            include_descriptions=True,
            include_wikipedia_title=True,
        )
    )

    assert len(entities) == 1

    p = entities[0]
    assert p.entity_id == "P19"
    assert p.name == "place of birth"
    assert p.wikipedia_title == ""
    assert p.description.startswith("most specific known")


def test_query_entity_via_api_and_simplify(
    wikidata_dump_sample_prop_records_file_path: Path, mocker: MockerFixture
) -> None:
    """Test the querying of a Wikidata entity via the API and simplification of the response."""
    with open(wikidata_dump_sample_prop_records_file_path, encoding="utf-8") as f:
        entities = json.loads(f.read())

    entities_dict = {entity["id"]: entity for entity in entities}
    query_func = wikidata_utils._query_entities_via_api  # noqa: SLF001
    mocker.patch(f"{query_func.__module__}.{query_func.__qualname__}", return_value=entities_dict)

    queried_entity = wikidata_utils.query_single_entity_via_api("P19")

    assert queried_entity == entities[0]


def test_load_type_hierarchy(type_hierarchy_path: Path) -> None:
    """Test the loading of the type hierarchy of Wikidata."""
    graph, type_id_to_node_map = wikidata_utils.load_type_hierarchy(type_hierarchy_path)
    assert len(graph) == 106019
    assert len(type_id_to_node_map) == 106999


def test_load_properties(properties_path: Path) -> None:
    """Test the loading of the properties of Wikidata."""
    properties = wikidata_utils.load_properties(properties_path)
    assert len(properties) == 12272
    prop = properties["P6"]
    assert prop["label"] == "head of government"
    assert len(prop["aliases"]) > 0
    assert len(prop["description"]) > 0
    assert len(prop["subproperty_of"]) > 0
