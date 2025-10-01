# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kebab.utils.dataset.re_docred.re_docred_dataset_builder import ReDocRedDatasetBuilder


properties_map: dict[str, dict[str, str]] = {
    "P17": {"label": "country"},
    "P27": {"label": "country of citizenship"},
    "P35": {"label": "head of state"},
    "P131": {"label": "located in the administrative territorial entity"},
    "P150": {"label": "contains administrative territorial entity"},
    "P159": {"label": "headquarters location"},
    "P241": {"label": "military branch"},
    "P276": {"label": "location"},
    "P355": {"label": "subsidiary"},
    "P361": {"label": "part of"},
    "P527": {"label": "has part(s)"},
    "P569": {"label": "date of birth"},
    "P570": {"label": "date of death"},
    "P576": {"label": "date of founding or creation"},
    "P585": {"label": "point in time"},
    "P607": {"label": "participated in conflict"},
    "P710": {"label": "participant"},
    "P749": {"label": "parent organization"},
    "P1001": {"label": "applies to jurisdiction"},
    "P1344": {"label": "participant in"},
}


@pytest.fixture
def re_docred_input_file_path() -> Path:
    """Return the path to a single Re-DocRED JSON document record."""
    return Path(__file__).parent / "data" / "datasets" / "re-docred" / "re_docred_single_example.json"


@pytest.fixture
def re_docred_input_with_unicode_characters_path() -> Path:
    """Return the path to a single Re-DocRED JSON document record with unicode characters."""
    return Path(__file__).parent / "data" / "datasets" / "re-docred" / "re_docred_single_example_with_unicode.json"


@pytest.fixture
def wikidata_properties_path() -> Path:
    """Return the path to the Wikidata properties file."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "wikidata_properties.json"


def test_build_dataset(re_docred_input_file_path: Path, wikidata_properties_path: Path) -> None:
    """Test the extraction of entities and properties from a single Re-DocRED JSON document record."""
    # load a single Re-DocRED document
    with open(re_docred_input_file_path, encoding="utf-8") as f:
        doc_record = json.load(f)[0]

    builder = ReDocRedDatasetBuilder(
        re_docred_dir=re_docred_input_file_path,
        wikidata_properties_path=wikidata_properties_path,
        output_dir=Path.cwd(),
    )

    example = builder.extract_example(doc_record, properties_map)
    assert len(example["entities"]) == 15
    assert len(example["document"].document_id) == 64
    assert example["document"].document_id == "f5734628a85470a424f1a3d960fe1df2d12159baf43db94f1ae629ad091596bd"
    assert example["document"].schema.schema_id == "plain_text"
    assert example["document"].data["title"] == "AirAsia Zest"
    assert len(example["document"].data["text"]) == 1092

    expected_entity_ids = ["0", "1", "2", "3", "4", "5", "6", "8", "9", "10", "11", "12", "14", "15", "16"]
    for i, entity in enumerate(example["entities"]):
        assert entity.entity_id == expected_entity_ids[i]
        assert "name" in entity.properties
        assert isinstance(entity.properties["name"], list)
        assert "type" in entity.properties
        assert isinstance(entity.properties["type"], list)

    entity = example["entities"][0]
    assert len(entity.properties["name"]) == 3
    assert set(entity.properties["name"]) == {"Zest Airways, Inc.", "Asian Spirit and Zest Air", "AirAsia Zest"}
    assert entity.properties["type"] == ["organization"]
    assert len(entity.properties) == 5
    assert entity.properties.keys() == {
        "name",
        "type",
        "country",
        "located in the administrative territorial entity",
        "headquarters location",
    }


def test_build_dataset_with_unicode_characters(
    re_docred_input_with_unicode_characters_path: Path,
    wikidata_properties_path: Path,
) -> None:
    """Test the extraction of entities and properties from a single Re-DocRED JSON document record with unicode characters."""
    # load a single Re-DocRED document
    with open(re_docred_input_with_unicode_characters_path, encoding="utf-8") as f:
        doc_record = json.load(f)[0]
    builder = ReDocRedDatasetBuilder(
        re_docred_dir=re_docred_input_with_unicode_characters_path,
        wikidata_properties_path=wikidata_properties_path,
        output_dir=Path.cwd(),
    )

    example = builder.extract_example(doc_record, properties_map)
    assert len(example["entities"]) == 18
    assert len(example["document"].document_id) == 64
    assert example["document"].document_id == "1bad3abc89f7c5b1bb839f75362955bcf46c255d342b1092bca84464b8b1edcf"
    assert example["document"].schema.schema_id == "plain_text"
    assert example["document"].data["title"] == "Jirō Shiizaki"
    assert len(example["document"].data["text"]) == 1373
    entity = example["entities"][0]
    assert len(entity.properties["name"]) == 3
    assert set(entity.properties["name"]) == {"Jirō Shiizaki", "Shiizaki", "椎崎二郎,Shiizaki Jirō"}
    assert entity.properties["type"] == ["person"]
    assert len(entity.properties) == 8
    assert entity.properties.keys() == {
        "name",
        "type",
        "participated in conflict",
        "country of citizenship",
        "date of birth",
        "date of death",
        "military branch",
        "participant in",
    }
