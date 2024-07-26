from __future__ import annotations

import json
from pathlib import Path

import pytest

from kebab.dataset.re_docred.re_docred_dataset_builder import ReDocRedDatasetBuilder


properties_map: dict[str, dict[str, str]] = {
    "P17": {"label": "country"},
    "P131": {"label": "located in the administrative territorial entity"},
    "P150": {"label": "contains administrative territorial entity"},
    "P159": {"label": "headquarters location"},
    "P355": {"label": "subsidiary"},
    "P576": {"label": "date of founding or creation"},
    "P749": {"label": "parent organization"},
}


@pytest.fixture
def re_docred_input_file_path() -> Path:
    """Return the path to a single T-REx JSON document record."""
    return Path(__file__).parent / "data" / "datasets" / "re-docred" / "re_docred_single_example.json"


def test_build_dataset(re_docred_input_file_path: Path) -> None:
    """Test the extraction of entities and properties from a single Re-DocRED JSON document record."""
    # load a single Re-DocRED document
    with open(re_docred_input_file_path, encoding="utf-8") as f:
        doc_record = json.load(f)[0]

    example = ReDocRedDatasetBuilder.extract_example(doc_record, properties_map)
    assert len(example["entities"]) == 17
    assert len(example["text"]) == 1108
    for entity in example["entities"]:
        assert "name" in entity
        assert isinstance(entity["name"], str)
        assert "type" in entity
        assert isinstance(entity["type"], str)
        if "alternative_names" in entity:
            assert isinstance(entity["alternative_names"], list)
            for name in entity["alternative_names"]:
                assert isinstance(name, str)

    entity = example["entities"][0]
    assert entity["name"] == "Zest Airways, Inc."
    assert entity["type"] == "Organization"
    assert len(entity["alternative_names"]) == 2
    assert len(entity) == 6
    assert entity.keys() == {
        "name",
        "type",
        "alternative_names",
        "country",
        "located_in_the_administrative_territorial_entity",
        "headquarters_location",
    }
