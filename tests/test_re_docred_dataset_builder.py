from __future__ import annotations

import json
from pathlib import Path

import pytest

from kebab.utils.dataset.re_docred.re_docred_dataset_builder import ReDocRedDatasetBuilder


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
    assert len(example["entities"]) == 13
    assert len(example["document"].document_id) == 64
    assert example["document"].document_id == "a057557c54e7d6eed98de8ba50deecd7de5b224adafa6dbf86e67654d0a2a7fc"
    assert example["document"].schema.schema_id == "plain_text"
    assert example["document"].data["title"] == "AirAsia Zest"
    assert len(example["document"].data["text"]) == 1093

    expected_entity_ids = ["0", "1", "2", "3", "4", "5", "6", "9", "10", "11", "12", "14", "15"]
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
