# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the REBEL fragment extractor."""

import json
from pathlib import Path

import pytest
from kebab.utils.dataset.rebel.rebel_fragment_extractor import RebelFragmentExtractor


@pytest.fixture
def rebel_single_record_file_path() -> Path:
    """Return the path to a single REBEL JSON document record."""
    return Path(__file__).parent / "data" / "datasets" / "rebel" / "rebel_single_doc.json"


@pytest.fixture
def rebel_sample_records_dir_path() -> Path:
    """Return the path to a directory with a sample of REBEL records."""
    return Path(__file__).parent / "data" / "datasets" / "rebel" / "sample"


def test_extract_entities(rebel_single_record_file_path: Path) -> None:
    """Test the extraction of entities and properties from a single REBEL JSON document record."""
    # load a single REBEL document
    with open(rebel_single_record_file_path, encoding="utf-8") as f:
        doc_record = json.load(f)

    entities = RebelFragmentExtractor.extract_entity_fragments_from_document(doc_record)
    assert len(entities) == 2

    entities = {entity_id: entity for entity_id, entity in entities.items() if len(entity.properties) > 1}
    assert len(entities) == 1

    main_ent_key = "Q1001000"
    assert main_ent_key in entities
    entity = entities[main_ent_key]

    assert entity.properties["name"] == ["Acme Widget Pro"]
    assert entity.metadata["doc_id"] == "12345678"

    for entity in entities.values():
        assert entity.source_ids == []
        assert len(entity.properties) > 0


def test_run(rebel_sample_records_dir_path: Path, tmp_path: Path) -> None:
    """Test the extraction of entities and properties from a single REBEL JSON document record."""
    extractor = RebelFragmentExtractor(rebel_dir=rebel_sample_records_dir_path, output_dir=tmp_path)
    extractor.run()

    output_file = tmp_path / RebelFragmentExtractor.ENTITY_FRAGMENTS_FILENAME
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert len(lines) == 6
