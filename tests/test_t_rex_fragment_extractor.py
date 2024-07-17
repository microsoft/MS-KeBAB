# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the T-REx fragment extractor."""

import json
from pathlib import Path

import pytest

from kebab.dataset.t_rex.t_rex_fragment_extractor import TRexFragmentExtractor


@pytest.fixture
def t_rex_input_file_path() -> Path:
    """Return the path to a single T-REx JSON document record."""
    return Path(__file__).parent / "data" / "datasets" / "trex" / "t_rex_single_doc.json"


def test_extract_entities(t_rex_input_file_path: Path) -> None:
    """Test the extraction of entities and properties from a single T-REx JSON document record."""
    # load a single T-REx document
    with open(t_rex_input_file_path, encoding="utf-8") as f:
        doc_record = json.load(f)[0]

    entities = TRexFragmentExtractor.extract_entity_fragments_from_document(doc_record)
    assert len(entities) == 27

    entities = {entity_id: entity for entity_id, entity in entities.items() if len(entity.properties) > 1}
    assert len(entities) == 10

    main_ent_key = "Q33199"
    assert main_ent_key in entities
    assert "Austroasiatic languages" in entities[main_ent_key].names

    source_ids = entities[main_ent_key].source_ids

    for entity in entities:
        assert entities[entity].source_ids == source_ids
        assert len(entities[entity].properties) > 0
