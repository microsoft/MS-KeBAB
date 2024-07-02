# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the T-REx fragment extractor."""

import json
from pathlib import Path

import pytest

from kebab.dataset.trex.t_rex_fragment_extractor import TRexFragmentExtractor


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
    assert len(entities) == 10

    main_ent_key = "Q33199"
    assert main_ent_key in entities
    assert "Austroasiatic languages" in entities[main_ent_key].names

    source_id = entities[main_ent_key].source_id

    for entity in entities:
        assert entities[entity].source_id == source_id
        assert len(entities[entity].properties) > 0
