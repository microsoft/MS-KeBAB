# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the REBEL fragment extractor on T-REx data."""

import json
from pathlib import Path

import pytest
from kebab.utils.dataset.rebel.rebel_fragment_extractor import RebelFragmentExtractor


@pytest.fixture
def t_rex_single_doc_file_path() -> Path:
    """Return the path to a single T-REx JSON document record."""
    return Path(__file__).parent / "data" / "datasets" / "trex" / "t_rex_single_doc.json"


def test_extract_entities(t_rex_single_doc_file_path: Path) -> None:
    """Test the extraction of entities and properties from a single T-REx JSON document record."""
    # load a single T-REx document
    with open(t_rex_single_doc_file_path, encoding="utf-8") as f:
        doc_record = json.load(f)[0]

    entities = RebelFragmentExtractor.extract_entity_fragments_from_document(doc_record)
    assert len(entities) == 3

    entities = {entity_id: entity for entity_id, entity in entities.items() if len(entity.properties) > 1}
    assert len(entities) == 1

    main_ent_key = "Q12345"
    assert main_ent_key in entities
    assert "FictionalLang" in entities[main_ent_key].properties["name"]

    source_ids = entities[main_ent_key].source_ids

    for entity in entities:
        assert entities[entity].source_ids == source_ids
        assert len(entities[entity].properties) > 0


def test_extract_sub_documents(t_rex_single_doc_file_path: Path) -> None:
    """Test the extraction of sub-documents from a single REBEL JSON document record."""
    # load a single T-REx document
    with open(t_rex_single_doc_file_path, encoding="utf-8") as f:
        doc_record = json.load(f)[0]

    sub_documents = RebelFragmentExtractor.extract_sub_documents_from_document(doc_record)
    sub_documents = list(sub_documents)
    assert len(sub_documents) == len(doc_record["sentences_boundaries"])

    all_entities = []
    all_triples = []

    for sub_doc in sub_documents:
        assert sub_doc["docid"] == doc_record["docid"]
        assert sub_doc["title"] == doc_record["title"]
        assert sub_doc["uri"] == doc_record["uri"]

        all_entities.extend(sub_doc["entities"])
        all_triples.extend(sub_doc["triples"])

    assert len(all_entities) == len(doc_record["entities"])
    # Note: Sub-document extraction may create duplicate triples across sentence boundaries
    # so we check that we have at least the original number of triples
    assert len(all_triples) >= len(doc_record["triples"])

    # Verify that all original entities are represented
    all_ent_json = {json.dumps(entity, sort_keys=True) for entity in all_entities}
    ent_json = {json.dumps(entity, sort_keys=True) for entity in doc_record["entities"]}
    # The entities should contain at least the original entities
    assert ent_json.issubset(all_ent_json)

    # Verify that all original triples are represented
    all_triples_json = {json.dumps(triple, sort_keys=True) for triple in all_triples}
    triples_json = {json.dumps(triple, sort_keys=True) for triple in doc_record["triples"]}
    # The triples should contain at least the original triples
    assert triples_json.issubset(all_triples_json)
