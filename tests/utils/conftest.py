# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared fixtures for tests using the data directory."""

from pathlib import Path

import pytest
from kebab.utils.dataset.wikidata.wikidata_utils import WikidataEntity


@pytest.fixture
def wikidata_entity() -> WikidataEntity:
    """Get a fictitious Wikidata entity for testing."""
    entity_dict = {
        "entity_id": "Q12345",
        "properties": {
            "name": ["Fictional Republic", "FR", "fr", "FRE", "Republic of Fiction"],
            "P31": ["Q7777", "Q8888", "Q9999", "Q1111", "Q2222", "Q3333"],
        },
        "metadata": {"wikipedia": "Fictional_Republic"},
    }

    return WikidataEntity.from_dict(entity_dict)


@pytest.fixture
def wikidata_full_type_hierarchy_file_path() -> Path:
    """Return the path to a JSON file with the type hierarchy of Wikidata."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "wikidata_type_hierarchy.jsonl"


@pytest.fixture
def wikidata_full_properties_file_path() -> Path:
    """Return the path to a JSON file with the properties of Wikidata."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "wikidata_properties.json"


@pytest.fixture
def wikidata_dump_sample_10_records_file_path() -> Path:
    """Return the path to the input file with a small subset of the Wikidata dump."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "sample" / "wikidata_dump_10_records.json"


@pytest.fixture
def wikidata_sample_simple_entities_file_path() -> Path:
    """Return the path to the input file with a subset of Wikidata simple entities."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "sample" / "wikidata_simple_entities.jsonl"
