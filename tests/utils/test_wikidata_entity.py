# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from kebab.utils.dataset.wikidata.wikidata_utils import TypeProperties, WikidataEntity


@pytest.fixture
def sample_wikidata_entity_belgium() -> WikidataEntity:
    """A sample Wikidata entity for testing."""
    entity = WikidataEntity(entity_id="Q12345")
    entity.properties["name"] = ["Fictional Republic", "FR", "fr", "FRE", "Republic of Fiction"]
    entity.properties[TypeProperties.INSTANCE_OF.value] = [
        "Q7777",
        "Q8888",
        "Q9999",
        "Q1111",
        "Q2222",
        "Q3333",
    ]
    entity.source_ids = ["source1"]
    entity.description = "Fictional Republic is a made-up country for testing purposes."
    entity.wikipedia_title = "Fictional_Republic"
    return entity


def test_wikidata_entity_fields(sample_wikidata_entity_belgium: WikidataEntity) -> None:
    """Test the fields of the WikidataEntity."""
    assert sample_wikidata_entity_belgium.entity_id == "Q12345"
    assert sample_wikidata_entity_belgium.properties["name"] == [
        "Fictional Republic",
        "FR",
        "fr",
        "FRE",
        "Republic of Fiction",
    ]
    assert sample_wikidata_entity_belgium.name == "Fictional Republic"
    assert sample_wikidata_entity_belgium.aliases == ["FR", "fr", "FRE", "Republic of Fiction"]
    assert sample_wikidata_entity_belgium.type == ["Q7777", "Q8888", "Q9999", "Q1111", "Q2222", "Q3333"]
    assert sample_wikidata_entity_belgium.description == "Fictional Republic is a made-up country for testing purposes."
    assert sample_wikidata_entity_belgium.wikipedia_title == "Fictional_Republic"

    # copy
    entity = WikidataEntity.from_dict(sample_wikidata_entity_belgium.to_dict())
    assert entity.entity_id == "Q12345"
    assert entity.properties["name"] == ["Fictional Republic", "FR", "fr", "FRE", "Republic of Fiction"]
    assert entity.name == "Fictional Republic"
    assert entity.aliases == ["FR", "fr", "FRE", "Republic of Fiction"]
    assert entity.type == ["Q7777", "Q8888", "Q9999", "Q1111", "Q2222", "Q3333"]
    assert entity.description == "Fictional Republic is a made-up country for testing purposes."
    assert entity.wikipedia_title == "Fictional_Republic"

    # minimal repr
    entity.description = ""
    entity.wikipedia_title = ""
    entity_dict = entity.to_dict(minimal_repr=True)

    assert entity_dict == {
        "entity_id": "Q12345",
        "properties": {
            "name": ["Fictional Republic", "FR", "fr", "FRE", "Republic of Fiction"],
            "P31": ["Q7777", "Q8888", "Q9999", "Q1111", "Q2222", "Q3333"],
        },
        "source_ids": ["source1"],
    }

    # modify fields
    entity.aliases = ["Fiction Land"]
    entity.type = []
    entity.wikipedia_title = "Fiction_Land"

    assert entity.properties["name"] == ["Fictional Republic", "Fiction Land"]
    assert entity.properties[TypeProperties.INSTANCE_OF.value] == []
    assert entity.name == "Fictional Republic"
    assert entity.aliases == ["Fiction Land"]
    assert entity.type == []
    assert entity.wikipedia_title == "Fiction_Land"


def test_wikidata_entity_formats() -> None:
    """Test the Wikidata entity serialisation and deserialisation."""
    # load old format entity
    old_entity_json = '{"id": "Q12345", "name": "Fictional Republic", "aliases": ["FR", "fr", "FRE", "Republic of Fiction"], "types": ["Q7777", "Q8888", "Q9999", "Q1111", "Q2222", "Q3333"], "wikipedia": "Fictional_Republic"}'
    entity = WikidataEntity.from_json(old_entity_json)
    assert entity.entity_id == "Q12345"
    assert entity.name == "Fictional Republic"
    assert entity.aliases == ["FR", "fr", "FRE", "Republic of Fiction"]
    assert entity.type == ["Q7777", "Q8888", "Q9999", "Q1111", "Q2222", "Q3333"]
    assert entity.wikipedia_title == "Fictional_Republic"

    entity_dict = entity.to_dict(minimal_repr=True)
    assert entity_dict == {
        "entity_id": "Q12345",
        "properties": {
            "name": ["Fictional Republic", "FR", "fr", "FRE", "Republic of Fiction"],
            "P31": ["Q7777", "Q8888", "Q9999", "Q1111", "Q2222", "Q3333"],
        },
        "metadata": {"wikipedia_title": "Fictional_Republic"},
    }

    old_entity_json = '{"id": "Q12345", "name": "Fictional Republic", "aliases": ["FR", "fr", "FRE", "Republic of Fiction"], "types": ["Q7777", "Q8888", "Q9999", "Q1111", "Q2222", "Q3333"]}'
    entity = WikidataEntity.from_json(old_entity_json)
    assert entity.wikipedia_title == ""
    entity_dict = entity.to_dict(minimal_repr=True)
    assert entity_dict == {
        "entity_id": "Q12345",
        "properties": {
            "name": ["Fictional Republic", "FR", "fr", "FRE", "Republic of Fiction"],
            "P31": ["Q7777", "Q8888", "Q9999", "Q1111", "Q2222", "Q3333"],
        },
    }


def test_wikidata_entity_to_json_roundtrip(sample_wikidata_entity_belgium: WikidataEntity) -> None:
    """Test the Wikidata entity serialisation and deserialisation."""
    entity_json = sample_wikidata_entity_belgium.to_json()
    entity = WikidataEntity.from_json(entity_json)
    assert entity == sample_wikidata_entity_belgium
    assert entity.to_json() == entity_json
    assert entity.to_dict() == sample_wikidata_entity_belgium.to_dict()

    # minimal repr
    entity.description = ""
    entity.wikipedia_title = ""
    entity_dict = entity.to_dict(minimal_repr=True)
    roundtrip = WikidataEntity.from_dict(entity_dict)
    assert roundtrip == entity.without_metadata()
