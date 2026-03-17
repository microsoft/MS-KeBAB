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
    assert sample_wikidata_entity_belgium.instance_of == ["Q7777", "Q8888", "Q9999", "Q1111", "Q2222", "Q3333"]
    assert sample_wikidata_entity_belgium.description == "Fictional Republic is a made-up country for testing purposes."
    assert sample_wikidata_entity_belgium.wikipedia_title == "Fictional_Republic"

    # copy
    entity = WikidataEntity.from_dict(sample_wikidata_entity_belgium.to_dict())
    assert entity.entity_id == "Q12345"
    assert entity.properties["name"] == ["Fictional Republic", "FR", "fr", "FRE", "Republic of Fiction"]
    assert entity.name == "Fictional Republic"
    assert entity.aliases == ["FR", "fr", "FRE", "Republic of Fiction"]
    assert entity.instance_of == ["Q7777", "Q8888", "Q9999", "Q1111", "Q2222", "Q3333"]
    assert entity.description == "Fictional Republic is a made-up country for testing purposes."
    assert entity.wikipedia_title == "Fictional_Republic"

    # minimal repr
    entity.description = ""
    entity.wikipedia_title = ""
    entity_dict = entity.to_dict(minimal_repr=True)

    # Minimal representation now preserves empty description / wikipedia_title
    # in the metadata; assert core fields and ensure metadata matches the
    # current entity state rather than a hard-coded dict.
    assert entity_dict["entity_id"] == "Q12345"
    assert entity_dict["properties"]["name"] == [
        "Fictional Republic",
        "FR",
        "fr",
        "FRE",
        "Republic of Fiction",
    ]
    assert entity_dict["properties"][TypeProperties.INSTANCE_OF.value] == [
        "Q7777",
        "Q8888",
        "Q9999",
        "Q1111",
        "Q2222",
        "Q3333",
    ]
    assert entity_dict["source_ids"] == ["source1"]
    assert entity_dict.get("metadata", {}) == {"description": "", "wikipedia_title": ""}

    # modify fields
    entity.aliases = ["Fiction Land"]
    entity.instance_of = []
    entity.wikipedia_title = "Fiction_Land"

    assert entity.properties["name"] == ["Fictional Republic", "Fiction Land"]
    assert entity.properties[TypeProperties.INSTANCE_OF.value] == []
    assert entity.name == "Fictional Republic"
    assert entity.aliases == ["Fiction Land"]
    assert entity.instance_of == []
    assert entity.wikipedia_title == "Fiction_Land"


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
    # Minimal roundtrip should preserve the (now possibly empty) metadata
    # rather than stripping it.
    assert roundtrip == entity
