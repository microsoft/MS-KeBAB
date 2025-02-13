import pytest
from kebab.utils.dataset.wikidata.wikidata_utils import TypeProperties, WikidataEntity


@pytest.fixture
def sample_wikidata_entity_belgium() -> WikidataEntity:
    """A sample Wikidata entity 1 for testing."""
    entity = WikidataEntity(entity_id="Q31")
    entity.properties["name"] = ["Belgium", "BE", "be", "BEL", "Kingdom of Belgium"]
    entity.properties[TypeProperties.INSTANCE_OF.value] = [
        "Q6256",
        "Q20181813",
        "Q185441",
        "Q1250464",
        "Q3624078",
        "Q43702",
    ]
    entity.source_ids = ["source1"]
    entity.description = "Belgium is a country in Western Europe."
    entity.wikipedia_title = "Belgium"
    return entity


def test_wikidata_entity_fields(sample_wikidata_entity_belgium: WikidataEntity) -> None:
    """Test the fields of the WikidataEntity."""
    assert sample_wikidata_entity_belgium.entity_id == "Q31"
    assert sample_wikidata_entity_belgium.properties["name"] == ["Belgium", "BE", "be", "BEL", "Kingdom of Belgium"]
    assert sample_wikidata_entity_belgium.name == "Belgium"
    assert sample_wikidata_entity_belgium.aliases == ["BE", "be", "BEL", "Kingdom of Belgium"]
    assert sample_wikidata_entity_belgium.type == ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"]
    assert sample_wikidata_entity_belgium.description == "Belgium is a country in Western Europe."
    assert sample_wikidata_entity_belgium.wikipedia_title == "Belgium"

    # copy
    entity = WikidataEntity.from_dict(sample_wikidata_entity_belgium.to_dict())
    assert entity.entity_id == "Q31"
    assert entity.properties["name"] == ["Belgium", "BE", "be", "BEL", "Kingdom of Belgium"]
    assert entity.name == "Belgium"
    assert entity.aliases == ["BE", "be", "BEL", "Kingdom of Belgium"]
    assert entity.type == ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"]
    assert entity.description == "Belgium is a country in Western Europe."
    assert entity.wikipedia_title == "Belgium"

    # minimal repr
    entity.description = ""
    entity.wikipedia_title = ""
    entity_dict = entity.to_dict(minimal_repr=True)

    assert entity_dict == {
        "entity_id": "Q31",
        "properties": {
            "name": ["Belgium", "BE", "be", "BEL", "Kingdom of Belgium"],
            "P31": ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"],
        },
        "source_ids": ["source1"],
    }

    # modify fields
    entity.aliases = ["Belgien"]
    entity.type = []
    entity.wikipedia_title = "Belgien"

    assert entity.properties["name"] == ["Belgium", "Belgien"]
    assert entity.properties[TypeProperties.INSTANCE_OF.value] == []
    assert entity.name == "Belgium"
    assert entity.aliases == ["Belgien"]
    assert entity.type == []
    assert entity.wikipedia_title == "Belgien"


def test_wikidata_entity_formats() -> None:
    """Test the Wikidata entity serialisation and deserialisation."""
    # load old format entity
    old_entity_json = '{"id": "Q31", "name": "Belgium", "aliases": ["BE", "be", "BEL", "Kingdom of Belgium"], "types": ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"], "wikipedia": "Belgium"}'
    entity = WikidataEntity.from_json(old_entity_json)
    assert entity.entity_id == "Q31"
    assert entity.name == "Belgium"
    assert entity.aliases == ["BE", "be", "BEL", "Kingdom of Belgium"]
    assert entity.type == ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"]
    assert entity.wikipedia_title == "Belgium"

    entity_dict = entity.to_dict(minimal_repr=True)
    assert entity_dict == {
        "entity_id": "Q31",
        "properties": {
            "name": ["Belgium", "BE", "be", "BEL", "Kingdom of Belgium"],
            "P31": ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"],
        },
        "metadata": {"wikipedia_title": "Belgium"},
    }

    old_entity_json = '{"id": "Q31", "name": "Belgium", "aliases": ["BE", "be", "BEL", "Kingdom of Belgium"], "types": ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"]}'
    entity = WikidataEntity.from_json(old_entity_json)
    assert entity.wikipedia_title == ""
    entity_dict = entity.to_dict(minimal_repr=True)
    assert entity_dict == {
        "entity_id": "Q31",
        "properties": {
            "name": ["Belgium", "BE", "be", "BEL", "Kingdom of Belgium"],
            "P31": ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"],
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
