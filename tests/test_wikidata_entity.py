from kebab.utils.dataset.wikidata.wikidata_utils import WikidataEntity


def test_wikidata_entity_formats() -> None:
    """Test the Wikidata entity serialisation and deserialisation."""
    # load old format entity
    old_entity_json = '{"id": "Q31", "name": "Belgium", "aliases": ["BE", "be", "BEL", "Kingdom of Belgium"], "types": ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"], "wikipedia": "Belgium"}'
    entity = WikidataEntity.from_json(old_entity_json)
    assert entity.entity_id == "Q31"
    assert entity.name == "Belgium"
    assert entity.aliases == ["BE", "be", "BEL", "Kingdom of Belgium"]
    assert entity.types == ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"]
    assert entity.wikipedia == "Belgium"

    entity_dict = entity.to_dict(minimal_repr=True)
    assert entity_dict == {
        "entity_id": "Q31",
        "properties": {
            "name": ["Belgium", "BE", "be", "BEL", "Kingdom of Belgium"],
            "P31": ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"],
        },
        "metadata": {"wikipedia": "Belgium"},
    }

    old_entity_json = '{"id": "Q31", "name": "Belgium", "aliases": ["BE", "be", "BEL", "Kingdom of Belgium"], "types": ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"]}'
    entity = WikidataEntity.from_json(old_entity_json)
    assert entity.wikipedia == ""
    entity_dict = entity.to_dict(minimal_repr=True)
    assert entity_dict == {
        "entity_id": "Q31",
        "properties": {
            "name": ["Belgium", "BE", "be", "BEL", "Kingdom of Belgium"],
            "P31": ["Q6256", "Q20181813", "Q185441", "Q1250464", "Q3624078", "Q43702"],
        },
    }
