# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from kebab.contracts.entity import Entity


@pytest.fixture
def sample_entity_1() -> Entity:
    """A sample entity 1 for testing."""
    entity = Entity(entity_id="entity_1")
    entity.properties["prop1"].extend(["value11", "value12"])
    entity.properties["prop2"].append("value31")
    entity.source_ids.extend(["source1", "source2"])
    entity.evidence_map["prop1"].extend([[0], [1]])
    entity.evidence_map["prop2"].append([0])
    entity.metadata["meta1"] = "data1"
    return entity


@pytest.fixture
def sample_entity_2() -> Entity:
    """A sample entity 2 for testing."""
    entity = Entity(entity_id="entity_2")
    entity.properties["prop1"].extend(["value11", "value13"])
    entity.properties["prop3"].append("value31")
    entity.source_ids.extend(["source2", "source4"])
    entity.evidence_map["prop1"].append([0])
    entity.evidence_map["prop2"].extend([[0], [1]])
    entity.metadata["meta2"] = "data2"
    return entity


def test_get_evidence_for_property_value(sample_entity_1: Entity) -> None:
    """
    Test the get_evidence_for_property_value method for specific property values and indices.
    """
    # Check evidence for the values of "prop1" and "prop2
    assert sample_entity_1.get_evidence_for_property_value("prop1", 0) == ["source1"]
    assert sample_entity_1.get_evidence_for_property_value("prop1", 1) == ["source2"]
    assert sample_entity_1.get_evidence_for_property_value("prop2", 0) == ["source1"]


def test_to_dict_and_from_dict(sample_entity_1: Entity) -> None:
    """
    Test the to_dict and from_dict methods.
    """
    data = sample_entity_1.to_dict(minimal_repr=False)
    round_trip = Entity.from_dict(data)
    assert round_trip.entity_id == sample_entity_1.entity_id


def test_without_metadata_minimal_repr(sample_entity_1: Entity) -> None:
    """
    Test the to_dict method with minimal_repr=True and without_metadata().
    """
    data = sample_entity_1.without_metadata().to_dict(minimal_repr=True)
    assert "metadata" not in data

    round_trip = Entity.from_dict(data)
    assert "metadata" not in round_trip.to_dict()

    # drop source_ids and evidence_map
    trimmed_entity = sample_entity_1.without_metadata()
    trimmed_entity.source_ids.clear()
    trimmed_entity.evidence_map.clear()
    data = trimmed_entity.to_dict(minimal_repr=True)
    assert "source_ids" not in data
    assert "evidence_map" not in data

    round_trip = Entity.from_dict(data)
    assert round_trip.source_ids == []
    assert round_trip.evidence_map == {}


def test_merge_method(sample_entity_1: Entity, sample_entity_2: Entity) -> None:
    """
    Test the merge method by merging two sample entities.
    """
    merged = sample_entity_1.merge_with(sample_entity_2)
    assert merged.entity_id == sample_entity_1.entity_id

    for prop in ["prop1", "prop2", "prop3"]:
        assert prop in merged.properties

    assert set(merged.properties["prop1"]) == {"value11", "value12", "value13"}
    assert merged.properties["prop2"] == ["value31"]
    assert merged.properties["prop3"] == ["value31"]
    assert set(merged.source_ids) == {"source1", "source2", "source4"}


def filter_and_check(entity: Entity, expected: Entity, filter_func) -> None:
    """
    Helper function to filter an entity and check the result.
    """
    filtered_entity = entity.filter_values(filter_func)
    assert filtered_entity.entity_id == expected.entity_id
    assert filtered_entity.properties == expected.properties
    assert filtered_entity.source_ids == expected.source_ids
    assert filtered_entity.evidence_map == expected.evidence_map


def test_filter(sample_entity_1: Entity) -> None:
    """
    Test the filter method by filtering out specific values.
    """
    # Filter out "value12" from "prop1"
    filter_and_check(
        sample_entity_1,
        expected=Entity.from_dict(
            {
                "entity_id": "entity_1",
                "properties": {"prop1": ["value11"], "prop2": ["value31"]},
                "source_ids": ["source1"],
                "evidence_map": {"prop1": [[0]], "prop2": [[0]]},
            }
        ),
        filter_func=lambda _, value: value != "value12",
    )
    # Filter out "prop2" by filtering the only value "value31"
    filter_and_check(
        sample_entity_1,
        expected=Entity.from_dict(
            {
                "entity_id": "entity_1",
                "properties": {"prop1": ["value11", "value12"]},
                "source_ids": ["source1", "source2"],
                "evidence_map": {"prop1": [[0], [1]]},
            }
        ),
        filter_func=lambda _, value: value != "value31",
    )

    # Filter out "source1" from source_ids by filtering out all values with that evidence
    filter_and_check(
        sample_entity_1,
        expected=Entity.from_dict(
            {
                "entity_id": "entity_1",
                "properties": {"prop1": ["value12"]},
                "source_ids": ["source2"],
                "evidence_map": {
                    "prop1": [[0]],
                },
            }
        ),
        filter_func=lambda _, value: value not in {"value11", "value31"},
    )

    # Filter out all values
    filter_and_check(
        sample_entity_1,
        expected=Entity.from_dict(
            {
                "entity_id": "entity_1",
                "properties": {},
                "source_ids": [],
                "evidence_map": {},
            }
        ),
        filter_func=lambda _, value: False,  # noqa: ARG005
    )

    # filter values from entity with empty evidence_map
    filter_and_check(
        Entity.from_dict(
            {
                "entity_id": "entity_1",
                "properties": {"prop1": ["value12"]},
                "source_ids": ["source2"],
                "evidence_map": {},
            }
        ),
        expected=Entity.from_dict(
            {
                "entity_id": "entity_1",
                "properties": {},
                "source_ids": [],
                "evidence_map": {},
            }
        ),
        filter_func=lambda _, value: False,  # noqa: ARG005
    )
