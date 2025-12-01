# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

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


def filter_and_check(entity: Entity, expected: Entity | None, filter_func) -> None:
    """
    Helper function to filter an entity and check the result.
    """
    filtered_entity = entity.filter_values(filter_func)
    if expected is None:
        assert filtered_entity is None
    else:
        assert filtered_entity is not None
        assert filtered_entity.entity_id == expected.entity_id
        assert filtered_entity is not None
        assert filtered_entity.properties == expected.properties
        assert filtered_entity is not None
        assert filtered_entity.source_ids == expected.source_ids
        assert filtered_entity is not None
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
        expected=None,
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
        expected=None,
        filter_func=lambda _, value: False,  # noqa: ARG005
    )


def remove_sources_and_check(entity: Entity, expected: Entity | None, remove_sources_list: list[str]) -> None:
    """
    Helper function to remove sources from an entity and check the result.
    """
    updated_entity = entity.remove_sources(remove_sources_list=remove_sources_list)
    if expected is None:
        assert updated_entity is None
    else:
        assert updated_entity is not None
        assert updated_entity.entity_id == expected.entity_id
        assert updated_entity is not None
        assert updated_entity.properties == expected.properties
        assert updated_entity is not None
        assert updated_entity.source_ids == expected.source_ids
        assert updated_entity is not None
        assert updated_entity.evidence_map == expected.evidence_map


def test_remove_sources(sample_entity_1: Entity) -> None:
    """
    Test the remove sources method.
    """
    # Filter out "source1" from source_ids by filtering out all values with that evidence
    remove_sources_and_check(
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
        remove_sources_list=["source1"],
    )

    # Filter out all sources
    remove_sources_and_check(
        sample_entity_1,
        expected=None,
        remove_sources_list=["source1", "source2"],
    )


def test_shuffled_properties_structure_and_contents(sample_entity_1: Entity) -> None:
    """Shuffled entity preserves values/evidence but changes ordering.

    The method is intended for randomization, so we only assert structural
    invariants and set equality, not specific order.
    """

    shuffled = sample_entity_1.with_shuffled_properties()

    # Same entity id and metadata
    assert shuffled.entity_id == sample_entity_1.entity_id
    assert shuffled.metadata == sample_entity_1.metadata

    # Same set of property ids
    assert set(shuffled.properties.keys()) == set(sample_entity_1.properties.keys())

    # For each property id, values form the same multiset
    for prop_id, values in sample_entity_1.properties.items():
        assert sorted(shuffled.properties[prop_id]) == sorted(values)

    # Evidence map keys are the same
    assert set(shuffled.evidence_map.keys()) == set(sample_entity_1.evidence_map.keys())

    # For each property, evidence lists line up with value list lengths
    for prop_id, values in shuffled.properties.items():
        evidences = shuffled.evidence_map.get(prop_id, [])
        # evidence list may be shorter if some values had no evidence
        assert len(evidences) <= len(values)

        # All evidence indices must be valid source indices
        for evidence_list in evidences:
            for idx in evidence_list:
                assert 0 <= idx < len(shuffled.source_ids)

        # As set, evidences must match original evidences
        original_evidences = sample_entity_1.evidence_map.get(prop_id, [])
        assert {tuple(sorted(e)) for e in evidences} == {tuple(sorted(e)) for e in original_evidences}

    # Evidence map not broken
    assert len(shuffled.properties["prop1"]) == 2
    for idx, value in enumerate(shuffled.properties["prop1"]):
        evid = shuffled.evidence_map["prop1"][idx]
        if value == "value11":
            assert evid == [0]
        elif value == "value12":
            assert evid == [1]

    assert shuffled.properties["prop2"] == ["value31"]
    assert shuffled.evidence_map["prop2"] == [[0]]


def test_shuffled_properties_does_not_mutate_original(sample_entity_1: Entity) -> None:
    """Calling shuffled_properties() must not mutate the original entity."""

    original_dict = sample_entity_1.to_dict()
    _ = sample_entity_1.with_shuffled_properties()
    assert sample_entity_1.to_dict() == original_dict


def test_sorted_properties_orders_values_and_properties() -> None:
    """with_sorted_properties sorts values and property IDs, with name first."""

    entity = Entity(
        entity_id="e1",
        properties={
            "b": ["beta", "alpha"],
            "name": ["Bob", "Alice"],
            "a": ["z", "y"],
        },
        source_ids=["s1", "s2"],
        evidence_map={
            "b": [[0], [1]],
            "name": [[0], [1]],
            "a": [[0], [1]],
        },
    )

    sorted_entity = entity.with_sorted_properties()

    # name must be first, remaining properties lexicographically sorted
    assert list(sorted_entity.properties.keys()) == ["name", "a", "b"]

    # Values within each property are sorted and evidence follows
    assert sorted_entity.properties["name"] == ["Alice", "Bob"]
    assert sorted_entity.evidence_map["name"] == [[1], [0]]

    assert sorted_entity.properties["a"] == ["y", "z"]
    assert sorted_entity.evidence_map["a"] == [[1], [0]]

    assert sorted_entity.properties["b"] == ["alpha", "beta"]
    assert sorted_entity.evidence_map["b"] == [[1], [0]]


def test_sorted_properties_without_name_property() -> None:
    """with_sorted_properties falls back to pure lexicographic order without name."""

    entity = Entity(
        entity_id="e2",
        properties={
            "c": ["2"],
            "a": ["1"],
            "b": ["3"],
        },
    )

    sorted_entity = entity.with_sorted_properties()

    assert list(sorted_entity.properties.keys()) == ["a", "b", "c"]
