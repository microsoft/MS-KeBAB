# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""This module contains the Entity class."""

# ANN401: typing.Any
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Self


class ValueType(Enum):
    """
    Names of value types.

    All properties are assumed to be collections of simple values.
    ValueType represents the element type of this collection.
    The element type can never be a collection itself (no nested collections).
    If an element is partially known and has a distribution over it, ValueType is the domain type of that distribution.
    """

    TEXT = "Text"
    NUMERIC = "Numeric"
    DATE = "Date"
    BOOLEAN = "Boolean"
    CATEGORY = "Category"
    REFERENCE = "Reference"


@dataclass
class DataType:
    """Specification of a data type."""

    # Unique identifier for the data type.
    data_type_id: str

    # Type of the data.
    value_type: ValueType

    # Textual description of the data type.
    description: str

    # List of possible values for a category data type, if applicable.
    category_values: list[Any] | None = None

    def __post_init__(self):
        """Post-initialisation checks."""
        assert self.data_type_id, "Data type ID must be provided."
        self.category_values = self.category_values or []

    def to_dict(self, as_reference: bool = False) -> dict[str, Any]:
        """
        Convert the data type to a dictionary.

        Args:
            as_reference: If true, only include the data type ID.
        """
        if as_reference:
            return {"data_type_id": self.data_type_id}

        return {
            "data_type_id": self.data_type_id,
            "value_type": self.value_type.value,
            "description": self.description,
            "category_values": self.category_values,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], data_types: dict[str, Self] | None = None) -> Self:
        """
        Create a data type from a dictionary.

        Args:
            data: Dictionary containing data type information.
            data_types: Dictionary of data types to use for resolving references.
        """
        data_type_id = data["data_type_id"]
        if data_types and data_type_id in data_types:
            return data_types[data_type_id]

        assert data["value_type"], "Value type must be provided."

        return cls(
            data_type_id=data["data_type_id"],
            value_type=ValueType(data["value_type"]),
            description=data["description"],
            category_values=data.get("category_values"),
        )


@dataclass
class Property:
    """Specification of a single property."""

    # Unique (within the experiment) identifier for the property.
    property_id: str

    # Data type that the property values must conform to.
    data_type: DataType

    # Textual description of the property.
    description: str

    # Human-readable name of the property.
    display_name: str | None = None

    # Whether the property is interpreted as a collection-valued property.
    # Having `is_collection = True` means that missing values are heavily penalized during evaluation.
    is_collection: bool = False

    def __post_init__(self):
        """Post-initialisation checks."""
        assert self.property_id, "Property ID must be provided."
        self.display_name = self.display_name or self.property_id

    def to_dict(self) -> dict[str, Any]:
        """Convert the property to a dictionary."""
        return {
            "property_id": self.property_id,
            "data_type_id": self.data_type.data_type_id,
            "description": self.description,
            "display_name": self.display_name,
            "is_collection": self.is_collection,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], data_types: dict[str, DataType]) -> Self:
        """Create a property from a dictionary."""
        return cls(
            property_id=data["property_id"],
            data_type=data_types[data["data_type_id"]],
            description=data["description"],
            display_name=data.get("display_name"),
            is_collection=data.get("is_collection", False),
        )


@dataclass
class PropertySchema:
    """
    A complete specification of all properties and data types available in a given experiment.

    This is a container for data types and properties and a helper functionality to load and save them.
    """

    # An optional name of the schema.
    name: str

    # Dictionary of data types, indexed by their unique IDs.
    # Always satisfied: data_types[some_id]["data_type_id"] == some_id
    data_types: dict[str, DataType]

    # Dictionary of properties, indexed by their unique IDs.
    # Always satisfied: properties[some_id]["property_id"] == some_id
    properties: dict[str, Property]

    def __init__(
        self,
        name: str = "",
        data_types: list[DataType] | None = None,
        properties: list[Property] | None = None,
    ) -> None:
        """Initialise the property schema."""
        self.name: str = name

        data_types = data_types or []
        properties = properties or []

        self.data_types: dict[str, DataType] = {dt.data_type_id: dt for dt in data_types} if data_types else {}
        self.properties: dict[str, Property] = {p.property_id: p for p in properties} if properties else {}

        assert len(self.data_types) == len(data_types), "Data type IDs must be unique."
        assert len(self.properties) == len(properties), "Property IDs must be unique."

        for prop in self.properties.values():
            assert prop.data_type.data_type_id in self.data_types, "Data type not found."

    def to_dict(self) -> dict[str, Any]:
        """Convert the property schema to a dictionary."""
        return {
            "name": self.name,
            "data_types": [data_type.to_dict() for data_type in self.data_types.values()],
            "properties": [prop.to_dict() for prop in self.properties.values()],
        }

    def to_file(self, path: Path) -> None:
        """Write the property schema to a JSON file."""
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a property schema from a dictionary."""
        data_types = {}
        for data_type_data in data["data_types"]:
            data_type = DataType.from_dict(data_type_data, data_types)
            data_types[data_type.data_type_id] = data_type

        return cls(
            name=data["name"],
            data_types=list(data_types.values()),
            properties=[Property.from_dict(prop_data, data_types) for prop_data in data["properties"]],
        )

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load a property schema from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class Entity:
    """Represents an entity with its property values and evidence that supports them."""

    # Unique identifier for the entity.
    entity_id: str

    # Dictionary of property values. Keys are property IDs, values are the lists of actual values.
    # For each property, a list element must be one of 3 things:
    #     - An actual value of type given by the ValueType of the property (string, number, etc.).
    #     - A set of values (treated as one-of alternatives).
    #     - A dictionary mapping values to probabilities.
    properties: dict[str, list[Any]] = field(default_factory=lambda: defaultdict(list))

    # List of source IDs that contributed to the entity.
    source_ids: list[str] = field(default_factory=list)

    # Map from property ID to a list that maps each value index to a list of evidence indices that support it.
    evidence_map: dict[str, list[list[int]]] = field(default_factory=lambda: defaultdict(list))

    # Additional metadata associated with the entity.
    # This is used for dataset creation only and should be empty for any benchmark task.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialisation checks."""
        assert self.entity_id is not None, "Entity ID must be provided."

        if self.properties is None:
            self.properties = defaultdict(list)

        # ensure properties is a defaultdict (won't be if e.g. loaded from JSON)
        if not isinstance(self.properties, defaultdict):
            self.properties = defaultdict(list, self.properties)

        if self.source_ids is None:
            self.source_ids = []

        if self.evidence_map is None:
            self.evidence_map = defaultdict(list)

        if not isinstance(self.evidence_map, defaultdict):
            self.evidence_map = defaultdict(list, self.evidence_map)

        if self.metadata is None:
            self.metadata = {}

    def __getitem__(self, property_: Property) -> list[Any]:
        """Lookup property values."""
        return self.properties[property_.property_id]

    def __contains__(self, property_: Property) -> bool:
        """Check if entity contains property."""
        return property_.property_id in self.properties

    def get_evidence_for_property_value(self, property_id: str, value_index: int) -> list[str]:
        """Get the evidence for a given property value."""
        assert 0 <= value_index < len(self.properties[property_id]), "Invalid value index."

        prop_evidence = self.evidence_map.get(property_id, [])
        evidence_indices = prop_evidence[value_index] if prop_evidence else []
        return [self.source_ids[source_id] for source_id in evidence_indices]

    def deduplicate_property_values(self) -> None:
        """Deduplicate values of each property."""
        for prop_id, values in self.properties.items():
            self.properties[prop_id] = list(dict.fromkeys(values))

    def deduplicate_source_ids(self) -> None:
        """Deduplicate source IDs."""
        self.source_ids = list(set(self.source_ids))

    def merge_with(self, other: Self, deduplicate_values: bool = True) -> Self:
        """Merge the other entity into this entity."""
        for prop_id, values in other.properties.items():
            self.properties[prop_id].extend(values)

        # TODO(pmyshkov): Handle evidence correctly instead of the below
        if deduplicate_values:
            self.deduplicate_property_values()

        if self.source_ids is not None and other.source_ids is not None:
            self.source_ids += other.source_ids
            self.deduplicate_source_ids()

        return self

    def to_dict(self, minimal_repr: bool = False) -> dict[str, Any]:
        """Convert the entity to a dictionary."""
        entity_dict = asdict(self)

        # remove metadata if empty
        if not entity_dict["metadata"]:
            del entity_dict["metadata"]

        if minimal_repr:
            for key in ["properties", "source_ids", "evidence_map"]:
                if key in entity_dict and not entity_dict[key]:
                    del entity_dict[key]

        return entity_dict

    def without_metadata(self) -> Self:
        """Return a new entity without metadata."""
        entity = self.__class__.from_dict(self.to_dict())
        entity.metadata = {}
        return entity

    def without_entity_id(self) -> Self:
        """Return a new entity without entity ID."""
        entity = self.__class__.from_dict(self.to_dict())
        entity.entity_id = ""
        return entity

    def filter_values(self, filter_func: Callable[[str, Any], bool]) -> Self | None:
        """Return a new entity with filtered values.

        Args:
            filter_func: A function that takes a property ID and a value, and returns True if the value should be kept.

        Returns:
            A new entity with filtered values.
        """
        updated_properties = deepcopy(self.properties)
        updated_evidence_map = deepcopy(self.evidence_map)
        source_ids = deepcopy(self.source_ids)
        for property_id, property_values in self.properties.items():
            indices_to_remove = []
            for idx, value in enumerate(property_values):
                if not filter_func(property_id, value):
                    indices_to_remove.append(idx)
            for idx in sorted(indices_to_remove, reverse=True):
                del updated_properties[property_id][idx]
                if idx < len(updated_evidence_map[property_id]):
                    del updated_evidence_map[property_id][idx]
            if not updated_properties[property_id]:
                updated_properties.pop(property_id, None)
                updated_evidence_map.pop(property_id, None)
        evidence_indices = set()
        for value_to_evidence in updated_evidence_map.values():
            for evidence in value_to_evidence:
                evidence_indices.update(evidence)
        missing_indices = set(range(len(self.source_ids))) - evidence_indices
        if missing_indices:
            # need to delete sources from source_ids and reflect that in evidence_map
            new_evidence_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(evidence_indices))}
            for property_id, value_to_evidence in updated_evidence_map.items():
                for value_idx, evidence in enumerate(value_to_evidence):
                    updated_evidence_map[property_id][value_idx] = [
                        new_evidence_idx[evidence_idx] for evidence_idx in evidence
                    ]
            for idx in sorted(missing_indices, reverse=True):
                del source_ids[idx]
        return (
            self.__class__(
                entity_id=self.entity_id,
                source_ids=source_ids,
                evidence_map=updated_evidence_map,
                metadata=self.metadata,
                properties=updated_properties,
            )
            if len(updated_properties) > 0
            else None
        )

    def remove_sources(self, remove_sources_list: list[str]) -> Self | None:
        """Return a new entity after removing property values corresponding to specified sources.

        Args:
            remove_sources_list: A list of sources IDs to remove.

        Returns:
            A new entity with filtered values.
        """
        updated_properties = deepcopy(self.properties)
        updated_evidence_map = deepcopy(self.evidence_map)
        updated_source_ids = deepcopy(self.source_ids)
        remove_sources_set = set(remove_sources_list)
        source_indices_to_remove = [
            idx for idx, source_id in enumerate(updated_source_ids) if source_id in remove_sources_set
        ]
        updated_source_ids = [source_id for source_id in updated_source_ids if source_id not in remove_sources_set]
        updated_evidence_map = {
            property_id: [
                [
                    evidence - sum([1 for source_index in source_indices_to_remove if source_index < evidence])
                    for evidence in evidences
                    if evidence not in source_indices_to_remove
                ]
                for evidences in values_to_evidences
            ]
            for property_id, values_to_evidences in updated_evidence_map.items()
        }
        for property_id, property_values in self.properties.items():
            for idx in reversed(range(len(property_values))):
                if len(updated_evidence_map[property_id][idx]) == 0:
                    del updated_properties[property_id][idx]
                    del updated_evidence_map[property_id][idx]
            if len(updated_properties[property_id]) == 0:
                del updated_properties[property_id]
                del updated_evidence_map[property_id]
        return (
            self.__class__(
                entity_id=self.entity_id,
                source_ids=updated_source_ids,
                evidence_map=updated_evidence_map,
                metadata=self.metadata,
                properties=updated_properties,
            )
            if len(updated_properties) > 0
            else None
        )

    def property_values_str(self) -> str:
        """Get the string representation of the entity properties and values."""
        properties_str = "|".join(
            f"{prop_id}:{','.join(sorted(values))}" for prop_id, values in sorted(self.properties.items())
        )

        return f"{self.entity_id} | {properties_str}"

    @classmethod
    def merge(cls, entities: list[Self], deduplicate_values: bool = True) -> Self:
        """Merge multiple entities into a single entity."""
        merged = cls(entity_id=entities[0].entity_id)

        for entity in entities:
            merged = merged.merge_with(entity, deduplicate_values=deduplicate_values)

        return merged

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create an entity from a dictionary."""
        return cls(**data)

    def to_json(self, minimal_repr: bool = False) -> str:
        """Convert the entity to a JSON string."""
        return json.dumps(self.to_dict(minimal_repr=minimal_repr))

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an entity from a JSON string."""
        # Note: dates will be represented as strings
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        """String representation of the entity."""
        return self.property_values_str()


class EntityUtilities:
    """Utilities for working with entities."""

    @staticmethod
    def load_entities(path_to_entities_dir: Path) -> Iterable[Entity]:
        """Load a collection of entities from a directory."""
        assert path_to_entities_dir.exists(), f"Directory not found: {path_to_entities_dir}"

        # jsons
        for entity_file in path_to_entities_dir.glob("*.json"):
            with open(entity_file, encoding="utf-8") as f:
                yield Entity.from_json(f.read())

        # json-lines
        for entity_file in path_to_entities_dir.glob("*.jsonl"):
            with open(entity_file, encoding="utf-8") as f:
                for line in f:
                    yield Entity.from_json(line)

    @staticmethod
    def save_collection(entities: list[Entity], path: Path) -> None:
        """Save a collection of entities as a JSON-lines file."""
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for entity in entities:
                f.write(entity.to_json() + "\n")

    @staticmethod
    def load_schema(path: Path) -> PropertySchema:
        """Load a property schema from a file."""
        return PropertySchema.from_file(path)

    @staticmethod
    def save_schema(schema: PropertySchema, path: Path) -> None:
        """Save a property schema to a directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        schema.to_file(path)
