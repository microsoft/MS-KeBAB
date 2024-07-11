# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""This module contains the Entity class."""

# ruff: noqa: ANN401
# ANN401: typing.Any
from __future__ import annotations

import datetime
import json
import pathlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self


class ValueType(Enum):
    """Names of value types."""

    TEXT = "Text"
    NUMERIC = "Numeric"
    DATE = "Date"
    BOOLEAN = "Boolean"
    CATEGORY = "Category"
    REFERENCE = "Reference"


@dataclass
class DataType:
    """Specification of a data type."""

    data_type_id: str
    """Unique identifier for the data type."""

    value_type: ValueType
    description: str
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

    property_id: str
    """Unique (within the experiment) identifier for the property."""

    data_type: DataType
    description: str
    display_name: str | None = None
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
    """A complete specification of all properties and data types available in a given experiment."""

    name: str  # optional
    data_types: dict[str, DataType]
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

    def to_file(self, path: pathlib.Path) -> None:
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
    def from_file(cls, path: pathlib.Path) -> Self:
        """Load a property schema from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class Entity:
    """Represents an entity with its property values and evidence that supports them."""

    entity_id: str

    property_values: dict[str, list[Any]] = field(default_factory=lambda: defaultdict(list))
    """Dictionary of property values. Keys are property IDs, values are the lists of actual values."""

    source_ids: list[str] = field(default_factory=list)
    evidence_map: dict[str, list[list[int]]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self) -> None:
        """Post-initialisation checks."""
        assert self.entity_id, "Entity ID must be provided."

        if self.property_values is None:
            self.property_values = defaultdict(list)

        if self.source_ids is None:
            self.source_ids = []

        if self.evidence_map is None:
            self.evidence_map = defaultdict(list)

    def get_evidence_for_property_value(self, property_id: str, value_index: int) -> list[str]:
        """Get the evidence for a given property value."""
        assert 0 <= value_index < len(self.property_values[property_id]), "Invalid value index."

        prop_evidence = self.evidence_map.get(property_id, [])
        evidence_indices = prop_evidence[value_index] if prop_evidence else []
        return [self.source_ids[source_id] for source_id in evidence_indices]

    def to_dict(self) -> dict[str, Any]:
        """Convert the entity to a dictionary."""
        return {
            "entity_id": self.entity_id,
            "property_values": self.property_values,
            "source_ids": self.source_ids,
            "evidence_map": self.evidence_map,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create an entity from a dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Convert the entity to a JSON string."""

        def json_serial(obj: Any) -> Any:
            """JSON serializer for objects not serializable by default json code."""
            if isinstance(obj, datetime.datetime | datetime.date):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        return json.dumps(self.to_dict(), default=json_serial)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an entity from a JSON string."""
        # Note: dates will be represented as strings
        return cls.from_dict(json.loads(json_str))


class EntityUtilities:
    """Utilities for working with entities."""

    @staticmethod
    def load_entities(path_to_entities_dir: pathlib.Path) -> Iterable[Entity]:
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
    def save_collection(entities: list[Entity], path: pathlib.Path) -> None:
        """Save a collection of entities as a JSON-lines file."""
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for entity in entities:
                f.write(entity.to_json() + "\n")

    @staticmethod
    def load_schema(path: pathlib.Path) -> PropertySchema:
        """Load a property schema from a file."""
        return PropertySchema.from_file(path)

    @staticmethod
    def save_schema(schema: PropertySchema, path: pathlib.Path) -> None:
        """Save a property schema to a directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        schema.to_file(path)
