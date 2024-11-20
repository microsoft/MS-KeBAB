# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""This module contains the EntityFragment class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kebab.entity import Entity


@dataclass
class EntityFragment(Entity):
    """
    Represents a lightweight entity fragment originating from T-REx and Wikidata.

    In T-REx, what we extract are fragments of Wikidata entities. They share the same Wikidata ID,
    and we later merge them via this ID to obtain a complete ground truth entity.
    """

    # These are not the types found in the fragment, but the types of the full entity - can be used to filter the
    # dataset.
    entity_types: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        """
        Return the names of the entity.
        In T-REx, names (surface forms) are handled separately from the other properties.
        """
        return self.properties["names"]

    @names.setter
    def names(self, value: list[str]) -> None:
        """Set the names of the entity."""
        self.properties["names"] = value

    def make_property_values_unique(self) -> None:
        """Make the values of each property unique."""
        for prop_id, values in self.properties.items():
            self.properties[prop_id] = list(set(values))

    def merge_with(self, other: EntityFragment) -> EntityFragment:
        """Merge the other entity fragment into the current fragment."""
        self.names.extend(other.names)
        for prop_id, values in other.properties.items():
            self.properties[prop_id].extend(values)

        self.make_property_values_unique()
        self.source_ids = list(set(self.source_ids + other.source_ids))

        return self

    def get_hashable_value_repr(self) -> str:
        """Return a hashable representation of the entity fragment."""
        properties_str = "|".join(
            f"{prop_id}:{','.join(sorted(values))}" for prop_id, values in sorted(self.properties.items())
        )

        return f"{self.entity_id} | {properties_str}"

    @classmethod
    def merge(cls, entities: list[EntityFragment]) -> EntityFragment:
        """Merge multiple entity fragments into a single entity fragment."""
        merged = EntityFragment(entity_id=entities[0].entity_id)

        for entity in entities:
            merged = merged.merge_with(entity)

        return merged

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the entity fragment."""
        base_dict = super().to_dict()
        base_dict["entity_types"] = self.entity_types

        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityFragment:
        """Create an entity fragment from a dictionary."""
        return cls(**data)
