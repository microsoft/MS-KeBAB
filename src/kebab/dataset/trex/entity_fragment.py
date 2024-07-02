# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""This module contains the EntityFragment class."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field


class EntityFragment(BaseModel):
    """
    Represents a lightweight entity fragment originating from T-REx and Wikidata.

    In T-REx, what we extract are fragments of Wikidata entities. They share the same Wikidata ID,
    and we later merge them via this ID to obtain a complete ground truth entity.
    """

    entity_id: str
    names: set[str] = Field(default_factory=set)
    properties: dict[str, set] = Field(default_factory=lambda: defaultdict(set))
    source_id: str = ""

    def merge_with(self, other: EntityFragment) -> EntityFragment:
        """Merge the other entity fragment into the current fragment."""
        if self.entity_id != other.entity_id:
            raise ValueError("Cannot merge entities with different IDs")

        self.names.update(other.names)
        for prop_id, values in other.properties.items():
            self.properties[prop_id].update(values)

        self.source_id = "MERGED"

        return self

    @classmethod
    def merge(cls, entities: list[EntityFragment]) -> EntityFragment:
        """Merge multiple entity fragments into a single entity fragment."""
        merged = EntityFragment(entity_id=entities[0].entity_id)

        for entity in entities:
            merged = merged.merge_with(entity)

        return merged
