"""
Given a collection of entities, substitute property values, optionally replace property IDs with names,
and optionally attach types to the entities. The values are resolved from the Wikidata dump of simple entities,
followed by queries to the Wikidata API for missing information.

Inputs:
- Entities to resolve (required)
- Wikidata simple entities (optional): used to resolve referenced entity values and entity types
- Wikidata properties map (optional): used to map property IDs to human-readable names
- Wikidata type hierarchy (optional): used to resolve type names and to attach types to metadata

Behavior when optional inputs are missing:
- No simple entities: referenced values are not de-referenced; keep original IDs; types are not attached
- No properties map: keep property IDs; do not resolve property names
- No type hierarchy: do not resolve type names; do not attach types to metadata
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from kebab.contracts.entity import Entity
from kebab.utils.dataset.wikidata import wikidata_utils
from kebab.utils.io_helpers import resolve_path


class WikidataResolver:
    """A class for resolving Wikidata entities and properties."""

    _logger: logging.Logger
    entities_path: Path
    wikidata_simple_entities_path: Path | None
    wikidata_properties_path: Path | None
    wikidata_type_hierarchy_path: Path | None
    resolve_property_names: bool
    resolve_property_values: bool
    attach_types: bool
    resolve_attached_types: bool
    query_api: bool
    save_with_minimal_repr: bool
    output_dir: Path
    entities_output_path: Path
    _has_simple_entities: bool
    _has_properties: bool
    _has_type_hierarchy: bool

    def __init__(
        self,
        *,
        entities_path: Path,
        wikidata_simple_entities_path: Path | None = None,
        wikidata_properties_path: Path | None = None,
        wikidata_type_hierarchy_path: Path | None = None,
        attach_types: bool = True,
        resolve_attached_types: bool = False,
        resolve_property_names: bool = True,
        resolve_property_values: bool = True,
        query_api: bool = True,
        save_with_minimal_repr: bool = True,
        output_dir: Path,
    ):
        """Initialize the Wikidata resolver."""
        self._logger = logging.getLogger(self.__class__.__name__)
        self.entities_path = resolve_path(entities_path)
        self.wikidata_simple_entities_path = (
            resolve_path(wikidata_simple_entities_path) if wikidata_simple_entities_path is not None else None
        )
        self.wikidata_properties_path = (
            resolve_path(wikidata_properties_path) if wikidata_properties_path is not None else None
        )
        self.wikidata_type_hierarchy_path = (
            resolve_path(wikidata_type_hierarchy_path) if wikidata_type_hierarchy_path is not None else None
        )
        self.resolve_property_names = resolve_property_names
        self.resolve_property_values = resolve_property_values
        self.attach_types = attach_types
        self.resolve_attached_types = resolve_attached_types
        self.query_api = query_api
        self.save_with_minimal_repr = save_with_minimal_repr
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entities_output_path = self.output_dir / self.entities_path.name
        self._has_simple_entities = False
        self._has_properties = False
        self._has_type_hierarchy = False

        if not self.entities_path.exists():
            raise FileNotFoundError(f"Entities file not found: {self.entities_path}")

        # Optional inputs may be omitted; related features will be disabled in run()

        if self.entities_path == self.entities_output_path:
            raise ValueError("Input and output entities paths are the same.")

        if self.entities_output_path.exists():
            self._logger.warning(f"Output entities file already exists: {self.entities_output_path}")
            self._logger.warning("It will be overwritten.")

    def run(self) -> None:
        """Run the dataset building pipeline."""
        # Determine availability of optional inputs and adjust behavior with warnings
        self._has_simple_entities = bool(
            self.wikidata_simple_entities_path and self.wikidata_simple_entities_path.exists()
        )
        self._has_properties = bool(self.wikidata_properties_path and self.wikidata_properties_path.exists())
        self._has_type_hierarchy = bool(
            self.wikidata_type_hierarchy_path and self.wikidata_type_hierarchy_path.exists()
        )

        if not self._has_simple_entities:
            if self.resolve_property_values:
                self._logger.warning(
                    "Simple entities not provided; referenced values will not be de-referenced (IDs kept)."
                )
                self._logger.warning("Disabling property value resolution.")
                self.resolve_property_values = False
            if self.attach_types or self.resolve_attached_types:
                self._logger.warning(
                    "Simple entities not provided; types cannot be determined. Disabling type attachment."
                )
                self.attach_types = False
                self.resolve_attached_types = False

        if not self._has_properties and self.resolve_property_names:
            self._logger.warning("Properties map not provided; property IDs will be kept (no name mapping).")
            self.resolve_property_names = False

        if not self._has_type_hierarchy:
            if self.resolve_attached_types:
                self._logger.warning(
                    "Type hierarchy not provided; cannot resolve type names for attached types. Disabling."
                )
                self.resolve_attached_types = False
            if self.attach_types:
                self._logger.warning("Type hierarchy not provided; not attaching types to entity metadata.")
                self.attach_types = False

        wikidata_entities: dict[str, wikidata_utils.WikidataEntity] = {}

        # load wikidata properties and type hierarchy (if available)
        wikidata_properties: dict[str, dict] = {}
        type_id_to_node: dict[str, dict] = {}

        if self._has_properties:
            self._logger.info("Loading Wikidata properties")
            wikidata_properties = wikidata_utils.load_properties(self.wikidata_properties_path)  # type: ignore[arg-type]

        if self._has_type_hierarchy:
            self._logger.info("Loading Wikidata type hierarchy")
            _, type_id_to_node = wikidata_utils.load_type_hierarchy(self.wikidata_type_hierarchy_path)  # type: ignore[arg-type]

        if self.resolve_property_values or self.attach_types:
            # collect all referenced IDs (for which we'll need to get Wikidata entities)
            self._logger.info("Collecting all referenced IDs")
            referenced_ids = self.collect_referenced_ids()

            # load and query wikidata entities (if available)
            if self._has_simple_entities:
                self._logger.info("Loading Wikidata simple entities")
                wikidata_entities = wikidata_utils.collect_wikidata_entities(
                    referenced_ids,
                    wikidata_entities_path=self.wikidata_simple_entities_path,  # type: ignore[arg-type]
                    query_api=self.query_api,
                )

        # substitute all properties and values
        self._logger.info("Substituting values into entities")
        self.substitute_values(
            wikidata_entities=wikidata_entities,
            wikidata_properties=wikidata_properties,
            type_id_to_node=type_id_to_node,
        )

        self._logger.info(f"Resolved entities saved to: {self.entities_output_path}")

    def collect_referenced_ids(self, entities_path: Path | None = None) -> set[str]:
        """Collect all distinct entity ids and reference property values."""
        entities_path = resolve_path(entities_path or self.entities_path)
        referenced_ids = set()

        with open(entities_path, encoding="utf-8") as f:
            for line in f:
                entity = Entity.from_json(line)

                if wikidata_utils.ENTITY_REFERENCE_REGEX_PATTERN.match(entity.entity_id):
                    referenced_ids.add(entity.entity_id)

                for prop_values in entity.properties.values():
                    for value in prop_values:
                        if wikidata_utils.ENTITY_REFERENCE_REGEX_PATTERN.match(value):
                            referenced_ids.add(value)

        self._logger.info(f"Total distinct referenced IDs: {len(referenced_ids):,}")
        return referenced_ids

    def substitute_values(
        self,
        wikidata_entities: dict[str, wikidata_utils.WikidataEntity],
        wikidata_properties: dict[str, dict],
        type_id_to_node: dict[str, dict],
        entities_path: Path | None = None,
        output_path: Path | None = None,
        save_with_minimal_repr: bool | None = None,
    ) -> None:
        """Substitute the actual values into the entities."""
        entities_path = resolve_path(entities_path or self.entities_path)
        output_path = resolve_path(output_path or self.entities_output_path)
        save_with_minimal_repr = save_with_minimal_repr or self.save_with_minimal_repr

        with open(output_path, mode="w", encoding="utf-8") as f_out, open(entities_path, encoding="utf-8") as f_in:
            for line in f_in:
                entity = Entity.from_json(line)

                self.substitute_values_into_entity(
                    entity,
                    wikidata_entities=wikidata_entities,
                    wikidata_properties=wikidata_properties,
                    type_id_to_node=type_id_to_node,
                )

                f_out.write(entity.to_json(minimal_repr=save_with_minimal_repr) + "\n")

    def substitute_values_into_entity(
        self,
        entity: Entity,
        *,
        wikidata_entities: dict[str, wikidata_utils.WikidataEntity],
        wikidata_properties: dict[str, dict],
        type_id_to_node: dict[str, dict],
    ) -> None:
        """Substitute the actual values into the entity."""
        # add de-referenced property values (excluding names)
        unknown_properties = defaultdict(int)

        if self.resolve_property_names or self.resolve_property_values:
            mapped_properties = {"name": entity.properties["name"]}
            for prop, prop_values in entity.properties.items():
                if prop == "name":
                    continue

                prop_name = wikidata_properties.get(prop, {}).get("label")
                if self.resolve_property_names and prop_name is None:
                    self._logger.warning(f"Property {prop} not found in the properties map")

                values = []
                for ref_value in prop_values:
                    value = None

                    if self.resolve_property_values:
                        if prop != wikidata_utils.TypeProperties.INSTANCE_OF.value:
                            ref_entity = wikidata_entities.get(ref_value)
                            if ref_entity is not None:
                                value = ref_entity.name
                        else:
                            entity_type = type_id_to_node.get(ref_value)
                            if entity_type is not None:
                                value = entity_type.get("name")

                    if (not self.resolve_property_values) or (
                        value is None and not wikidata_utils.ENTITY_REFERENCE_REGEX_PATTERN.match(ref_value)
                    ):
                        value = ref_value

                    if value is not None:
                        values.append(value)
                    else:
                        unknown_properties[prop_name] += 1

                if values:
                    key = prop
                    vals = prop_values

                    if self.resolve_property_names:
                        key = prop_name or prop

                    if self.resolve_property_values:
                        vals = values

                    mapped_properties[key] = vals
            entity.properties = mapped_properties

        if self.attach_types:
            entity_types = wikidata_entities[entity.entity_id].type if entity.entity_id in wikidata_entities else []

            if self.resolve_attached_types:
                entity_types = [type_id_to_node[t]["name"] if t in type_id_to_node else t for t in entity_types]

            entity.metadata["type"] = entity_types

        for prop, count in sorted(unknown_properties.items(), key=lambda x: x[1], reverse=True):
            self._logger.warning(f"Unknown values for property {prop}: {count:,}")
