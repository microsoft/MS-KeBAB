"""
Given a collection of entities, substitute the actual values for properties, replace property ids with property names,
and attach types to the entities. The values are resolved from the Wikidata dump of simple entities, followed by
quries to the Wikidata API for missing information.

Required inputs:
- Entities to resolve
- Wikidata simple entities (filtered dump)
- Wikidata properties map
- Wikidata type hierarchy
"""

from __future__ import annotations

import logging
import pathlib
from collections import defaultdict

from kebab.contracts.entity import Entity
from kebab.utils.dataset.wikidata import wikidata_utils


class WikidataResolver:
    """A class for resolving Wikidata entities and properties."""

    def __init__(
        self,
        *,
        entities_path: pathlib.Path,
        wikidata_simple_entities_path: pathlib.Path,
        wikidata_properties_path: pathlib.Path,
        wikidata_type_hierarchy_path: pathlib.Path,
        attach_types: bool = True,
        resolves_types: bool = False,
        resolve_property_names: bool = True,
        resolve_property_values: bool = True,
        query_api: bool = True,
        output_dir: pathlib.Path,
    ):
        """Initialize the Wikidata resolver."""
        self._logger: logging.Logger = logging.getLogger(__name__)

        self.entities_path: pathlib.Path = entities_path
        self.wikidata_simple_entities_path: pathlib.Path = wikidata_simple_entities_path
        self.wikidata_properties_path: pathlib.Path = wikidata_properties_path
        self.wikidata_type_hierarchy_path: pathlib.Path = wikidata_type_hierarchy_path

        self.resolve_property_names: bool = resolve_property_names
        self.resolve_property_values: bool = resolve_property_values
        self.attach_types: bool = attach_types
        self.resolves_types: bool = resolves_types
        self.query_api: bool = query_api

        self.output_dir: pathlib.Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.entities_output_path = self.output_dir / self.entities_path.name

        if not self.entities_path.exists():
            raise FileNotFoundError(f"Entities file not found: {self.entities_path}")

        if not self.wikidata_simple_entities_path.exists():
            raise FileNotFoundError(f"Simple entities file not found: {self.wikidata_simple_entities_path}")

        if not self.wikidata_properties_path.exists():
            raise FileNotFoundError(f"Properties file not found: {self.wikidata_properties_path}")

        if not self.wikidata_type_hierarchy_path.exists():
            raise FileNotFoundError(f"Type hierarchy file not found: {self.wikidata_type_hierarchy_path}")

        if self.entities_path == self.entities_output_path:
            raise ValueError("Input and output entities paths are the same.")

        if self.entities_output_path.exists():
            self._logger.warning(f"Output entities file already exists: {self.entities_output_path}")
            self._logger.warning("It will be overwritten.")

    def run(self) -> None:
        """Run the dataset building pipeline."""
        # collect all referenced IDs (for which we'll need to get Wikidata entities)
        self._logger.info("Collecting all referenced IDs")
        referenced_ids = self.collect_referenced_ids()

        # load wikidata properties and type hierarchy
        self._logger.info("Loading Wikidata properties")
        wikidata_properties = wikidata_utils.load_properties(self.wikidata_properties_path)

        self._logger.info("Loading Wikidata type hierarchy")
        _, type_id_to_node = wikidata_utils.load_type_hierarchy(self.wikidata_type_hierarchy_path)

        # load and query wikidata entities, properties, and type hierarchy
        self._logger.info("Loading Wikidata simple entities")
        wikidata_entities = wikidata_utils.collect_wikidata_entities(
            referenced_ids, wikidata_entities_path=self.wikidata_simple_entities_path, query_api=self.query_api
        )

        # substitute all properties and values
        self._logger.info("Substituting values into entities")
        self.substitute_values(
            self.entities_path,
            wikidata_entities=wikidata_entities,
            wikidata_properties=wikidata_properties,
            type_id_to_node=type_id_to_node,
            output_path=self.entities_output_path,
        )

    def collect_referenced_ids(self, entities_path: pathlib.Path | None = None) -> set[str]:
        """Collect all distinct entity ids and reference property values."""
        entities_path = entities_path or self.entities_path
        referenced_ids = set()

        with open(entities_path, encoding="utf-8") as f:
            for line in f:
                entity = Entity.from_json(line)
                referenced_ids.add(entity.entity_id)

                for prop_values in entity.properties.values():
                    for value in prop_values:
                        if wikidata_utils.ENTITY_REFERENCE_REGEX_PATTERN.match(value):
                            referenced_ids.add(value)

        self._logger.info(f"Total distinct referenced IDs: {len(referenced_ids):,}")
        return referenced_ids

    def substitute_values(
        self,
        entities_path: pathlib.Path,
        wikidata_entities: dict[str, wikidata_utils.WikidataEntity],
        wikidata_properties: dict[str, dict],
        type_id_to_node: dict[str, dict],
        output_path: pathlib.Path,
    ) -> None:
        """Substitute the actual values into the entities."""
        entities_path = entities_path or self.entities_path
        output_path = output_path or self.entities_output_path

        with open(entities_path, encoding="utf-8") as f:
            entities = [Entity.from_json(line) for line in f]

        for entity in entities:
            self.substitute_values_into_entity(
                entity,
                wikidata_entities=wikidata_entities,
                wikidata_properties=wikidata_properties,
                type_id_to_node=type_id_to_node,
            )

        with open(output_path, mode="w", encoding="utf-8") as f:
            for entity in entities:
                line = entity.to_json()
                f.write(line + "\n")

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
            mapped_properties = {"names": entity.properties["names"]}
            for prop, prop_values in entity.properties.items():
                if prop == "names":
                    continue

                prop_name = wikidata_properties.get(prop, {}).get("label")

                if prop_name is None:
                    self._logger.warning(f"Property {prop} not found in the properties map.")
                    continue

                values = []
                for ref_value in prop_values:
                    value = None

                    if prop != wikidata_utils.TypeProperties.INSTANCE_OF.value:
                        ref_entity = wikidata_entities.get(ref_value)
                        if ref_entity is not None:
                            value = ref_entity.name
                    else:
                        entity_type = type_id_to_node.get(ref_value)
                        if entity_type is not None:
                            value = entity_type.get("name")

                    if value is None and not wikidata_utils.ENTITY_REFERENCE_REGEX_PATTERN.match(ref_value):
                        value = ref_value

                    if value is not None:
                        values.append(value)
                    else:
                        unknown_properties[prop_name] += 1

                if values:
                    key = prop
                    vals = prop_values

                    if self.resolve_property_names:
                        key = prop_name

                    if self.resolve_property_values:
                        vals = values

                    mapped_properties[key] = vals
            entity.properties = mapped_properties

        if self.attach_types:
            entity_types = wikidata_entities[entity.entity_id].types if entity.entity_id in wikidata_entities else []

            if self.resolves_types:
                entity_types = [type_id_to_node[t]["name"] if t in type_id_to_node else t for t in entity_types]

            entity.metadata["types"] = entity_types

        for prop, count in sorted(unknown_properties.items(), key=lambda x: x[1], reverse=True):
            self._logger.warning(f"Unknown values for property {prop}: {count:,}")
