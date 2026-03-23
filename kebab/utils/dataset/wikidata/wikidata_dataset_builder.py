# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Build a dataset of entities of a particular type (and its subtypes) from Wikidata.
Include the names, descriptions, aliases and selected properties of the entities.

Necessary inputs:
- Wikidata dump in JSON format (e.g. from https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz).
- A file with the type hierarchy for Wikidata.

The workflow of building the dataset is as follows:
1. Collect all subtypes of the given type.
2. Filter the full JSON dump to the entities that have the given type and extract the required fields.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from kebab.utils.dataset.wikidata import wikidata_utils


class WikidataDatasetBuilder:
    """Build a dataset of entities of a particular type (and its subtypes) from Wikidata."""

    WIKIDATA_DATASET_OUTPUT_FILENAME: str = "wikidata_dataset.jsonl"

    _logger: logging.Logger
    wikidata_dump_path: Path | None
    wikidata_type_hierarchy_path: Path | None
    wikidata_properties_path: Path | None
    wikidata_simple_entities_path: Path
    output_dir: Path
    type_ids: list[str]
    property_ids: list[str]

    def __init__(
        self,
        *,
        wikidata_json_dump_path: Path | None = None,
        wikidata_type_hierarchy_path: Path | None = None,
        wikidata_properties_path: Path | None = None,
        wikidata_simple_entities_path: Path,
        output_dir: Path | None = None,
        type_ids: list[str] | None = None,
        property_ids: list[str] | None = None,
    ):
        """Initialize the WikidataDatasetBuilder."""
        self._logger = logging.getLogger(self.__class__.__name__)
        self.wikidata_dump_path = wikidata_json_dump_path
        self.wikidata_type_hierarchy_path = wikidata_type_hierarchy_path
        self.wikidata_properties_path = wikidata_properties_path
        self.wikidata_simple_entities_path = wikidata_simple_entities_path
        self.output_dir = output_dir or Path.cwd()
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.type_ids = type_ids or []
        self.property_ids = property_ids or []

    def run(self) -> None:
        """Run the extraction process."""
        self._logger.info("Running the Wikidata dataset builder.")

        # load type hierarchy
        graph, type_id_to_node = self.load_type_hierarchy(self.wikidata_type_hierarchy_path)
        all_subtypes = {
            st for t in self.type_ids for st in wikidata_utils.collect_all_subtypes(graph, type_id_to_node, t)
        }
        self._logger.info(f"Found {len(all_subtypes):,d} subtypes")

        # load properties
        wikidata_properties = self.load_properties(self.wikidata_properties_path)
        if not self.property_ids:
            self.property_ids = []
            for prop_key, prop in wikidata_properties.items():
                label = prop["label"]
                if "ID" in label:
                    continue

                self.property_ids.append(prop_key)

        if not all(p.startswith("P") for p in self.property_ids):
            property_ids = self.get_property_ids(wikidata_properties, self.property_ids)
        else:
            property_ids = self.property_ids

        # extract entities
        entities = list(self.extract_entities(all_subtypes, property_ids, self.wikidata_dump_path))

        referenced_ids = self.collect_referenced_ids(entities)
        wikidata_entities = wikidata_utils.collect_wikidata_entities(referenced_ids, self.wikidata_simple_entities_path)

        for entity in entities:
            self.substitute_values(
                entity,
                wikidata_entities=wikidata_entities,
                wikidata_properties=wikidata_properties,
                type_id_to_node=type_id_to_node,
            )

        # save dataset
        with open(self.output_dir / self.WIKIDATA_DATASET_OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            for entity in entities:
                f.write(json.dumps(entity, ensure_ascii=False) + "\n")

    def load_type_hierarchy(self, type_hierarchy_path: Path | None = None) -> tuple[dict[str, dict], dict[str, dict]]:
        """Load the hierarchy of Wikidata types."""
        self._logger.info("Loading the hierarchy of Wikidata types")
        input_path = type_hierarchy_path or self.wikidata_type_hierarchy_path

        if input_path is None:
            raise ValueError("Path to the hierarchy of Wikidata types is not provided")

        graph = {}
        type_id_to_node = {}

        with open(input_path, encoding="utf-8") as f:
            for line in f:
                node = json.loads(line.strip())
                graph[node["id"]] = node
                for node_id in node["merged_ids"]:
                    type_id_to_node[node_id] = node

                for node_id in node["redirect_from_ids"]:
                    type_id_to_node[node_id] = node

        self._logger.info(f"Type hierarchy contains {len(graph):,d} nodes")

        return graph, type_id_to_node

    def load_properties(self, properties_path: Path | None) -> dict[str, dict]:
        """Load the properties of Wikidata."""
        self._logger.info("Loading the properties of Wikidata.")
        input_path = properties_path or self.wikidata_properties_path

        if input_path is None:
            raise ValueError("Path to the properties of Wikidata is not provided.")

        with open(input_path, encoding="utf-8") as f:
            properties = json.load(f)

        self._logger.info(f"Loaded {len(properties):,d} properties")

        return properties

    def get_property_ids(self, properties: dict[str, dict], property_names: Iterable[str]) -> list[str]:
        """Get the property IDs for the given property names."""
        name_to_ids = defaultdict(list)
        for property_id, property_data in properties.items():
            name_to_ids[property_data["label"]].append(property_id)

        property_ids = []
        for property_name in property_names:
            if property_name == "names":
                continue

            prop_id = name_to_ids.get(property_name)
            if not prop_id:
                raise ValueError(f"Property {property_name} not found in the properties.")

            if len(prop_id) > 1:
                raise ValueError(f"Multiple properties found for {property_name}: {prop_id}")

            property_ids.append(prop_id[0])

        return property_ids

    def extract_entities(
        self,
        type_ids: Iterable[str],
        property_ids: Iterable[str],
        wikidata_dump_path: Path | None = None,
    ) -> Iterable[wikidata_utils.WikidataEntity]:
        """Extract entities of the given types from Wikidata."""
        input_path = wikidata_dump_path or self.wikidata_dump_path

        if input_path is None:
            raise ValueError("Wikidata JSON dump path is required.")

        self._logger.info(f"Extracting entities from {self.wikidata_dump_path}")

        type_ids = set(type_ids)

        entities = wikidata_utils.extract_wikidata_entities_from_dump(
            wikidata_json_dump_path=input_path,
            properties=property_ids,
            input_entity_predicate=lambda x: x.get("type") == "item",
            output_entity_predicate=lambda x: x.instance_of is not None and any(t in type_ids for t in x.instance_of),
        )

        return entities

    def collect_referenced_ids(self, entities: Iterable[wikidata_utils.WikidataEntity]) -> set[str]:
        """Collect all distinct entity ids and reference property values."""
        referenced_ids = set()

        for entity in entities:
            for prop_values in entity.properties.values():
                for value in prop_values:
                    if value is None:
                        continue

                    if wikidata_utils.ENTITY_REFERENCE_REGEX_PATTERN.match(value):
                        referenced_ids.add(value)

        self._logger.info(f"Total distinct referenced IDs: {len(referenced_ids):,}")

        return referenced_ids

    def substitute_values(
        self,
        entity: wikidata_utils.WikidataEntity,
        *,
        wikidata_entities: dict[str, wikidata_utils.WikidataEntity],
        wikidata_properties: dict[str, dict],
        type_id_to_node: dict[str, dict],
    ) -> None:
        """Substitute the actual values in the entity."""
        entity.instance_of = [type_id_to_node[t]["name"] for t in entity.instance_of if t in type_id_to_node]

        mapped_properties = {}

        for prop, prop_values in entity.properties.items():
            prop_name = wikidata_properties.get(prop, {}).get("label")

            if prop_name is None:
                self._logger.warning(f"Property {prop} not found in the properties map")
                continue

            values = []
            for ref_value in prop_values:
                if ref_value is None:
                    continue

                value = None

                ref_entity = wikidata_entities.get(ref_value)
                if ref_entity is not None:
                    value = ref_entity.name

                if value is None and not wikidata_utils.ENTITY_REFERENCE_REGEX_PATTERN.match(ref_value):
                    value = ref_value

                if value is not None:
                    values.append(value)

            if values:
                mapped_properties[prop_name] = list(set(values))

        entity.properties = mapped_properties
