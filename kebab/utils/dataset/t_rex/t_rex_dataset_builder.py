"""
Build entity fragment datasets based on T-REx and Wikidata.

Required inputs:
- T-REx entity fragments
- Wikidata simple entities
- Wikidata properties
- Wikidata type hierarchy

Workflow:
- Filter fragment contents based on the number of properties they have, remove names not found on Wikidata, etc.
- Filter fragments based on how big the entity is, and what types it has.
- Substitute (de-reference) all properties and values.
- Write the dataset.

Example output:
{"entity_id": "Q25295", "properties": {"names": ["language family"]}, "source_ids": ["81317a6457106bc3ae89d179d65ec7ed", "http://www.wikidata.org/entity/Q33199"], "evidence_map": {}, "entity_types": ["languoid class"]}
{"entity_id": "Q11708", "properties": {"names": ["Southeast Asia"]}, "source_ids": ["81317a6457106bc3ae89d179d65ec7ed", "http://www.wikidata.org/entity/Q33199"], "evidence_map": {}, "entity_types": ["geographic region"]}
{"entity_id": "Q668", "properties": {"names": ["India"], "shares border with": ["People's Republic of China", "Bangladesh", "Nepal"]}, "source_ids": ["81317a6457106bc3ae89d179d65ec7ed", "http://www.wikidata.org/entity/Q33199"], "evidence_map": {},
"""

from __future__ import annotations

import gc
import json
import logging
import pathlib
import re
import typing
from collections import defaultdict
from collections.abc import Iterable

from kebab.contracts.entity import Entity
from kebab.utils.dataset.wikidata import wikidata_utils


class TRexDatasetBuilder:
    """Build entity fragment datasets based on T-REx and Wikidata."""

    ENTITIES_DATASET_FILENAME: str = "t_rex_entities_dataset.jsonl"
    ENTITY_EMBEDDINGS_FILENAME: str = "t_rex_entities_embeddings.jsonl"

    DISALLOWED_NAMES: typing.ClassVar[set[str]] = {"", "It", "He", "She", "They"}

    SUBSTITUTE_ACTUAL_NAME_IF_MISSING: bool = True

    WIKIDATA_REFERENCE_REGEX_PATTERN: re.Pattern = re.compile(r"Q\d+")

    # For measuring properties negatively affected by dereferencing
    _DEBUG_UNKNOWN_PROPERTIES: typing.ClassVar[dict[str, int]] = defaultdict(int)

    def __init__(
        self,
        *,
        fragments_path: pathlib.Path,
        wikidata_simple_entities_path: pathlib.Path,
        wikidata_properties_path: pathlib.Path,
        wikidata_type_hierarchy_path: pathlib.Path,
        output_dir: pathlib.Path,
        type_ids: list[str] | None = None,
        min_name_count: int = 1,
        min_fragment_property_count: int = 0,
        min_entity_fragment_count: int = 0,
        max_entity_fragment_count: int | None = None,
    ):
        """Initialize the dataset creator."""
        self._logger: logging.Logger = logging.getLogger(__name__)

        self.fragments_path: pathlib.Path = fragments_path
        self.wikidata_simple_entities_path: pathlib.Path = wikidata_simple_entities_path
        self.wikidata_properties_path: pathlib.Path = wikidata_properties_path
        self.wikidata_type_hierarchy_path: pathlib.Path = wikidata_type_hierarchy_path

        self.output_dir: pathlib.Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.type_ids: list[str] = type_ids or []
        self.min_name_count: int = min_name_count
        self.min_fragment_property_count: int = min_fragment_property_count
        self.min_entity_fragment_count: int = min_entity_fragment_count
        self.max_entity_fragment_count: int | None = max_entity_fragment_count

        self.entities_output_path = self.output_dir / self.ENTITIES_DATASET_FILENAME
        self.embeddings_output_path = self.output_dir / self.ENTITY_EMBEDDINGS_FILENAME

    def run(self) -> None:
        """Run the dataset building pipeline."""
        # load the input fragments
        fragments = self.load_fragments()
        referenced_ids = self.collect_referenced_ids(fragments)

        # load wikidata properties and type hierarchy
        wikidata_properties = wikidata_utils.load_properties(self.wikidata_properties_path)
        _, type_id_to_node = self.load_type_hierarchy()

        # get wikidata entities, properties, and type hierarchy
        wikidata_entities = self.get_wikidata_simple_entities(referenced_ids)

        # filter fragment contents
        filtered_fragments = self.filter_fragments(
            (f for fragments in fragments.values() for f in fragments),
            wikidata_entities=wikidata_entities,
            wikidata_properties=wikidata_properties,
        )

        fragments = defaultdict(list)
        for fragment in filtered_fragments:
            fragments[fragment.entity_id].append(fragment)

        self._logger.info(f"Content-filtered fragments: {sum(len(v) for v in fragments.values()):,}")

        # filter fragments based on how big the entity is
        fragments = self.filter_entities_by_fragment_count(fragments)

        # substitute (de-reference) all properties and values
        for entity_fragments in fragments.values():
            for fragment in entity_fragments:
                self.substitute_values(
                    fragment,
                    wikidata_entities=wikidata_entities,
                    wikidata_properties=wikidata_properties,
                    type_id_to_node=type_id_to_node,
                )

        for prop, count in sorted(self._DEBUG_UNKNOWN_PROPERTIES.items(), key=lambda x: x[1], reverse=True):
            self._logger.warning(f"Unknown values for property {prop}: {count:,}")

        # write the dataset
        self.write_dataset(fragments)

    def load_fragments(
        self,
        fragments_path: pathlib.Path | None = None,
        remove_duplicates: bool = True,
    ) -> dict[str, list[Entity]]:
        """Load entity fragments from file."""
        fragments_path = fragments_path or self.fragments_path

        fragments = defaultdict(list)
        seen = set()
        duplicates_removed = 0

        with open(fragments_path, encoding="utf-8") as f:
            for line in f:
                fragment = Entity.from_json(line)

                if remove_duplicates:
                    values = set()
                    values.add(fragment.entity_id)
                    values.update(fragment.properties["names"])
                    for prop, prop_values in fragment.properties.items():
                        values.add(prop)
                        values.update(prop_values)

                    record = tuple(sorted(values))

                    if record in seen:
                        duplicates_removed += 1
                        continue

                    seen.add(record)

                fragments[fragment.entity_id].append(fragment)

        if duplicates_removed:
            self._logger.warning(f"Removed {duplicates_removed:,} duplicate fragments.")

        self._logger.info(
            f"Loaded fragments: entities: {len(fragments):,}, fragments: {sum(len(v) for v in fragments.values()):,}"
        )

        return fragments

    def collect_referenced_ids(self, fragments: dict[str, list[Entity]]) -> set[str]:
        """Collect all distinct entity ids and reference property values."""
        # collect all distinct reference property values
        referenced_ids = set()

        for entity_id, entity_fragments in fragments.items():
            referenced_ids.add(entity_id)

            for fragment in entity_fragments:
                for prop_values in fragment.properties.values():
                    for value in prop_values:
                        if self.WIKIDATA_REFERENCE_REGEX_PATTERN.match(value):
                            referenced_ids.add(value)

        self._logger.info(f"Total distinct referenced IDs: {len(referenced_ids):,}")

        return referenced_ids

    def get_wikidata_simple_entities(
        self,
        ids_to_include: set[str],
        wikidata_simple_entities_path: pathlib.Path | None = None,
        query_api: bool = False,
        append_to_file: bool = False,
    ) -> dict[str, dict]:
        """
        Load a list of simple entities from a file.
        These are then used to de-reference entity-valued properties in the dataset, and to attach entity types.
        """
        wikidata_simple_entities_path = wikidata_simple_entities_path or self.wikidata_simple_entities_path

        if wikidata_simple_entities_path is None:
            raise ValueError("Path to the list of simple entities is not provided.")

        self._logger.info(f"Loading Wikidata simple entities from {wikidata_simple_entities_path}")

        required_ids = set(ids_to_include)
        entities = {}
        with open(wikidata_simple_entities_path, encoding="utf-8") as f:
            for line in f:
                entity = json.loads(line)

                if entity["id"] not in required_ids:
                    continue

                entities[entity["id"]] = entity
                required_ids.remove(entity["id"])

                # if len(entities) > 1000:
                #     break

        if required_ids:
            if query_api:
                self._logger.info(f"Querying {len(required_ids):,} entities via the Wikidata API")

                q_entities = wikidata_utils.query_entities_via_api(required_ids)

                if q_entities is None:
                    self._logger.error("Failed to query entities via the Wikidata API.")
                    return entities

                q_entities = {
                    k: wikidata_utils.convert_to_simple_entity(e, properties=[], include_descriptions=False)
                    for k, e in q_entities.items()
                    if e is not None
                }

                if append_to_file:
                    with open(wikidata_simple_entities_path, mode="a", encoding="utf-8") as f:
                        for entity in q_entities.values():
                            f.write(json.dumps(entity, ensure_ascii=False) + "\n")
                    self._logger.info(f"Appended {len(q_entities):,} entities to {wikidata_simple_entities_path}")

                entities.update(q_entities)
            else:
                self._logger.warning(f"Failed to find {len(required_ids):,} entities in the list of simple entities")

        self._logger.info(f"Found {len(entities):,} Wikidata entities.")

        return entities

    def load_type_hierarchy(
        self, type_hierarchy_path: pathlib.Path | None = None
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """Load the hierarchy of Wikidata types."""
        self._logger.info("Loading the hierarchy of Wikidata types.")
        input_path = type_hierarchy_path or self.wikidata_type_hierarchy_path

        if input_path is None:
            raise ValueError("Path to the hierarchy of Wikidata types is not provided.")

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

    def filter_fragments(
        self,
        entity_fragments: Iterable[Entity],
        *,
        wikidata_entities: dict[str, dict] | None = None,
        wikidata_properties: dict[str, dict] | None = None,
        filter_to_wikidata_names: bool = True,
        min_name_count: int | None = None,
        min_property_count: int | None = None,
        type_ids: list[str] | None = None,
    ) -> Iterable[Entity]:
        """Filter fragments based on the number of properties they have and other criteria."""
        min_property_count = min_property_count if min_property_count is not None else self.min_fragment_property_count
        min_name_count = min_name_count if min_name_count is not None else self.min_name_count
        type_ids = type_ids or self.type_ids

        filtered_out_properties = defaultdict(int)

        def filter_contents(fragment: Entity) -> Entity:
            """Filter fragment contents."""
            names = (name for name in fragment.properties["names"] if name not in self.DISALLOWED_NAMES)

            # filter out names that are not in Wikidata
            if filter_to_wikidata_names and wikidata_entities:
                wikidata_entity = wikidata_entities[fragment.entity_id]
                wikidata_names = {wikidata_entity["name"], *wikidata_entity["aliases"]}
                names = (name for name in names if name in wikidata_names)

            fragment.properties["names"] = list(names)

            # filter out properties that are not in the provided properties map
            if wikidata_properties:
                filtered_properties = {
                    prop: values
                    for prop, values in fragment.properties.items()
                    if prop in wikidata_properties or prop == "names"
                }

                for prop in fragment.properties:
                    if prop not in filtered_properties:
                        filtered_out_properties[prop] += 1

                fragment.properties = filtered_properties

            return fragment

        for f in entity_fragments:
            if (filter_to_wikidata_names or type_ids) and wikidata_entities and f.entity_id not in wikidata_entities:
                continue

            fragment = filter_contents(f)

            if not fragment.properties["names"]:
                if len(fragment.properties) <= 1:
                    continue

                if not self.SUBSTITUTE_ACTUAL_NAME_IF_MISSING or not wikidata_entities:
                    continue

                fragment.properties["names"] = [wikidata_entities[fragment.entity_id]["name"]]

            if min_name_count and len(fragment.properties["names"]) < min_name_count:
                continue

            if len(fragment.properties) < min_property_count:
                continue

            if type_ids and wikidata_entities:
                entity_types = wikidata_entities[fragment.entity_id]["types"]
                if not any(t in type_ids for t in entity_types):
                    continue

            yield fragment

        for prop, count in filtered_out_properties.items():
            self._logger.warning(f"Filtered out {count:,} values of property {prop}")

    def filter_entities_by_fragment_count(
        self,
        fragments: dict[str, list[Entity]],
        min_entity_fragment_count: int | None = None,
        max_entity_fragment_count: int | None = None,
    ) -> dict[str, list[Entity]]:
        """Filter entities by the number of fragments they have."""
        min_entity_fragment_count = (
            min_entity_fragment_count if min_entity_fragment_count is not None else self.min_entity_fragment_count
        )

        max_entity_fragment_count = (
            max_entity_fragment_count if max_entity_fragment_count is not None else self.max_entity_fragment_count
        )

        fragments = {
            entity_id: entity_fragments
            for entity_id, entity_fragments in fragments.items()
            if min_entity_fragment_count <= len(entity_fragments)
            and (max_entity_fragment_count is None or (len(entity_fragments) <= max_entity_fragment_count))
        }

        gc.collect()

        self._logger.info(
            f"Filtered entities: {len(fragments):,}, fragments: {sum(len(v) for v in fragments.values()):,}"
        )

        return fragments

    def write_dataset(self, fragments: dict[str, list[Entity]]) -> None:
        """Write the entity dataset."""
        with open(self.entities_output_path, mode="w", encoding="utf-8") as f:
            for entity_fragments in fragments.values():
                for fragment in entity_fragments:
                    line = fragment.to_json()
                    f.write(line + "\n")

    def substitute_values(
        self,
        fragment: Entity,
        *,
        wikidata_entities: dict[str, dict],
        wikidata_properties: dict[str, dict],
        type_id_to_node: dict[str, dict],
    ) -> None:
        """Substitute the actual values in the fragment."""
        # add de-referenced property values (excluding names)
        mapped_properties = {"names": fragment.properties["names"]}

        for prop, prop_values in fragment.properties.items():
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
                    entity = wikidata_entities.get(ref_value)
                    if entity is not None:
                        value = entity["name"]
                else:
                    entity_type = type_id_to_node.get(ref_value)
                    if entity_type is not None:
                        value = entity_type.get("name")

                if value is None and not self.WIKIDATA_REFERENCE_REGEX_PATTERN.match(ref_value):
                    value = ref_value

                if value is not None:
                    values.append(value)
                else:
                    self._DEBUG_UNKNOWN_PROPERTIES[prop_name] += 1

            if values:
                mapped_properties[prop_name] = values

        fragment.properties = mapped_properties

        entity_types = wikidata_entities.get(fragment.entity_id, {}).get("types", [])

        # entity_type_values = list({type_id_to_node[t]["name"] for t in entity_types if t in type_id_to_node})
        # fragment.entity_types = entity_type_values
        fragment._entity_types = entity_types  # noqa: SLF001

    # TODO(pmyshkov): Move this method to a separate dataset access class
    @classmethod
    def collect_all_subtypes(cls, graph: dict[str, dict], type_id: str) -> set[str]:
        """Collect all subtypes of the given type."""
        type_id_to_node = {}
        for node in graph.values():
            for node_id in node["merged_ids"]:
                type_id_to_node[node_id] = node

            for node_id in node["redirect_from_ids"]:
                type_id_to_node[node_id] = node

        type_id = type_id_to_node[type_id]["id"]

        subtypes = set()
        stack = [type_id]
        while stack:
            current_type_id = stack.pop()

            if current_type_id not in graph:
                logging.warning(f"Type {current_type_id} not found in the graph")
                continue

            for merged_id in graph[current_type_id]["merged_ids"]:
                subtypes.add(merged_id)

            for redirect_id in graph[current_type_id]["redirect_from_ids"]:
                subtypes.add(redirect_id)

            logging.info(f"Added {current_type_id} ({graph[current_type_id]['name']}) to the subtypes")

            for child_id in graph[current_type_id]["children"]:
                stack.append(child_id)  # noqa: PERF402

        return subtypes
