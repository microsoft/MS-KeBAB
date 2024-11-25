# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Extract type hierarchy from Wikidata.

The extraction process runs in two steps:
1. Collect all Wikidata type entities and their reference counts by looking through the dump and then querying
   the Wikidata API.
2. Build the hierarchy of Wikidata type entities, including removing cycles and grandparent links.

Necessary input:
- Wikidata dump in JSON format (e.g. from https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz).

Example output:
- `wikidata_types_hierarchy.jsonl`, each line like:
{"id": "Q13442814",
 "merged_ids": ["Q13442814"],
 "name": "scholarly article",
 "descriptions": "article in an academic publication, ...",
 "aliases": ["article", "academic paper", "research paper", ...],
 "parents": ["Q591041", "Q191067", "Q55915575"],
 "children": ["Q187685", "Q110716513", ...],
 "ref_count": 39393815,
 "subgraph_size": 3,
 "subgraph_height": 2,
 "subgraph_ref_count": 89393815},
...

The workflow of extracting the hierarchy is as follows:
1. Collect all Wikidata type entities and their reference counts by looking through the Wikidata dump.
2. Construct the type entities by querying the Wikidata API, recursively include parent type entities and same-as entities.
3. Build the graph of entities.
4. Convert same-as links into symmetric parent links, if the conditions are met.
5. Find and merge all cycles in the graph.
6. Remove all transitive ancestor links from the graph.
7. Compute subgraph statistics and save the hierarchy to a file.
"""

from __future__ import annotations

import json
import logging
import pathlib
import typing
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from kebab.utils.dataset.wikidata import wikidata_utils
from kebab.utils.dataset.wikidata.wikidata_utils import TypeProperties


class WikidataHierarchyExtractor:
    """Extract type hierarchy from Wikidata."""

    # TODO(pmyshkov): These parameters should eventually be removed and only one workflow should be used.
    # whether to merge entities that are linked with the "same as" property
    MERGE_ON_SAME_AS: bool = True
    # only allow same-as links that exist in both directions
    MERGE_ON_SAME_AS_DISALLOW_ASYMMETRIC: bool = True
    # disallow same-as links where one node is a parent of the other
    MERGE_ON_SAME_AS_DISALLOW_ANCESTOR_LINKS: bool = True
    # only allow same-as links that have a common parent
    MERGE_ON_SAME_AS_ONLY_WITH_SHARED_PARENT: bool = True
    # drop all properties that have an "of" qualifier
    DROP_PROPERTIES_WITH_OF_QUALIFIER: bool = False

    # this is only used for saving cycles as a text file, for debugging or fixing Wikidata
    _DEBUG_SAME_AS_EDGES: typing.ClassVar[set[tuple[str, str]]] = set()

    WIKIDATA_CONCRETE_ENTITIES_FILENAME = "wikidata_concrete_entities.jsonl"
    WIKIDATA_TYPE_ENTITY_IDS_WITH_COUNTS_FILENAME = "wikidata_type_entity_ids_with_ref_counts.jsonl"
    WIKIDATA_TYPE_ENTITIES_FILENAME = "wikidata_type_entities.jsonl"
    WIKIDATA_MERGED_CYCLES_FILENAME = "wikidata_merged_cycles.txt"
    WIKIDATA_REMOVED_GRANDPARENT_LINKS_FILENAME = "wikidata_removed_grandparent_links.txt"
    WIKIDATA_HIERARCHY_FILENAME = "wikidata_type_hierarchy.jsonl"

    def __init__(
        self,
        *,
        wikidata_json_dump_path: pathlib.Path | None = None,
        output_dir: pathlib.Path | None = None,
    ):
        """Initialize the Wikidata hierarchy extractor."""
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.wikidata_dump_path: pathlib.Path | None = wikidata_json_dump_path
        self.output_dir: pathlib.Path = output_dir or pathlib.Path.cwd()

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Run the extraction process."""
        # preprocessing - collect all Wikidata type entities (if not already done)
        self.run_collect_wikidata_type_entities()

        # build the Wikidata hierarchy
        self.run_build_wikidata_hierarchy()

    def run_collect_wikidata_type_entities(self):
        """
        Collect all Wikidata type entities and their reference counts by looking through the dump and then querying
        the Wikidata API.

        None of these steps are done if the output files already exist.
        """
        # extract concrete entities (those that have an "instance of" property)
        self.extract_concrete_entities_from_wikidata_dump()

        # extract type entities from the concrete entities
        self.extract_type_entities_from_concrete_entities()

        # construct the type entities by querying the Wikidata API, include parent type entities
        self.construct_wikidata_type_entities()

    def run_build_wikidata_hierarchy(self, type_entity_ids_whitelist_path: pathlib.Path | None = None):
        """Build the hierarchy of Wikidata type entities."""
        # load the Wikidata type entities
        type_entities = self.load_type_entities()

        # get the graph of entities with links to parent and children entities
        if type_entity_ids_whitelist_path and type_entity_ids_whitelist_path.exists():
            with open(type_entity_ids_whitelist_path, encoding="utf-8") as f:
                whitelist = json.load(f)
        else:
            whitelist = {}

        self._logger.info(f"Whitelist: {len(whitelist):,} entities.")
        graph = self.build_type_entities_graph(type_entities, allowed_ids=whitelist)

        # merge cycles
        self._logger.info("Merging cycles in the graph.")
        merged_cycles = self.merge_cycles(graph)
        self.write_node_sequences(merged_cycles, type_entities, self.output_dir / self.WIKIDATA_MERGED_CYCLES_FILENAME)

        # remove grandparent links
        self._logger.info("Removing grandparent links in the graph.")
        removed_links = self.remove_grandparent_links(graph)
        self.write_node_sequences(
            removed_links, type_entities, self.output_dir / self.WIKIDATA_REMOVED_GRANDPARENT_LINKS_FILENAME
        )

        # add subgraph sizes
        self._logger.info("Adding statistics to the nodes and saving the hierarchy.")
        self.add_subgraph_sizes(graph, type_entities)

        # write the hierarchy to a file
        self.write_wikidata_hierarchy(graph, type_entities)

    def extract_concrete_entities_from_wikidata_dump(
        self,
        wikidata_dump_path: pathlib.Path | None = None,
        working_dir: pathlib.Path | None = None,
        skip_when_output_exists: bool = True,
    ) -> pathlib.Path:
        """
        Extract all entities from the Wikidata dump that have an "instance of" property ("concrete" entities).
        Keep only simplified objects - entity ID and the values of the "instance of" property.
        This is a simple preprocessing step to reduce the amount of data to process later and enable quicker iteration.
        (The full entity dump is >1.5TB).

        This is to later extract all type entities (those that are the values of the "instance of" property of other
        entities).
        """
        input_path = wikidata_dump_path or self.wikidata_dump_path
        if input_path is None:
            raise ValueError("Path to the Wikidata dump is not provided.")

        working_dir = working_dir or self.output_dir
        output_path = working_dir / self.WIKIDATA_CONCRETE_ENTITIES_FILENAME

        if skip_when_output_exists and output_path.exists():
            self._logger.info(f"Skipping extraction of concrete entities, output file exists: {output_path}")
            return output_path

        with open(output_path, mode="w", encoding="utf-8", newline="\n") as f:
            entities = wikidata_utils.extract_simple_entities_from_dump(
                wikidata_json_dump_path=input_path,
                properties=[TypeProperties.INSTANCE_OF.value],
                input_entity_predicate=lambda x: x.get("type") == "item",
                output_entity_predicate=lambda x: len(x["types"]) > 0,
                include_descriptions=False,
            )

            for entity in entities:
                f.write(json.dumps(entity) + "\n")

        return output_path

    def extract_type_entities_from_concrete_entities(
        self,
        concrete_entities_path: pathlib.Path | None = None,
        working_dir: pathlib.Path | None = None,
        skip_when_output_exists: bool = True,
    ) -> pathlib.Path:
        """
        Extract all type entities from the concrete entities in Wikidata dump,
        i.e. collect all the values of the "instance of" property of all entities in the dump.

        Also collect the number of references to the type entities for future use in model training.
        This is populated in the "ref_count" field of the output entities.

        Note that this might not necessarily result in all possible type entities, e.g. an entity might be a parent
        of a type entity (e.g. "human" is a subclass of "mammal"), but not have a single concrete entity that
        is an "instance of" it (e.g. all entities might be either "human" or another specific type, but not "mammal").
        """
        working_dir = working_dir or self.output_dir
        concrete_entities_path = concrete_entities_path or (working_dir / self.WIKIDATA_CONCRETE_ENTITIES_FILENAME)
        output_path = working_dir / self.WIKIDATA_TYPE_ENTITY_IDS_WITH_COUNTS_FILENAME

        if skip_when_output_exists and output_path.exists():
            self._logger.info(f"Skipping extraction of type entities, output file exists: {output_path}")
            return output_path

        counter = defaultdict(int)
        with open(concrete_entities_path, encoding="utf-8") as f:
            for line in f:
                entity = json.loads(line.strip())
                for value in entity["types"]:
                    counter[value] += 1

        with open(output_path, mode="w", encoding="utf-8", newline="\n") as f:
            for entity_id, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
                f.write(json.dumps({"id": entity_id, "ref_count": count}) + "\n")

        self._logger.info(f"Saved {len(counter):,} Wikidata type entity IDs to {output_path}")

        return output_path

    def query_for_type_entities_via_api(self, known_type_entity_ids: list[str]) -> dict[str, dict]:
        """
        Query the Wikidata API for the full information on all the type entities available.

        This includes collecting all parent types of the entities in the list via the "subclass of" property,
        recursively querying them until no new entities are found.

        The returned dictionary might not contain some of the passed entity IDs - this happens due to redirects.
        """
        type_entities = {}
        to_query = set(known_type_entity_ids)

        properties = [TypeProperties.SUBCLASS_OF.value, TypeProperties.SAME_AS.value]

        # redirect map:
        # - redirects are when we query for one ID but get an entity with another ID from the API
        # - we need to keep track of these to update the parent links in the entities
        redirect_map = {}

        prohibited_qualifiers = []

        if self.DROP_PROPERTIES_WITH_OF_QUALIFIER:
            prohibited_qualifiers.append("P642")  # deprecated "of" qualifier

        while to_query:
            # query the entities
            new_entities = wikidata_utils.query_entities_via_api(to_query) or {}
            new_entities = {
                k: wikidata_utils.convert_to_simple_entity(
                    e, properties=properties, prohibited_qualifiers=prohibited_qualifiers
                )
                for k, e in new_entities.items()
            }

            # update the redirect map
            new_redirect_map = {k: e["id"] for k, e in new_entities.items()}
            redirect_map.update(new_redirect_map)

            type_entities.update(new_entities)

            # collect the parent type entities and same-as entities
            to_query = set()
            for entity in new_entities.values():
                to_query.update(entity.get("properties", {}).get(TypeProperties.SUBCLASS_OF.value, []))
                to_query.update(entity.get("properties", {}).get(TypeProperties.SAME_AS.value, []))

            # filter out the entities that we already have
            to_query -= set(type_entities.keys())

        self._logger.info(
            f"Collected {len(type_entities):,} type entities,"
            f" including {len(type_entities) - len(known_type_entity_ids):,} additional parent entities."
        )

        if redirect_map:
            self._logger.info(f"Redirect map: {len(redirect_map):,} entries.")

        # apply the redirect map - update the parent links
        for entity in type_entities.values():
            parents = entity.get("properties", {}).get(TypeProperties.SUBCLASS_OF.value, [])
            entity["properties"][TypeProperties.SUBCLASS_OF.value] = [redirect_map.get(p, p) for p in parents]

        # rebuild the map since the IDs might have changed due to redirects
        type_entities = {e["id"]: e for e in type_entities.values()}

        # attach the redirect map
        redirect_from_map = defaultdict(list)
        for k, v in redirect_map.items():
            if k != v:
                redirect_from_map[v].append(k)

        for entity in type_entities.values():
            entity["redirect_from"] = redirect_from_map.get(entity["id"], [])

        return type_entities

    def construct_wikidata_type_entities(
        self,
        entity_ids_with_ref_counts_path: pathlib.Path | None = None,
        working_dir: pathlib.Path | None = None,
        skip_when_output_exists: bool = True,
    ) -> pathlib.Path:
        """
        Construct all the type entities from Wikidata.

        This is done by querying the Wikidata API to get the full information about the entities and then
        recursively querying for the base entities.
        """
        working_dir = working_dir or self.output_dir
        entity_ids_with_ref_counts_path = entity_ids_with_ref_counts_path or (
            working_dir / self.WIKIDATA_TYPE_ENTITY_IDS_WITH_COUNTS_FILENAME
        )

        output_path = self.output_dir / self.WIKIDATA_TYPE_ENTITIES_FILENAME

        if skip_when_output_exists and output_path.exists():
            self._logger.info(f"Skipping construction of type entities, output file exists: {output_path}")
            return output_path

        with open(entity_ids_with_ref_counts_path, encoding="utf-8") as f:
            entity_ids_with_counts = {e["id"]: e["ref_count"] for e in (json.loads(line.strip()) for line in f)}

        type_entities = self.query_for_type_entities_via_api(list(entity_ids_with_counts.keys()))

        # attach ref counts
        for entity_id, entity in type_entities.items():
            entity["ref_count"] = entity_ids_with_counts.get(entity_id, 0)

        with open(output_path, mode="w", encoding="utf-8", newline="\n") as f:
            for entity in sorted(type_entities.values(), key=lambda x: x["ref_count"], reverse=True):
                f.write(json.dumps(entity) + "\n")

        self._logger.info(f"Saved {len(type_entities):,} Wikidata type entities to {output_path}")

        return output_path

    def load_type_entities(self, type_entities_path: pathlib.Path | None = None) -> dict[str, dict]:
        """Load the type entities from the file."""
        type_entities_path = type_entities_path or (self.output_dir / self.WIKIDATA_TYPE_ENTITIES_FILENAME)

        with open(type_entities_path, encoding="utf-8") as f:
            type_entities = [json.loads(line.strip()) for line in f]

        # verify consistency
        keys = {e["id"] for e in type_entities}
        assert len(keys) == len(type_entities), "Duplicate entity IDs found."

        for entity in type_entities:
            parents = entity.get("properties", {}).get(TypeProperties.SUBCLASS_OF.value, [])
            assert all(p in keys for p in parents), f"Invalid parent entity ID found in entity {entity['id']}."

        type_entities = {e["id"]: e for e in type_entities}

        self._logger.info(f"Loaded {len(type_entities):,} type entities from {type_entities_path}")

        return type_entities

    def build_type_entities_graph(
        self,
        entities: dict[str, dict],
        allowed_ids: Iterable[str] | None = None,
    ) -> dict[str, dict]:
        """
        Construct the graph of the type entities.
        Each node contains the parent and children entity IDs, and does not contain the content of the entities.

        Args:
            entities: the type entities with their properties
            allowed_ids: the allow-list of IDs of the entities to include in the graph, ignored if None or empty
        """
        # TODO(pmyshkov): Once we're settled on the workflow, switch to using networkx for the graph operations.
        graph = {
            entity["id"]: {
                "entity_id": entity["id"],
                "parents": set(entity.get("properties", {}).get(TypeProperties.SUBCLASS_OF.value, [])),
                "merged_ids": {entity["id"]},
            }
            for entity in entities.values()
        }

        # apply the same-as links by connecting the same-as nodes as parents of each other (creating a back link cycle)
        if self.MERGE_ON_SAME_AS:
            same_as_count = 0
            same_as_skipped_count = 0
            for entity_id in graph:
                same_as = entities[entity_id].get("properties", {}).get(TypeProperties.SAME_AS.value, [])
                for same_as_id in same_as:
                    merge = True

                    if same_as_id in graph:
                        # disallow asymmetric
                        if self.MERGE_ON_SAME_AS_ONLY_WITH_SHARED_PARENT and entity_id not in entities[same_as_id].get(
                            "properties", {}
                        ).get(TypeProperties.SAME_AS.value, []):
                            merge = False

                        # disallow those without a common parent
                        if self.MERGE_ON_SAME_AS_ONLY_WITH_SHARED_PARENT:
                            common_parents = graph[entity_id]["parents"] & graph[same_as_id]["parents"]
                            if not common_parents:
                                merge = False

                        # disallow ancestor links
                        if self.MERGE_ON_SAME_AS_DISALLOW_ANCESTOR_LINKS and (
                            same_as_id in graph[entity_id]["parents"] or entity_id in graph[same_as_id]["parents"]
                        ):
                            merge = False
                    else:
                        merge = False

                    if merge:
                        graph[same_as_id]["parents"].add(entity_id)
                        graph[entity_id]["parents"].add(same_as_id)
                        same_as_count += 1
                        self._DEBUG_SAME_AS_EDGES.add((entity_id, same_as_id))
                        self._DEBUG_SAME_AS_EDGES.add((same_as_id, entity_id))
                    else:
                        same_as_skipped_count += 1

            self._logger.info(f"Applied {same_as_count:,} same-as links, skipped {same_as_skipped_count:,}.")

        # create the children links
        self.rebuild_children_links(graph)

        # apply the list of allowed IDs
        if allowed_ids:
            allowed_ids = set(allowed_ids)
            for entity_id in list(graph.keys()):
                if entity_id not in allowed_ids:
                    # remove from parents
                    for parent_id in graph[entity_id]["parents"]:
                        graph[parent_id]["children"].discard(entity_id)

                    # remove from children
                    for child_id in graph[entity_id]["children"]:
                        graph[child_id]["parents"].discard(entity_id)

                    del graph[entity_id]

        self._logger.info(f"Constructed the graph with {len(graph):,} nodes.")

        # remove self references
        self.remove_self_references(graph)

        # validate the graph
        for entity_id, node in graph.items():
            assert node["entity_id"] == entity_id, f"Invalid entity ID in the node: {node['entity_id']} != {entity_id}"
            assert all(p in graph for p in node["parents"]), f"Invalid parent entity ID in the node {entity_id}."
            assert all(c in graph for c in node["children"]), f"Invalid child entity ID in the node {entity_id}."

        return graph

    def write_wikidata_hierarchy(
        self,
        graph: dict[str, dict],
        type_entities: dict[str, dict],
        output_path: pathlib.Path | None = None,
    ) -> None:
        """Write the Wikidata hierarchy to a file."""
        output_path = output_path or (self.output_dir / self.WIKIDATA_HIERARCHY_FILENAME)

        with open(output_path, mode="w", encoding="utf-8", newline="\n") as f:
            for node in sorted(graph.values(), key=lambda n: type_entities[n["entity_id"]]["ref_count"], reverse=True):
                entity_id = node["entity_id"]
                entity = type_entities[entity_id]
                simple_entity = {
                    "id": entity_id,
                    "merged_ids": list(node["merged_ids"]),
                    "redirect_from_ids": list(
                        {n for merged_id in node["merged_ids"] for n in type_entities[merged_id]["redirect_from"]}
                    ),
                    "name": entity["name"],
                    "descriptions": list({type_entities[merged_id]["description"] for merged_id in node["merged_ids"]}),
                    "aliases": list(
                        {a for merged_id in node["merged_ids"] for a in type_entities[merged_id]["aliases"]}
                        | {type_entities[merged_id]["name"] for merged_id in node["merged_ids"]}
                    ),
                    "parents": list(node["parents"]),
                    "children": list(node["children"]),
                    "ref_count": sum(type_entities[merged_id]["ref_count"] for merged_id in node["merged_ids"]),
                    "subgraph_size": node["subgraph_size"],
                    "subgraph_height": node["subgraph_height"],
                    "subgraph_ref_count": node["subgraph_ref_count"],
                }

                f.write(json.dumps(simple_entity) + "\n")

        self._logger.info(f"Saved the Wikidata hierarchy to {output_path}.")

    @classmethod
    def get_root_entities(cls, graph: dict[str, dict]) -> set[str]:
        """Get the root entities, i.e. those that do not have any parents."""
        return {entity_id for entity_id, node in graph.items() if not node["parents"]}

    @classmethod
    def get_leaf_entities(cls, graph: dict[str, dict]) -> set[str]:
        """Get the leaf entities, i.e. those that do not have any children."""
        return {entity_id for entity_id, node in graph.items() if not node["children"]}

    @classmethod
    def rebuild_children_links(cls, graph: dict[str, dict]) -> None:
        """Rebuild the children links in the entities."""
        for node in graph.values():
            node["children"] = set()

        for entity_id, node in graph.items():
            for parent_id in node["parents"]:
                graph[parent_id]["children"].add(entity_id)

    @classmethod
    def remove_self_references(cls, graph: dict[str, dict]) -> None:
        """Remove self references from the graph."""
        for node in graph.values():
            node["parents"].discard(node["entity_id"])
            node["children"].discard(node["entity_id"])

    @classmethod
    def assert_no_self_references(cls, graph: dict[str, dict]) -> None:
        """Assert that there are no self references in the graph."""
        for node in graph.values():
            assert node["entity_id"] not in node["parents"], f"Self reference in parents for {node['entity_id']}."
            assert node["entity_id"] not in node["children"], f"Self reference in children for {node['entity_id']}."

    @classmethod
    def sort_topologically(
        cls,
        graph: dict[str, dict],
        root_entities_sort_key: Callable[[dict], Any] | None = None,
    ) -> list[str]:
        """Return the IDs of the nodes in the topological order."""
        visited = set()
        result = []

        def visit(entity_id: str) -> None:
            """Visit the entity and its children."""
            if entity_id in visited:
                return

            visited.add(entity_id)

            for child_id in graph[entity_id]["children"]:
                visit(child_id)

            result.append(entity_id)

        root_entities = cls.get_root_entities(graph)
        if root_entities_sort_key:
            root_entities = sorted(root_entities, key=lambda x: root_entities_sort_key(graph[x]))

        for entity_id in root_entities:
            visit(entity_id)

        remaining_in_cycles = set(graph.keys()) - visited

        if remaining_in_cycles:
            for entity_id in remaining_in_cycles:
                visit(entity_id)

        return result

    @classmethod
    def collect_all_backlink_cycles(cls, graph: dict[str, dict]) -> list[tuple[str, ...]]:
        """Collect all paths of length 2 that form a cycle in the graph."""
        cycles = [
            (entity_id, parent_id, entity_id)
            for entity_id, node in graph.items()
            for parent_id in node["parents"]
            if entity_id in graph[parent_id]["parents"] and entity_id < parent_id
        ]

        return cycles

    @classmethod
    def collect_cycles(cls, graph: dict[str, dict], eager: bool = False) -> list[tuple[str, ...]]:
        """
        Collect all paths that form cycles in the graph.

        Not a very efficient implementation, but should be fine for the Wikidata hierarchy.
        Setting `eager` to True will collect some, but not all, cycles, and exit much faster so that the cycles
        can be merged. It is guaranteed to find at least one cycle if any cycles exist.
        """
        if eager:
            cycles = cls.collect_all_backlink_cycles(graph)
            if cycles:
                return cycles

        visited = set()
        cycle_paths: set[tuple] = set()

        def visit(entity_id: str, current_path: list[str], path_nodes: set[str]) -> None:
            """Visit the node and its children, DFS."""
            if eager and entity_id in visited:
                return

            visited.add(entity_id)

            node = graph[entity_id]
            children = node["children"]

            for child_id in list(children):
                if child_id in path_nodes:
                    cycle_path = [*current_path[current_path.index(child_id) :], child_id]
                    cycle_paths.add(tuple(cycle_path))
                    continue

                current_path.append(child_id)
                path_nodes.add(child_id)
                visit(child_id, current_path=current_path, path_nodes=path_nodes)
                current_path.pop()
                path_nodes.remove(child_id)

        for entity_id in reversed(cls.sort_topologically(graph)):
            if entity_id not in visited:
                visit(entity_id, current_path=[entity_id], path_nodes={entity_id})

        return list(cycle_paths)

    @classmethod
    def merge_entities(cls, graph: dict[str, dict], entity_ids: Iterable[str]) -> None:
        """Merge the entities into a single entity."""
        entity_ids = set(entity_ids)

        merged_entity_id = min(entity_ids)
        merged_node = graph[merged_entity_id]
        merged_parents = merged_node["parents"]
        merged_children = merged_node["children"]

        for entity_id, node in graph.items():
            if "merged_ids" not in node:
                node["merged_ids"] = {entity_id}

        # merge the parents and children into the selected node
        for entity_id in entity_ids:
            if entity_id == merged_entity_id:
                continue

            node = graph[entity_id]

            # replace the references in the parent and child nodes
            for parent_id in node["parents"]:
                graph[parent_id]["children"].remove(entity_id)
                graph[parent_id]["children"].add(merged_entity_id)

            for child_id in node["children"]:
                graph[child_id]["parents"].remove(entity_id)
                graph[child_id]["parents"].add(merged_entity_id)

            # update the new node with the parents and children
            merged_parents.update(node["parents"])
            merged_children.update(node["children"])
            merged_node["merged_ids"].update(node["merged_ids"])

        for entity_id in entity_ids:
            if entity_id != merged_entity_id:
                del graph[entity_id]

        # remove self references
        merged_parents.discard(merged_entity_id)
        merged_children.discard(merged_entity_id)

    @classmethod
    def merge_cycles(cls, graph: dict[str, dict]) -> list[tuple[str, ...]]:
        """Collect cycles and merge them iteratively until no more cycles are found."""
        merged_cycles = []

        while cycles := cls.collect_cycles(graph, eager=True):
            modified_node_ids = set()
            merged_cycles_count = 0

            for cycle in cycles:
                if any(node_id in modified_node_ids for node_id in cycle):
                    continue

                merged_cycles.append(cycle)
                cls.merge_entities(graph, cycle)
                modified_node_ids.update(cycle)
                cls.assert_no_self_references(graph)
                merged_cycles_count += 1

            logging.info(f"Found {len(cycles):,} cycles, merged {merged_cycles_count:,} of them.")

        logging.info(f"Merged a total {len(merged_cycles):,} cycles.")

        return merged_cycles

    @classmethod
    def remove_grandparent_links(cls, graph: dict[str, dict]) -> list[tuple[str, str]]:
        """Remove the grandparent links from the entities."""
        node_id_to_all_parents_cache = {}
        grandparent_links_removed = []

        def collect_ancestors(entity_id: str) -> set[str]:
            """Collect all ancestor entities of the entity, recursively up to the roots."""
            if entity_id in node_id_to_all_parents_cache:
                return node_id_to_all_parents_cache[entity_id]

            parents = set(graph[entity_id]["parents"])
            for parent_id in list(parents):
                parents.update(collect_ancestors(parent_id))

            node_id_to_all_parents_cache[entity_id] = parents

            return parents

        for entity_id, node in graph.items():
            parent_ancestors = {e for parent_id in node["parents"] for e in collect_ancestors(parent_id)}
            grandparent_links_removed += [(entity_id, p) for p in node["parents"] if p in parent_ancestors]
            node["parents"] -= parent_ancestors

        cls.rebuild_children_links(graph)
        logging.info(f"Removed {len(grandparent_links_removed):,} grandparent links.")

        return grandparent_links_removed

    @classmethod
    def add_subgraph_sizes(cls, graph: dict[str, dict], entities: dict[str, dict] | None = None) -> None:
        """Add the sizes of the sub-graphs to the nodes."""
        descendants_cache = {}
        visited = set()

        def visit(entity_id: str) -> None:
            """Visit the entity and its children."""
            if entity_id in visited:
                return

            visited.add(entity_id)

            node = graph[entity_id]

            for child_id in node["children"]:
                visit(child_id)

            if entity_id in descendants_cache:
                descendants = descendants_cache[entity_id]
            else:
                descendants = {d for child_id in node["children"] for d in descendants_cache[child_id]}
                descendants.add(entity_id)
                descendants_cache[entity_id] = descendants

            node["subgraph_size"] = len(descendants)

            node["subgraph_height"] = (
                max((graph[child_id]["subgraph_height"] for child_id in node["children"]), default=-1) + 1
            )

            node["subgraph_ref_count"] = sum(graph[descendant_id]["ref_count"] for descendant_id in descendants)

        for node in graph.values():
            node["subgraph_size"] = None
            node["subgraph_height"] = None
            node["subgraph_ref_count"] = None

            if entities is not None:
                node["ref_count"] = sum(entities[merged_id]["ref_count"] for merged_id in node["merged_ids"])
            else:
                node["ref_count"] = 0

        for entity_id in cls.get_root_entities(graph):
            visit(entity_id)

    @classmethod
    def write_node_sequences(
        cls,
        sequences: Iterable[Iterable[str]],
        entities: dict[str, dict],
        path: pathlib.Path,
        sort_by_len: bool = False,
    ) -> None:
        """Write the node sequences to a file."""
        sequences_ = [list(sequence) for sequence in sequences]

        def present_node(node_id: str) -> str:
            """Get the name of the entity."""
            return f"{node_id} ({entities[node_id]['name']})"

        sequences_ = sorted(sequences_, key=len, reverse=True) if sort_by_len else sequences_

        with open(path, mode="w", encoding="utf-8", newline="\n") as f:
            for sequence in sequences_:
                line = ""
                prev_node_id = None
                for node_id in sequence:
                    if prev_node_id is not None:
                        if (prev_node_id, node_id) in cls._DEBUG_SAME_AS_EDGES:
                            line += " == "
                        else:
                            line += " -> "

                    line += present_node(node_id)
                    prev_node_id = node_id

                f.write(line + "\n")
