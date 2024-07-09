# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Extract the names of entities of a particular type (and its subtypes) from Wikidata.

Necessary inputs:
- A file with simplified "concrete" Wikidata entities.
- A file with the type hierarchy for Wikidata.

Example output:
- `wikidata_person_names.jsonl`, each line like:
{"id": "Q23",
 "name": "George Washington",
  "description": "president of the United States from 1789 to 1797",
   "aliases": ["Father of the United States", "The American Fabius", "American Fabius"]}...

The workflow of extracting the hierarchy is as follows:
1. Collect all subtypes of the given type.
2. Filter the concrete entities to the ones that have the given type and have the required number of aliases.
"""

from __future__ import annotations

import json
import logging
import pathlib
from collections.abc import Iterable

from kebab.dataset.wikidata.wikidata_hierarchy_extractor import TypeProperties


class WikidataNamesExtractor:
    """Extract names of entities of a particular types from Wikidata."""

    WIKIDATA_NAMES_OUTPUT_FILENAME: str = "wikidata_names.jsonl"

    def __init__(
        self,
        *,
        path_to_wikidata_concrete_entities: pathlib.Path | None = None,
        path_to_wikidata_type_hierarchy: pathlib.Path | None = None,
        output_dir: pathlib.Path | None = None,
        type_ids: list[str] | None = None,
        min_aliases: int = 2,
    ):
        """Initialize the WikidataNamesExtractor."""
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.path_to_wikidata_concrete_entities: pathlib.Path | None = path_to_wikidata_concrete_entities
        self.path_to_wikidata_type_hierarchy: pathlib.Path | None = path_to_wikidata_type_hierarchy
        self.output_dir: pathlib.Path = output_dir or pathlib.Path.cwd()

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.type_ids: list[str] = type_ids or []
        self.min_aliases: int = min_aliases

    def run(self) -> None:
        """Run the extraction process."""
        self._logger.info("Extracting names of entities from Wikidata.")

        graph = self.load_hierarchy(self.path_to_wikidata_type_hierarchy)
        all_subtypes = {st for t in self.type_ids for st in self.collect_all_subtypes(t, graph)}

        self._logger.info(f"Extracting names for {len(all_subtypes):,d} subtypes")

        with open(self.output_dir / self.WIKIDATA_NAMES_OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            for entity in self.filter_entities(
                self.load_concrete_entities(self.path_to_wikidata_concrete_entities), all_subtypes
            ):
                # don't need any properties, only the aliases
                del entity["properties"]
                f.write(json.dumps(entity, ensure_ascii=False) + "\n")

    def load_hierarchy(self, path_to_hierarchy: pathlib.Path | None) -> dict[str, dict]:
        """Load the hierarchy of Wikidata types."""
        self._logger.info("Loading the hierarchy of Wikidata types.")
        input_path = path_to_hierarchy or self.path_to_wikidata_type_hierarchy

        if input_path is None:
            raise ValueError("Path to the hierarchy of Wikidata types is not provided.")

        graph = {}
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                node = json.loads(line.strip())
                graph[node["id"]] = node

        self._logger.info(f"Type hierarchy contains {len(graph):,d} nodes")

        return graph

    def load_concrete_entities(self, path_to_entities: pathlib.Path | None) -> Iterable[dict]:
        """Load the list of concrete entities from Wikidata."""
        input_path = path_to_entities or self.path_to_wikidata_concrete_entities

        if input_path is None:
            raise ValueError("Path to the list of concrete entities is not provided.")

        count = 0
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                entity = json.loads(line.strip())
                count += 1
                yield entity

        self._logger.info(f"Read {count:,d} entities")

    def filter_entities(
        self,
        entities: Iterable[dict],
        type_ids: set[str],
        min_aliases: int | None = None,
    ) -> Iterable[dict]:
        """Filter entities by type IDs."""
        min_aliases = min_aliases or self.min_aliases
        skipped = 0
        count = 0

        for entity in entities:
            types = entity["properties"].get(TypeProperties.INSTANCE_OF.value, [])

            if not types:
                skipped += 1
                continue

            aliases = entity.get("aliases", {})
            if len(aliases) < min_aliases:
                skipped += 1
                continue

            if any(t in type_ids for t in types):
                count += 1
                yield entity

        self._logger.info(f"Extracted {count:,d} entities, skipped {skipped:,d} entities")

    @classmethod
    def collect_all_subtypes(cls, type_id: str, graph: dict[str, dict]) -> set[str]:
        """Collect all subtypes of the given type."""
        subtypes = set()
        stack = [type_id]
        while stack:
            current_type_id = stack.pop()

            if current_type_id not in graph:
                logging.warning(f"Type {current_type_id} not found in the graph")
                continue

            for merged_id in graph[current_type_id]["merged_ids"]:
                subtypes.add(merged_id)

            logging.info(f"Added {current_type_id} ({graph[current_type_id]['name']}) to the subtypes")

            for child_id in graph[current_type_id]["children"]:
                stack.append(child_id)  # noqa: PERF402

        return subtypes
