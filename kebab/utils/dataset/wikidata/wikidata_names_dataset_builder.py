# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Build a dataset of names and aliases of entities of a particular type (and its subtypes) from Wikidata.

Necessary inputs:
- A file with simplified extracted Wikidata entities.
- A file with the type hierarchy for Wikidata.

Example output:
- `wikidata_person_names.jsonl`, each line like:
{"id": "Q23",
 "name": "George Washington",
  "description": "president of the United States from 1789 to 1797",
   "aliases": ["Father of the United States", "The American Fabius", "American Fabius"]}...

The workflow of building the dataset is as follows:
1. Collect all subtypes of the given type.
2. Filter the simple entities to the ones that have the given type and have the required number of aliases.
"""

from __future__ import annotations

import json
import logging
import pathlib
from collections.abc import Iterable

from kebab.utils.dataset.wikidata import wikidata_utils


class WikidataNamesDatasetBuilder:
    """Build a dataset of names and aliases of entities of a particular type (and its subtypes) from Wikidata."""

    WIKIDATA_NAMES_OUTPUT_FILENAME: str = "wikidata_names.jsonl"

    def __init__(
        self,
        *,
        wikidata_entities_path: pathlib.Path,
        wikidata_type_hierarchy_path: pathlib.Path,
        output_dir: pathlib.Path | None = None,
        type_ids: list[str] | None = None,
        min_aliases: int = 2,
    ):
        """Initialize the WikidataNamesDatasetBuilder."""
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.wikidata_entities_path: pathlib.Path = wikidata_entities_path
        self.wikidata_type_hierarchy_path: pathlib.Path = wikidata_type_hierarchy_path
        self.output_dir: pathlib.Path = output_dir or pathlib.Path.cwd()

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.type_ids: list[str] = type_ids or []
        self.min_aliases: int = min_aliases

    def run(self) -> None:
        """Run the extraction process."""
        self._logger.info("Extracting names of entities from Wikidata.")

        graph, type_id_to_node = self.load_hierarchy(self.wikidata_type_hierarchy_path)
        all_subtypes = {
            st for t in self.type_ids for st in wikidata_utils.collect_all_subtypes(graph, type_id_to_node, t)
        }

        self._logger.info(f"Extracting names for {len(all_subtypes):,d} subtypes")

        with open(self.output_dir / self.WIKIDATA_NAMES_OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            for entity in self.filter_entities(
                wikidata_utils.load_wikidata_entities(self.wikidata_entities_path), all_subtypes
            ):
                f.write(entity.to_json() + "\n")

    def load_hierarchy(self, type_hierarchy_path: pathlib.Path | None) -> dict[str, dict]:
        """Load the hierarchy of Wikidata types."""
        self._logger.info("Loading the hierarchy of Wikidata types")
        input_path = type_hierarchy_path or self.wikidata_type_hierarchy_path

        if input_path is None:
            raise ValueError("Path to the hierarchy of Wikidata types is not provided")

        graph = {}
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                node = json.loads(line.strip())
                graph[node["id"]] = node

        self._logger.info(f"Type hierarchy contains {len(graph):,d} nodes")

        return graph

    def filter_entities(
        self,
        entities: Iterable[wikidata_utils.WikidataEntity],
        type_ids: set[str],
        min_aliases: int | None = None,
    ) -> Iterable[wikidata_utils.WikidataEntity]:
        """Filter entities by type IDs."""
        min_aliases = min_aliases or self.min_aliases
        skipped = 0
        count = 0

        for entity in entities:
            types = entity.type
            aliases = entity.aliases
            if len(aliases) < min_aliases:
                skipped += 1
                continue

            if any(t in type_ids for t in types):
                count += 1
                yield entity

        self._logger.info(f"Extracted {count:,d} entities, skipped {skipped:,d} entities")
