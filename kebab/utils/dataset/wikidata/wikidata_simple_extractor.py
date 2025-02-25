# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# ruff: noqa: S101
"""
Extract simple entities (ID + names + types) from Wikidata, fetch all properties via API, etc.

Inputs:
- Wikidata dump in JSON format (e.g. from https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz).
- Internet connection to query the Wikidata API.

Example outputs:
- `wikidata_simple_entities.jsonl`, each line like:
{"entity_id": "Q31", "properties": {"P31": ["Q3624078", "Q43702", "Q6256", "Q20181813", "Q1250464"], "name": ["Belgium", "Kingdom of Belgium", "BEL", "be", "\ud83c\udde7\ud83c\uddea", "BE"]}}
{"entity_id": "Q8", "properties": {"P31": ["Q331769", "Q60539479"], "name": ["happiness", "joy", "happy"]}}
"""

from __future__ import annotations

import json
import logging
import pathlib

from kebab.utils.dataset.wikidata import wikidata_utils


class WikidataSimpleExtractor:
    """Get simplified entities (WikidataEntity) and properties from Wikidata."""

    WIKIDATA_SIMPLE_ENTITIES_FILENAME = "wikidata_simple_entities.jsonl"
    WIKIDATA_PROPERTIES_FILENAME = "wikidata_properties.json"

    def __init__(
        self,
        *,
        wikidata_json_dump_path: pathlib.Path | None = None,
        run_entity_extraction: bool,
        run_property_fetch: bool,
        output_dir: pathlib.Path | None = None,
    ):
        """Initialize the extractor."""
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.wikidata_dump_path: pathlib.Path | None = wikidata_json_dump_path
        self.run_entity_extraction: bool = run_entity_extraction
        self.run_property_fetch: bool = run_property_fetch

        self.output_dir: pathlib.Path = output_dir or pathlib.Path.cwd()

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Run the extraction steps."""
        if self.run_entity_extraction:
            self.extract_simple_entities()

        if self.run_property_fetch:
            self.fetch_properties()

    def extract_simple_entities(self, wikidata_dump_path: pathlib.Path | None = None) -> pathlib.Path:
        """Extract simple entities (ID + names + types) from Wikidata."""
        input_path = wikidata_dump_path or self.wikidata_dump_path
        if input_path is None:
            raise ValueError("Wikidata JSON dump path is required.")

        output_path = self.output_dir / self.WIKIDATA_SIMPLE_ENTITIES_FILENAME

        self._logger.info(f"Extracting simple entities from {self.wikidata_dump_path} to {output_path}...")

        entities = wikidata_utils.extract_wikidata_entities_from_dump(
            wikidata_json_dump_path=input_path,
            properties=[wikidata_utils.TypeProperties.INSTANCE_OF.value],
            input_entity_predicate=lambda x: x.get("type") == "item",
        )

        done = 0
        with open(output_path, "w", encoding="utf-8", newline="\n") as output_file:
            for entity in entities:
                output_file.write(entity.to_json(minimal_repr=True) + "\n")

                done += 1
                if done % 100_000 == 0:
                    self._logger.info(f"Saved {done:,} entities")

        return output_path

    def fetch_properties(self) -> pathlib.Path:
        """Fetch all properties from Wikidata."""
        output_path = self.output_dir / self.WIKIDATA_PROPERTIES_FILENAME

        self._logger.info(f"Fetching properties from Wikidata to {output_path}...")

        wikidata_utils.fetch_properties_via_api(output_path=output_path)
        self._logger.info(f"Fetched {len(json.loads(output_path.read_text()))} properties.")

        return output_path
