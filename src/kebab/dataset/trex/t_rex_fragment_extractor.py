# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Extract entity fragments from T-REx dataset.

- Process the raw documents from T-REx, collect all entities and relationships contained in them.
- Attach a unique source ID to each entity fragment.
- Optionally, verify that the Wikidata property keys exist in the Wikidata properties file.

Example output:
{"entity_id":"Q902","names":["Bangladesh"],"properties":{"P47":["Q668"]},"source_id":"81317a6457106bc3ae89d179d65ec7ed"}
{"entity_id":"Q837","names":["Nepal"],"properties":{"P47":["Q148","Q668"]},"source_id":"81317a6457106bc3ae89d179d65ec7ed"}
{"entity_id":"Q148","names":["China"],"properties":{"P47":["Q837","Q668"]},"source_id":"81317a6457106bc3ae89d179d65ec7ed"}
"""

# ruff: noqa: S101

from __future__ import annotations

import hashlib
import json
import logging
import typing
from collections.abc import Iterable
from pathlib import Path

from kebab.dataset import wikidata_utils
from kebab.dataset.trex.entity_fragment import EntityFragment


class TRexFragmentExtractor:
    """Extract entity fragments from T-REx dataset."""

    ENTITY_PREFIXES: typing.ClassVar[set[str]] = {"http://www.wikidata.org/entity/"}
    PROPERTY_PREFIXES: typing.ClassVar[set[str]] = {"http://www.wikidata.org/prop/direct/"}

    DATETIME_TYPE_SUFFIX: typing.ClassVar[str] = "^^http://www.w3.org/2001/XMLSchema#dateTime"

    def __init__(
        self,
        *,
        trex_dir: Path,
        path_to_wikidata_properties: Path,
        output_dir: Path,
    ) -> None:
        """
        Initialize the fragment extractor.

        Args:
            trex_dir: Path to the T-REx data directory (will be *.json-globbed against).
            path_to_wikidata_properties: Path to the Wikidata properties file.
            output_dir: Path to the output directory ("all_entities_fragments.jsonl" will be created there).
        """
        self._logger: logging.Logger = logging.getLogger(__name__)

        self.data_dir: Path = trex_dir
        self.path_to_wikidata_properties: Path = path_to_wikidata_properties
        self.output_dir: Path = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_ent_fragments_output_path = self.output_dir / "all_entities_fragments.jsonl"
        assert not self.all_ent_fragments_output_path.exists(), "Output file already exists."

        self.wikidata_properties: dict[str, dict] = {}

    def run(self) -> None:
        """Run the fragment extraction process."""
        # extract entity fragments
        self.extract_entity_fragments()

        # validate entity fragments
        self.wikidata_properties = wikidata_utils.load_properties(self.path_to_wikidata_properties)
        self.validate_entity_fragments()

    def extract_entity_fragments(self) -> None:
        """Extract all entity fragments from T-REx data and store as a flat JSON-lines file."""
        err_count = 0

        with open(self.all_ent_fragments_output_path, mode="w", encoding="utf-8") as f:
            for record in self._load_trex_data():
                entities = self.extract_entity_fragments_from_document(record)
                for entity in entities.values():
                    try:
                        f.write(entity.model_dump_json() + "\n")
                    except ValueError:
                        err_count += 1

        if err_count:
            self._logger.error(f"Errors while serializing entities: {err_count}")

    def validate_entity_fragments(self) -> None:
        """Validate entity fragments, e.g. whether the properties used are known to the system."""
        if not self.wikidata_properties:
            self._logger.warning("Wikidata properties are not loaded - skipping validation.")
            return

        unknown_properties = set()
        with open(self.all_ent_fragments_output_path, encoding="utf-8") as f:
            for line in f:
                entity = EntityFragment.model_validate_json(line)
                for prop_id in entity.properties:
                    if prop_id not in self.wikidata_properties:
                        unknown_properties.add(prop_id)

        for prop_id in sorted(unknown_properties):
            self._logger.warning(f"Unknown property: {prop_id}")

    @classmethod
    def extract_entity_fragments_from_document(cls, record: dict, drop_empty: bool = True) -> dict[str, EntityFragment]:
        """Extract entities and properties from a single T-REx document record."""
        entities: dict[str, EntityFragment] = {}

        # use text + title md5 hash as a global unique source_id
        text = record["title"] + "\n" + record["text"]
        source_id = hashlib.md5(text.encode()).hexdigest()  # noqa: S324

        invalid_entities = 0

        # process entity records
        for item in record["entities"]:
            entity_id = cls.get_entity_id_from_uri(item["uri"])

            if not entity_id:
                invalid_entities += 1
                continue

            if entity_id in entities:
                entities[entity_id].names.add(item["surfaceform"])
                continue

            entity = EntityFragment(
                entity_id=entity_id,
                names={item["surfaceform"]},
                source_id=source_id,
            )

            entities[entity.entity_id] = entity

        if invalid_entities:
            logging.debug(f"Found {invalid_entities} invalid entities in the document")

        not_found_objects = 0

        # process property records
        for item in record["triples"]:
            property_id = cls.get_property_id_from_uri(item["predicate"]["uri"])
            object_id = cls.get_entity_id_from_uri(item["object"]["uri"])
            subject_id = cls.get_entity_id_from_uri(item["subject"]["uri"])

            # only take triplets where the entity we are creating the properties for is the subject of the triplet
            if subject_id in entities:
                entities[subject_id].properties[property_id].add(object_id)
            else:
                not_found_objects += 1

        if not_found_objects:
            logging.debug(f"Not found {not_found_objects} object entities in the document")

        if drop_empty:
            entities = {k: v for k, v in entities.items() if v.properties}

        return entities

    @classmethod
    def get_entity_id_from_uri(cls, uri: str) -> str | None:
        """Extract entity ID from a URI."""
        for prefix in cls.ENTITY_PREFIXES:
            if uri.startswith(prefix):
                return uri[len(prefix) :]

        if uri.endswith(cls.DATETIME_TYPE_SUFFIX):
            return uri[: -len(cls.DATETIME_TYPE_SUFFIX)]

        return None

    @classmethod
    def get_property_id_from_uri(cls, uri: str) -> str:
        """Extract property ID from a URI."""
        for prefix in cls.PROPERTY_PREFIXES:
            if uri.startswith(prefix):
                return uri[len(prefix) :]

        raise ValueError(f"Unknown property URI prefix: {uri}")

    def _load_trex_data(self) -> Iterable[dict]:
        """Load T-REx data from multiple files, iteratively."""
        error_count = 0
        err = None
        files = list(self.data_dir.glob("*.json"))
        self._logger.info(f"Found {len(files)} files in the directory.")

        for count, file in enumerate(files, start=1):
            with open(file, encoding="utf-8") as f:
                try:
                    yield from json.load(f)
                except json.JSONDecodeError as e:
                    error_count += 1
                    err = e

            self._logger.info(f"Processed {count}/{len(files)} ({100 * count / len(files):.2f}%) files.")

        if error_count:
            self._logger.error(f"Errors while decoding JSON: {error_count}, last error: {err}")
