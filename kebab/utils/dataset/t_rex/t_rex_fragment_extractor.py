# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Extract entity fragments from T-REx dataset.

- Process the raw documents from T-REx, collect all entities and relationships contained in them.
- Attach a unique source ID to each entity fragment.
- Optionally, verify that the Wikidata property keys exist in the Wikidata properties file.

Example output:
{"entity_id": "Q25295", "properties": {"names": ["language family"]}, "source_ids": ["81317a6457106bc3ae89d179d65ec7ed", "http://www.wikidata.org/entity/Q33199"], "evidence_map": {}, "entity_types": []}
{"entity_id": "Q11708", "properties": {"names": ["Southeast Asia"]}, "source_ids": ["81317a6457106bc3ae89d179d65ec7ed", "http://www.wikidata.org/entity/Q33199"], "evidence_map": {}, "entity_types": []}
"""

from __future__ import annotations

import hashlib
import json
import logging
import typing
from collections.abc import Iterable
from pathlib import Path

from kebab.contracts.entity import Entity


# TODO(pmyshkov): This class is no longer relevant and will be replaced with REBEL extractor.


class TRexFragmentExtractor:
    """Extract entity fragments from T-REx dataset."""

    ENTITY_PREFIXES: typing.ClassVar[set[str]] = {"http://www.wikidata.org/entity/"}
    PROPERTY_PREFIXES: typing.ClassVar[set[str]] = {"http://www.wikidata.org/prop/direct/"}

    DATETIME_TYPE_SUFFIX: typing.ClassVar[str] = "^^http://www.w3.org/2001/XMLSchema#dateTime"

    ENTITY_FRAGMENTS_FILENAME: str = "t_rex_entity_fragments.jsonl"

    DEBUG_DUMP_DOC_FILES: bool = False  # whether to dump the original document file records as separate files

    def __init__(
        self,
        *,
        t_rex_dir: Path,
        output_dir: Path,
    ) -> None:
        """
        Initialize the fragment extractor.

        Args:
            t_rex_dir: Path to the T-REx data directory (will be *.json-globbed against).
            wikidata_properties_path: Path to the Wikidata properties file.
            output_dir: Path to the output directory ("all_entities_fragments.jsonl" will be created there).
        """
        self._logger: logging.Logger = logging.getLogger(__name__)

        self.t_rex_dir: Path = t_rex_dir
        self.output_dir: Path = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.entity_fragments_output_path = self.output_dir / self.ENTITY_FRAGMENTS_FILENAME
        assert not self.entity_fragments_output_path.exists(), "Output file already exists."

    def run(self) -> None:
        """Run the fragment extraction process."""
        # extract entity fragments
        self.extract_entity_fragments()

    def extract_entity_fragments(self, remove_duplicates: bool = True) -> None:
        """Extract all entity fragments from T-REx data and store as a flat JSON-lines file."""
        err_count = 0
        seen = set()

        with open(self.entity_fragments_output_path, mode="w", encoding="utf-8") as f:
            for record in self._load_t_rex_data():
                fragments = self.extract_entity_fragments_from_document(record)
                for fragment in fragments.values():
                    if remove_duplicates:
                        str_repr = fragment.get_hashable_value_repr()
                        if str_repr in seen:
                            continue

                        seen.add(str_repr)

                    try:
                        f.write(fragment.to_json() + "\n")
                    except ValueError:
                        err_count += 1

                    # dump the original document file as a separate file
                    if self.DEBUG_DUMP_DOC_FILES:
                        debug_predicate = len(fragment.names) > 8 and fragment.entity_id == "Q2351189"  # noqa: PLR2004
                        if debug_predicate:
                            f_name = self.output_dir / f"__{fragment.source_ids[0]}.json"
                            if not f_name.exists():
                                with open(f_name, mode="w", encoding="utf-8") as doc_f:
                                    json.dump(record, doc_f, ensure_ascii=False, indent=2)

                            return

        if err_count:
            self._logger.error(f"Errors while serializing entities: {err_count}")

    @classmethod
    def extract_entity_fragments_from_document(cls, record: dict, drop_empty: bool = True) -> dict[str, Entity]:
        """Extract entity fragments and properties from a single T-REx document record."""
        fragments: dict[str, Entity] = {}

        # use text + title md5 hash as a global unique source_id
        text = record["title"] + "\n" + record["text"]
        source_id = hashlib.md5(text.encode()).hexdigest()  # noqa: S324
        doc_id = record["docid"]

        invalid_entities = 0

        # process entity records
        for item in record["entities"]:
            entity_id = cls.get_entity_id_from_uri(item["uri"])

            if not entity_id or not entity_id.startswith("Q"):
                invalid_entities += 1
                continue

            if entity_id not in fragments:
                fragment = Entity(
                    entity_id=entity_id,
                    source_ids=[source_id, doc_id],
                )

                fragments[entity_id] = fragment

            fragments[entity_id].names.append(item["surfaceform"])

        if invalid_entities:
            logging.debug(f"Found {invalid_entities} invalid entities in the document")

        not_found_objects = 0

        # process property records
        for item in record["triples"]:
            property_id = cls.get_property_id_from_uri(item["predicate"]["uri"])
            object_id = cls.get_entity_id_from_uri(item["object"]["uri"])
            subject_id = cls.get_entity_id_from_uri(item["subject"]["uri"])

            # only take triplets where the entity we are creating the properties for is the subject of the triplet
            if subject_id in fragments:
                fragments[subject_id].properties[property_id].append(object_id)
            else:
                not_found_objects += 1

        if not_found_objects:
            logging.debug(f"Not found {not_found_objects} object entities in the document")

        if drop_empty:
            fragments = {k: v for k, v in fragments.items() if v.properties}

        for fragment in fragments.values():
            fragment.deduplicate_property_values()

        return fragments

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

    def _load_t_rex_data(self) -> Iterable[dict]:
        """Load T-REx data from multiple files, iteratively."""
        error_count = 0
        err = None
        files = list(self.t_rex_dir.glob("*.json"))
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
