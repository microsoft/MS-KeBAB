# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Extract entity fragments from REBEL dataset.
This can be run on T-REx dataset as well.

- Process the raw documents from REBEL, collect all entities and relationships contained in them.
- Attach a unique source ID to each entity fragment.
- Optionally, verify that the Wikidata property keys exist in the Wikidata properties file.

TODO: Update
Example output:
{"entity_id": "Q25295", "properties": {"names": ["language family"]}, "source_ids": ["81317a6457106bc3ae89d179d65ec7ed", "http://www.wikidata.org/entity/Q33199"], "evidence_map": {}, "entity_types": []}
{"entity_id": "Q11708", "properties": {"names": ["Southeast Asia"]}, "source_ids": ["81317a6457106bc3ae89d179d65ec7ed", "http://www.wikidata.org/entity/Q33199"], "evidence_map": {}, "entity_types": []}
"""

# ruff: noqa: S101

from __future__ import annotations

import hashlib
import json
import logging
import typing
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from kebab.utils.dataset.wikidata.wikidata_utils import WikidataEntity


class RebelFragmentExtractor:
    """Extract entity fragments from REBEL dataset."""

    ENTITY_PREFIXES: typing.ClassVar[set[str]] = {"http://www.wikidata.org/entity/"}
    PROPERTY_PREFIXES: typing.ClassVar[set[str]] = {"http://www.wikidata.org/prop/direct/"}

    DATETIME_TYPE_SUFFIX: typing.ClassVar[str] = "^^http://www.w3.org/2001/XMLSchema#dateTime"
    DECIMAL_TYPE_SUFFIX: typing.ClassVar[str] = "^^http://www.w3.org/2001/XMLSchema#decimal"

    DISALLOWED_NAMES: typing.ClassVar[set[str]] = {"", "it", "he", "she", "they"}

    ENTITY_FRAGMENTS_FILENAME: str = "rebel_entity_fragments.jsonl"

    INCLUDE_TITLE_AS_PROPERTY: bool = False
    INCLUDE_MENTION_AS_PROPERTY: bool = False

    SPLIT_INTO_SUB_DOCUMENTS: bool = False  # Applies only to T-REx dataset (REBEL doesn't have sentence boundaries)

    DEBUG_DUMP_DOC_FILES: bool = False  # whether to dump the original document file records as separate files

    _UNSAFE_FRAGMENT_COUNTER: int = 0

    def __init__(self, *, rebel_dir: Path, output_dir: Path) -> None:
        """
        Initialize the fragment extractor.

        Args:
            rebel_dir: Path to the REBEL data directory (will be *.json-globbed against).
            output_dir: Path to the output directory ("all_entities_fragments.jsonl" will be created there).
        """
        self._logger: logging.Logger = logging.getLogger(__name__)

        self.rebel_dir: Path = rebel_dir

        self.output_dir: Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entity_fragments_output_path = self.output_dir / self.ENTITY_FRAGMENTS_FILENAME
        assert not self.entity_fragments_output_path.exists(), "Output file already exists."

    def run(self) -> None:
        """Run the fragment extraction process."""
        self.extract_entity_fragments()

    def extract_entity_fragments(self, remove_duplicates: bool = True) -> None:
        """Extract all entity fragments from REBEL data and store as a flat JSON-lines file."""
        err_count = 0
        seen = set()
        self._UNSAFE_FRAGMENT_COUNTER = 0

        stats = Counter()
        frag_count = 0

        with open(self.entity_fragments_output_path, mode="w", encoding="utf-8") as f:
            for rec in self._load_rebel_data():
                if not self.SPLIT_INTO_SUB_DOCUMENTS:
                    sub_docs = [rec]
                else:
                    # TODO(pmyshkov): inefficient, should be done in a single pass, but it's nontrivial due to the need
                    # to remove duplicates -- revisit if we decide we need this
                    temp_result = self.extract_entity_fragments_from_document(rec)
                    if not temp_result:
                        continue

                    sub_docs = self.extract_sub_documents_from_document(rec)

                for record in sub_docs:
                    fragments = self.extract_entity_fragments_from_document(
                        record, apply_filter=not self.SPLIT_INTO_SUB_DOCUMENTS, stats=stats
                    )
                    for fragment in fragments.values():
                        if remove_duplicates:
                            str_repr = fragment.get_hashable_repr()
                            if str_repr in seen:
                                continue

                            seen.add(str_repr)

                        try:
                            f.write(fragment.to_json() + "\n")
                            frag_count += 1
                        except ValueError:
                            err_count += 1

                        # dump the original document file as a separate file
                        if self.DEBUG_DUMP_DOC_FILES:
                            debug_predicate = len(fragment.name) > 8 and fragment.entity_id == "Q2351189"  # noqa: PLR2004
                            if debug_predicate:
                                f_name = self.output_dir / f"__{fragment.source_ids[0]}.json"
                                if not f_name.exists():
                                    with open(f_name, mode="w", encoding="utf-8") as doc_f:
                                        json.dump(record, doc_f, ensure_ascii=False, indent=2)

                                return

        if err_count:
            self._logger.error(f"Errors while serializing entities: {err_count}")

        self._logger.info(f"Wrote {frag_count} entity fragments.")

        for key, value in stats.items():
            self._logger.info(f"{key}: {value}")

    @classmethod
    def extract_entity_fragments_from_document(
        cls, record: dict, drop_empty: bool = True, apply_filter: bool = True, stats: Counter | None = None
    ) -> dict[str, WikidataEntity]:
        """Extract entity fragments and properties from a single REBEL document record."""
        if stats is None:
            stats = Counter()

        stats["documents"] += 1

        if apply_filter:
            # skip the disambiguation pages
            if record["text"].endswith("refer to:"):
                stats["disambiguation_pages"] += 1
                return {}

            # skip short text documents
            if len(record["text"]) < 100:  # noqa: PLR2004
                stats["short_text_documents"] += 1
                return {}

        # use text + title md5 hash as a global unique source_id
        text = record["title"] + "\n" + record["text"]
        source_text_hash = hashlib.md5(text.encode()).hexdigest()  # noqa: S324
        doc_id = record["docid"]
        main_entity_uri = record["uri"]

        invalid_entities = 0

        fragments: dict[str, WikidataEntity] = {}
        entity_uris = set()

        # process entity records
        for item in record["entities"]:
            entity_uri = item["uri"]
            entity_id = cls.get_entity_id_from_uri(entity_uri)

            if not entity_id or not entity_id.startswith("Q"):
                invalid_entities += 1
                continue

            entity_uris.add(entity_uri)

            if entity_id not in fragments:
                fragment = WikidataEntity(entity_id=entity_id)
                fragment.metadata["doc_id"] = doc_id
                fragment.metadata["source_text_hash"] = source_text_hash
                fragment.metadata["fragment_id"] = str(cls._UNSAFE_FRAGMENT_COUNTER)

                if cls.INCLUDE_TITLE_AS_PROPERTY:
                    fragment.properties["trex_title"].append(record["title"])

                if cls.INCLUDE_MENTION_AS_PROPERTY:
                    fragment.properties["trex_mention"].append(record["text"])

                cls._UNSAFE_FRAGMENT_COUNTER += 1

                fragments[entity_id] = fragment

            fragments[entity_id].properties["name"].append(item["surfaceform"])

        if apply_filter and main_entity_uri not in entity_uris:
            stats["main_entity_not_found"] += 1
            return {}

        if invalid_entities:
            logging.debug(f"Found {invalid_entities} invalid entities in the document")

        not_found_objects = 0

        # process property records
        for item in record["triples"]:
            property_id = cls.get_property_id_from_uri(item["predicate"]["uri"])
            object_id = cls.get_entity_id_from_uri(item["object"]["uri"])
            subject_id = cls.get_entity_id_from_uri(item["subject"]["uri"])

            if property_id is None or object_id is None or subject_id is None:
                continue

            # only take triplets where the entity we are creating the properties for is the subject of the triplet
            if subject_id in fragments:
                fragments[subject_id].properties[property_id].append(object_id)
            else:
                not_found_objects += 1

        if not_found_objects:
            logging.debug(f"Not found {not_found_objects} object entities in the document")

        # deduplicate property values and remove disallowed names
        for fragment in fragments.values():
            fragment.deduplicate_property_values()

            fragment.properties["name"] = [
                name for name in fragment.properties["name"] if name.lower() not in cls.DISALLOWED_NAMES
            ]

        # keep only fragments with at least one name
        if drop_empty:
            fragments = {k: v for k, v in fragments.items() if v.properties["name"]}

        return fragments

    @classmethod
    def extract_sub_documents_from_document(cls, record: dict) -> Iterable[dict]:
        """
        Extract sub-documents (corresponding to sentences) from a single REBEL document record.
        This applies only to T-REx dataset.
        """
        template = {
            "docid": record["docid"],
            "title": record["title"],
            "uri": record["uri"],
        }

        entity_count = 0
        triple_count = 0

        # TODO(pmyshkov): update boundaries (if we decide to use them, and to this method to begin with)

        for sentence_boundary in record["sentences_boundaries"]:
            sub_document = template.copy()
            sub_document["text"] = record["text"][sentence_boundary[0] : sentence_boundary[1]]
            sub_document["entities"] = []
            sub_document["triples"] = []

            for entity in record["entities"]:
                if sentence_boundary[0] <= entity["boundaries"][0] < entity["boundaries"][1] <= sentence_boundary[1]:
                    sub_document["entities"].append(entity)
                    entity_count += 1

            for triple in record["triples"]:
                predicate = triple["predicate"]
                obj = triple["object"]
                subj = triple["subject"]

                if predicate["boundaries"] and (
                    sentence_boundary[0]
                    <= predicate["boundaries"][0]
                    < predicate["boundaries"][1]
                    <= sentence_boundary[1]
                ):
                    sub_document["triples"].append(triple)
                    triple_count += 1
                    continue

                if obj["boundaries"] and (
                    sentence_boundary[0] <= obj["boundaries"][0] < obj["boundaries"][1] <= sentence_boundary[1]
                ):
                    sub_document["triples"].append(triple)
                    triple_count += 1
                    continue

                if subj["boundaries"] and (
                    sentence_boundary[0] <= subj["boundaries"][0] < subj["boundaries"][1] <= sentence_boundary[1]
                ):
                    sub_document["triples"].append(triple)
                    triple_count += 1
                    continue

            yield sub_document

        entities_original = len(record["entities"])
        triples_original = len(record["triples"])

        if entity_count != entities_original:
            logging.debug(f"Found {entities_original - entity_count} entities not in sub-documents")

        if triple_count < triples_original:
            logging.debug(f"Found {triples_original - triple_count} triples not in sub-documents")

    @classmethod
    def get_entity_id_from_uri(cls, uri: str) -> str | None:
        """Extract entity ID from a URI."""
        if uri.startswith("Q"):
            return uri

        for prefix in cls.ENTITY_PREFIXES:
            if uri.startswith(prefix):
                return uri[len(prefix) :]

        if uri.endswith(cls.DATETIME_TYPE_SUFFIX):
            return uri[: -len(cls.DATETIME_TYPE_SUFFIX)]

        if uri.endswith(cls.DECIMAL_TYPE_SUFFIX):
            return uri[: -len(cls.DECIMAL_TYPE_SUFFIX)]

        return None

    @classmethod
    def get_property_id_from_uri(cls, uri: str) -> str:
        """Extract property ID from a URI."""
        if uri.startswith("P"):
            return uri

        for prefix in cls.PROPERTY_PREFIXES:
            if uri.startswith(prefix):
                return uri[len(prefix) :]

        raise ValueError(f"Unknown property URI prefix: {uri}")

    def _load_rebel_data(self) -> Iterable[dict]:
        """Load REBEL data from multiple files, iteratively."""
        error_count = 0
        err = None

        json_files = list(self.rebel_dir.glob("*.json"))
        self._logger.info(f"Found {len(json_files)} JSON files in the directory.")

        for count, file in enumerate(json_files, start=1):
            with open(file, encoding="utf-8") as f:
                try:
                    yield from json.load(f)
                except json.JSONDecodeError as e:
                    error_count += 1
                    err = e

            self._logger.info(f"Processed {count}/{len(json_files)} ({100 * count / len(json_files):.2f}%) files.")

        jsonl_files = list(self.rebel_dir.glob("*.jsonl"))
        self._logger.info(f"Found {len(jsonl_files)} JSONL files in the directory.")

        for count, file in enumerate(jsonl_files, start=1):
            with open(file, encoding="utf-8") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:  # noqa: PERF203
                        error_count += 1
                        err = e

            self._logger.info(f"Processed {count}/{len(jsonl_files)} ({100 * count / len(jsonl_files):.2f}%) files.")

        if error_count:
            self._logger.error(f"Errors while decoding JSON: {error_count}, last error: {err}")
