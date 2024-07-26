# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Extract the text paragraph and corresponding to it entities from Re-DocRED dataset.
The dataset can be found here: https://github.com/tonytan48/Re-DocRED/tree/main.
The dataset contains 500 paragraphs in test set, 500 paragraphs in dev set, and 3052
paragraphs in train set.
"""

from __future__ import annotations

import json
import logging
import typing
from collections.abc import Iterable
from pathlib import Path

from kebab.dataset.wikidata import wikidata_utils


class ReDocRedDatasetBuilder:
    """Extract text paragraphs and entities from Re-DocRED dataset."""

    EXTRACTION_DATASET_FILENAME: str = "re_docred_extraction_dataset.jsonl"
    PUNCTUATION: typing.ClassVar[set[str]] = set(".:!,;?-_(){}'")
    PROPERTIES_TO_DROP: typing.ClassVar[set[str]] = {"pos", "global_pos", "index", "sent_id", "properties"}
    TYPE_MAP: typing.ClassVar[dict[str, str]] = {
        "PER": "Person",
        "PERMISC": "Person and Miscellaneous",
        "PERLOC": "Person and Location",
        "PERORG": "Person and Organization",
        "PERLOCMISC": "Person, Location, and Miscellaneous",
        "TIME": "Time",
        "TIMENUM": "Time and Number",
        "LOC": "Location",
        "LOCMISC": "Location and Miscellaneous",
        "LOCORG": "Location and Organization",
        "LOCPER": "Location and Person",
        "NUM": "Number",
        "NUMMISC": "Number and Miscellaneous",
        "ORG": "Organization",
        "ORGMISC": "Organization and Miscellaneous",
        "ORGLOC": "Organization and Location",
        "ORGPER": "Organization and Person",
        "MISC": "Miscellaneous",
        "MISCORG": "Miscellaneous and Organization",
        "MISCPER": "Miscellaneous and Person",
        "MISCNUM": "Miscellaneous and Number",
        "MISCPERLOC": "Miscellaneous, Person, and Location",
        "MISCLOC": "Miscellaneous and Location",
    }

    def __init__(
        self,
        *,
        re_docred_dir: Path,
        wikidata_properties_path: Path,
        output_dir: Path,
    ) -> None:
        """
        Initialize the dataset extractor.

        Args:
            re_docred_dir: Path to the Re-DocRED data directory.
            wikidata_properties_path: Path to the Wikidata properties file.
            output_dir: Path to the output directory.
        """
        self._logger: logging.Logger = logging.getLogger(__name__)

        self.re_docred_dir: Path = re_docred_dir
        self.wikidata_properties_path: Path = wikidata_properties_path
        self.output_dir: Path = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extraction_dataset_output_path = self.output_dir / self.EXTRACTION_DATASET_FILENAME
        assert not self.extraction_dataset_output_path.exists(), "Output file already exists."

    def run(self) -> None:
        """Run the dataset creation process."""
        entries = self._load_dataset()
        wikidata_properties = wikidata_utils.load_properties(self.wikidata_properties_path)
        extraction_dataset = []
        for entry in entries:
            example = self.extract_example(entry, wikidata_properties)
            extraction_dataset.append(example)

        # save the dataset
        self._write_dataset(extraction_dataset)

    @classmethod
    def get_text(cls, entry: dict) -> str:
        """Extract the text from an entry."""
        text = ""
        for sentence in entry["sents"]:
            for token in sentence:
                if text == "":
                    text += token
                else:
                    if token in cls.PUNCTUATION:
                        text += token
                    else:
                        text += " " + token
        return entry["title"] + "\n\n" + text

    @classmethod
    def merge_entities(cls, entities: list[dict]) -> dict[str, str | list[str]]:
        """Merge the entity names and types into a single entity."""
        merged_entities = {}
        for entity in entities:
            for k, v in entity.items():
                if k not in cls.PROPERTIES_TO_DROP:
                    if k not in merged_entities:
                        merged_entities[k] = set()
                    elif k == "name":
                        # Save only first name as a name and everything else as alternative names
                        if "alternative_names" not in merged_entities:
                            merged_entities["alternative_names"] = set()
                        merged_entities["alternative_names"].add(str(v))
                        continue

                    merged_entities[k].add(str(v))

        for k, v in merged_entities.items():
            if k != "alternative_names" and len(v) == 1:
                merged_entities[k] = v.pop()
            else:
                merged_entities[k] = list(v)

            # map the entity type
            if k == "type":
                entity_type = "".join(merged_entities[k])
                merged_entities[k] = cls.TYPE_MAP.get(entity_type, entity_type)

        return merged_entities

    @classmethod
    def extract_entities(cls, entry: dict) -> list[dict[str, str | list[str]]]:
        """Extract the names of the entities from an entry."""
        return [cls.merge_entities(entry) for entry in entry["vertexSet"]]

    @classmethod
    def extract_properties(
        cls, entry: dict, entities: list[dict[str, str | list[str]]], wikidata_properties: dict
    ) -> list[dict[str, str | list[str]]]:
        """Augment the entities with corresponding properties."""
        for prop in entry["labels"]:
            rel_entity_index, entity_index, property_id = prop["t"], prop["h"], prop["r"]
            property_label = wikidata_properties[property_id]["label"]
            property_label = property_label.replace(" ", "_")
            entities[entity_index][property_label] = entities[rel_entity_index]["name"]

        return entities

    @classmethod
    def extract_example(cls, entry: dict, wikidata_properties: dict) -> dict:
        """Extract an example from the Re-DocRED dataset."""
        entities = cls.extract_entities(entry)
        entities = cls.extract_properties(entry, entities, wikidata_properties)
        text = cls.get_text(entry)
        text_id = str(hash(text))
        return {"text_id": text_id, "text": text, "entities": entities}

    def _load_dataset(self) -> Iterable[dict]:
        """Load Re-DocRED data from multiple files, iteratively."""
        error_count = 0
        err = None
        files = list(self.re_docred_dir.glob("*.json"))
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

    def _write_dataset(self, extraction_dataset: list[dict]) -> None:
        """Write the extraction dataset to disk."""
        error_count = 0
        err = None
        with open(self.extraction_dataset_output_path, "w", encoding="utf-8") as f:
            for entry in extraction_dataset:
                try:
                    json.dump(entry, f)
                    f.write("\n")
                except TypeError as e:  # noqa: PERF203
                    error_count += 1
                    err = e

        if error_count:
            self._logger.error(f"Errors while saving the dataset: {error_count}, last error: {err}")
