# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Extract the text paragraph and corresponding to it entities from Re-DocRED dataset.
The dataset can be found here: https://github.com/tonytan48/Re-DocRED/tree/main.
The dataset contains 500 paragraphs in test set, 500 paragraphs in dev set, and 3052
paragraphs in train set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import typing
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from kebab.contracts.document import Document, DocumentSchema, DocumentUtilities
from kebab.contracts.entity import Entity, PropertySchema, Property, ValueType, DataType
from kebab.utils import io_helpers
from kebab.utils.dataset.wikidata import wikidata_utils
from kebab.utils.io_helpers import get_value_type


class ReDocRedDatasetBuilder:
    """Extract text paragraphs and entities from Re-DocRED dataset."""

    EXTRACTS_FILENAME: str = "extracts.jsonl"
    ENTITIES_FILENAME: str = "entities.jsonl"
    PROPERY_SCHEMA_FILENAME: str = "property_schema.json"
    PUNCTUATION: typing.ClassVar[set[str]] = set(".:!,;?-_)}]'#%@")
    PUNCTUATION_WO_SPACE: typing.ClassVar[set[str]] = set("-_({['/@$")
    PROPERTIES_TO_DROP: typing.ClassVar[set[str]] = {"pos", "global_pos", "index", "sent_id", "properties"}
    TYPES_TO_DROP: typing.ClassVar[set[str]] = {"number"}
    TYPE_MAP: typing.ClassVar[dict[str, str]] = {
        "PER": "person",
        "PERMISC": "person and miscellaneous",
        "PERLOC": "person and location",
        "PERORG": "person and organization",
        "PERLOCMISC": "person, location, and miscellaneous",
        "TIME": "time",
        "TIMENUM": "time and number",
        "LOC": "location",
        "LOCMISC": "location and miscellaneous",
        "LOCORG": "location and organization",
        "LOCPER": "location and person",
        "NUM": "number",
        "NUMMISC": "number and miscellaneous",
        "ORG": "organization",
        "ORGMISC": "organization and miscellaneous",
        "ORGLOC": "organization and location",
        "ORGPER": "organization and person",
        "MISC": "miscellaneous",
        "MISCORG": "miscellaneous and organization",
        "MISCPER": "miscellaneous and person",
        "MISCNUM": "miscellaneous and number",
        "MISCPERLOC": "miscellaneous, person, and location",
        "MISCLOC": "miscellaneous and location",
    }

    MIN_PROPERTY_COUNT: typing.ClassVar[int] = 2

    SCHEMAS: typing.ClassVar[dict[str, DocumentSchema]] = DocumentUtilities.load_schemas(
        Path(__file__).parents[3] / "configs" / "doc_schemas"
    )

    _logger: logging.Logger
    re_docred_dir: Path
    wikidata_properties_path: Path
    output_dir: Path
    extracts_output_path: Path
    entities_output_path: Path

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
        self._logger = logging.getLogger(__name__)
        self.re_docred_dir = re_docred_dir
        self.wikidata_properties_path = wikidata_properties_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extracts_output_path = self.output_dir / self.EXTRACTS_FILENAME
        self.entities_output_path = self.output_dir / self.ENTITIES_FILENAME
        self.property_schema_path = self.output_dir / self.PROPERY_SCHEMA_FILENAME
        assert not self.extracts_output_path.exists(), "Extracts output file already exists."
        assert not self.entities_output_path.exists(), "Entities output file already exists."

    def run(self) -> None:
        """Run the dataset creation process."""
        entries = self._load_dataset()
        wikidata_properties = wikidata_utils.load_properties(self.wikidata_properties_path)
        extraction_dataset = []
        property_to_data_type = {
            "name": DataType.from_dict(
                {"data_type_id": "text", "value_type": ValueType.TEXT.value, "description": "text"}
            ),
            "type": DataType.from_dict(
                {"data_type_id": "text", "value_type": ValueType.TEXT.value, "description": "text"}
            ),
        }
        for entry in entries:
            example = self.extract_example(entry, wikidata_properties)
            for entity in example["entities"]:
                for prop in entity.properties:
                    if prop not in property_to_data_type:
                        property_to_data_type[prop] = DataType.from_dict(
                            {
                                "data_type_id": "reference",
                                "value_type": ValueType.REFERENCE.value,
                                "description": "reference to another entity",
                            }
                        )
            extraction_dataset.append(example)
        property_schema = self.build_property_schema(
            property_to_data_type,
            [entity for example in extraction_dataset for entity in example["entities"]])

        # save the dataset
        self._write_dataset(extraction_dataset)
        property_schema.to_file(self.property_schema_path)

    @classmethod
    def get_document(cls, entry: dict) -> Document:
        """Extract the text from an entry."""
        text = ""
        for sentence in entry["sents"]:
            for token in sentence:
                try:
                    decoded_token = token.encode("unicode-escape").decode("unicode-escape")
                except UnicodeDecodeError:
                    decoded_token = token
                if text == "":
                    text += decoded_token
                else:
                    if decoded_token in cls.PUNCTUATION or decoded_token[0] in cls.PUNCTUATION:
                        text += decoded_token
                    else:
                        if text[-1] not in cls.PUNCTUATION_WO_SPACE:
                            text += " "
                        text += decoded_token

        text_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        data = {"title": entry["title"], "text": text}
        schema = cls.SCHEMAS["plain_text"]
        return Document(text_id, schema, data)

    def merge_entities(self, entities: list[dict]) -> dict[str, set[str]]:
        """Merge the entity names and types into a single entity."""
        merged_entity = {}
        for entity in entities:
            for k, v in entity.items():
                if k not in self.PROPERTIES_TO_DROP:
                    if k not in merged_entity:
                        merged_entity[k] = set()

                    merged_entity[k].add(str(v))

        merged_entity["type"] = {self.TYPE_MAP.get(t, t) for t in merged_entity["type"]}

        return merged_entity

    def extract_entities(self, entry: dict) -> list[dict[str, set[str]]]:
        """Extract the names of the entities from an entry."""
        return [self.merge_entities(value) for value in entry["vertexSet"]]

    def extract_properties(
        self, entry: dict, entities: list[dict[str, set[str]]], wikidata_properties: dict
    ) -> list[dict[str, set[str]]]:
        """Augment the entities with corresponding properties."""
        for prop in entry["labels"]:
            rel_entity_index, entity_index, property_id = prop["t"], prop["h"], prop["r"]
            property_label = wikidata_properties[property_id]["label"]
            if property_label not in entities[entity_index]:
                entities[entity_index][property_label] = set()
            entities[entity_index][property_label].add(str(rel_entity_index))
        return entities
    
    def build_property_schema(self, property_to_data_type: dict[str, DataType], entities: list[Entity]) -> PropertySchema:
        """Build the property schema from the property to value types mapping."""
        property_schema = PropertySchema()
        properties = {}
        data_types = {dt.data_type_id: dt for dt in property_to_data_type.values()}
        is_collection = defaultdict(bool)
        for entity in entities:
            for prop, values in entity.properties.items():
                if len(values) > 1:
                    is_collection[prop] = True
        for prop, data_type in property_to_data_type.items():
            properties[prop] = Property.from_dict(
                {
                    "property_id": prop,
                    "data_type_id": data_type.data_type_id,
                    "description": prop,
                    "is_collection": is_collection[prop],
                },
                data_types,
            )
        property_schema = PropertySchema.from_dict(
        {
            "name": "schema",
            "properties": [prop.to_dict() for prop in properties.values()],
            "data_types": [x.to_dict() for x in data_types.values()],
        })
        return property_schema

    def filter_entities(self, entities: list[Entity]) -> list[Entity]:
        """Filter out entities based on certain criteria."""
        filtered_entities = []
        dropped_entity_ids = set()
        for entity in entities:
            if any((entity_type in self.TYPES_TO_DROP) for entity_type in entity.properties["type"]):
                self._logger.debug(f"filtering {entity.to_json()}: has type in types_to_drop.")
                dropped_entity_ids.add(entity.entity_id)
                continue
            filtered_entities.append(entity)
        # update properties to remove references to dropped entities
        for entity in filtered_entities:
            properties_to_drop = set()
            for prop, values in entity.properties.items():
                if prop in ("name", "type"):
                    continue
                updated_values = set()
                for value in values:
                    if value not in dropped_entity_ids:
                        updated_values.add(value)
                if not updated_values:
                    self._logger.debug(
                        f"filtering all values of the property {prop} in entity {entity.to_json()}."
                    )
                    properties_to_drop.add(prop)
                entity.properties[prop] = sorted(updated_values)
            for prop_name in properties_to_drop:
                del entity.properties[prop_name]

        # # update entity IDs to be consecutive
        # id_map = {entity.entity_id: str(idx) for idx, entity in enumerate(filtered_entities)}
        # for entity in filtered_entities:
        #     entity.entity_id = id_map[entity.entity_id]
        # # update references in properties
        # for entity in filtered_entities:
        #     for prop, values in entity.properties.items():
        #         if property_to_data_type[prop].value_type != ValueType.REFERENCE.value:
        #             continue
        #         updated_values = {id_map[v] for v in values if v in id_map}
        #         entity.properties[prop] = sorted(updated_values)
        return filtered_entities

    def extract_example(self, entry: dict, wikidata_properties: dict) -> dict:
        """Extract an example from the Re-DocRED dataset."""
        entities = self.extract_entities(entry)
        entities = self.extract_properties(entry, entities, wikidata_properties)
        entities = [Entity(str(i), {k: sorted(v) for k, v in entity.items()}) for i, entity in enumerate(entities)]
        entities = self.filter_entities(entities)
        document = self.get_document(entry)
        return {"document": document, "entities": entities}

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
        with (
            open(self.entities_output_path, "w", encoding="utf-8") as entities_file,
            open(self.extracts_output_path, "w", encoding="utf-8") as extracts_file,
        ):
            for entry in extraction_dataset:
                current_pos_extracts = extracts_file.tell()
                current_pos_entities = entities_file.tell()
                try:
                    json.dump(entry["document"], extracts_file, cls=io_helpers.CustomEncoder, ensure_ascii=False)
                    extracts_file.write("\n")
                    json.dump(entry["entities"], entities_file, cls=io_helpers.CustomEncoder, ensure_ascii=False)
                    entities_file.write("\n")
                except TypeError as e:
                    error_count += 1
                    err = e
                    extracts_file.seek(current_pos_extracts)
                    entities_file.seek(current_pos_entities)

        if error_count:
            self._logger.error(f"Errors while saving the dataset: {error_count}, last error: {err}")
