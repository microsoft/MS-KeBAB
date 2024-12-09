# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Iterable, List, Tuple, TypeVar

from kebab.contracts.document import Document, DocumentUtilities
from kebab.contracts.entity import Entity


class CustomEncoder(json.JSONEncoder):
    """JSON encoder for entities and documents."""

    def default(self, o: Any) -> Any:  # noqa: ANN401, D102
        if isinstance(o, Entity):
            return o.to_dict()
        if isinstance(o, Document):
            return o.to_dict()
        return super().default(o)


def save_to_json(data: Any, file_path: Path) -> None:
    """
    Save data to a JSON file.

    Args:
        data: The serializable data to save.
        file_path: The path where the data should be saved.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# TODO (allenwang): make sure this is imported before the classes that follow
DataItemType = TypeVar("DataItemType")


class ItemReader(ABC, Generic[DataItemType]):
    """
    Abstract base class for data item readers.

    Type Parameters:
    DataItemType: The type of the data items to be read.
    """

    @abstractmethod
    def read_items(self) -> Iterable[DataItemType]:
        """
        Abstract method to read items.

        Returns:
        Iterable[DataItemType]: An iterable of data items of the specified type.
        """


class DocumentJsonlReader(ItemReader[Document]):
    """
    Reader for JSONL files containing `Document` objects.

    This class reads a JSONL file, where each line represents a serialized `Document`.
    It uses schemas to parse and validate the fields of the documents.
    """

    def __init__(self, path: Path):
        self.schemas = DocumentUtilities.load_schemas(Path(__file__).parent / "configs" / "schemas")
        self.path = path

    def read_items(self) -> Iterable[Document]:
        return DocumentUtilities.load_documents(self.path, self.schemas)


class EntityListJsonlReader(ItemReader[List[Entity]]):
    """
    Reader for JSONL files containing lists of `Entity` objects.

    This class reads a JSONL file, where each line contains a serialized list of `Entity` objects.
    """

    def __init__(self, path: Path):
        self.path = path

    def read_items(self) -> Iterable[List[Entity]]:
        with open(self.path, encoding="utf-8") as file:
            for line in file:
                yield [Entity.from_json(entity_json) for entity_json in json.loads(line)]


class EntityPairJsonlReader(ItemReader[Tuple[Entity, Entity]]):
    """
    Reader for JSONL files containing pairs of `Entity` objects.

    This class reads a JSONL file, where each line contains exactly two serialized `Entity` objects.
    """

    def __init__(self, path: Path):
        self.path = path

    def read_items(self) -> Iterable[Tuple[Entity, Entity]]:
        for entities in EntityListJsonlReader(self.path).read_items():
            if len(entities) != 2:
                raise ValueError(f"Expected exactly two entities in line: {entities}")
            yield (entities[0], entities[1])


class BooleanFileReader(ItemReader[bool]):
    """
    Reader for files containing boolean values.

    This class reads a file where each line contains either "0" or "1", representing
    `False` and `True` respectively.
    """

    def __init__(self, path: Path):
        self.path = path

    def read_items(self) -> Iterable[bool]:
        with open(self.path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line == "0":
                    yield False
                elif line == "1":
                    yield True
                else:
                    raise ValueError(f"Invalid line content for BooleanReader: {line}")


class ItemWriter(ABC, Generic[DataItemType]):
    """
    Abstract base class for output item writers.

    Type Parameters:
    DataItemType: The type of the output items to be written.
    """

    @abstractmethod
    def write_items(self, items: Iterable[DataItemType]) -> None:
        """
        Abstract method to write items.

        Args:
        items (Iterable[DataItemType]): An iterable of output items to be written.
        """


class EntityListJsonlWriter(ItemWriter[List[Entity]]):
    """
    Writer for JSONL files containing lists of `Entity` objects.

    This class writes a JSONL file, where each line contains a serialized list of `Entity` objects.
    """

    def __init__(self, path: Path):
        self.path = path

    def write_items(self, items: Iterable[List[Entity]]) -> None:
        with open(self.path, "w", encoding="utf-8") as file:
            for entity_list in items:
                json_line = json.dumps(entity_list, ensure_ascii=False)
                file.write(json_line + "\n")


class BooleanFileWriter(ItemWriter[bool]):
    """
    Writer for files containing boolean values.

    This class writes a file where each line contains "0" or "1", representing
    `False` and `True` respectively.
    """

    def __init__(self, path: Path):
        self.path = path

    def write_items(self, items: Iterable[bool]) -> None:
        with open(self.path, "w", encoding="utf-8") as file:
            for item in items:
                file.write("1\n" if item else "0\n")
