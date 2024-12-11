# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Iterable, TypeVar

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


def save_dict_to_json(data: dict, file_path: Path) -> None:
    """
    Saves data to a JSON file.

    Args:
        data: The serializable dict data to save.
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
        """
        Initializes the JSONL reader for documents.

        Args:
            path: The file path to the JSONL file.
        """
        self.schemas = DocumentUtilities.load_schemas(Path(__file__).parents[1] / "configs" / "schemas")
        self.path = path

    def read_items(self) -> Iterable[Document]:
        """
        Reads the JSONL file and yields `Document` objects.

        Returns:
            Iterable[Document]: An iterable of `Document` objects.
        """
        return DocumentUtilities.load_documents(self.path, self.schemas)


class EntityListJsonlReader(ItemReader[list[Entity]]):
    """
    Reader for JSONL files containing lists of `Entity` objects.

    This class reads a JSONL file, where each line contains a serialized list of `Entity` objects.
    """

    def __init__(self, path: Path):
        """
        Initializes the JSONL reader for entity lists.

        Args:
            path: The file path to the JSONL file.
        """
        self.path = path

    def read_items(self) -> Iterable[list[Entity]]:
        """
        Reads the JSONL file and yields lists of `Entity` objects.

        Returns:
            Iterable[list[Entity]]: An iterable of lists of `Entity` objects.
        """
        with open(self.path, encoding="utf-8") as file:
            for line in file:
                yield [Entity.from_dict(entity_dict) for entity_dict in json.loads(line)]


class EntityPairJsonlReader(ItemReader[tuple[Entity, Entity]]):
    """
    Reader for JSONL files containing pairs of `Entity` objects.

    This class reads a JSONL file, where each line contains exactly two serialized `Entity` objects.
    """

    def __init__(self, path: Path):
        """
        Initializes the JSONL reader for entity pairs.

        Args:
            path: The file path to the JSONL file.
        """
        self.path = path

    def read_items(self) -> Iterable[tuple[Entity, Entity]]:
        """
        Reads the JSONL file and yields pairs of `Entity` objects.

        Returns:
            Iterable[tuple[Entity, Entity]]: An iterable of pairs of `Entity` objects.
        """
        for entities in EntityListJsonlReader(self.path).read_items():
            if len(entities) != 2:  # noqa: PLR2004 magic-value-comparison
                raise ValueError(f"Expected exactly two entities in line: {entities}")
            yield (entities[0], entities[1])


class BooleanFileReader(ItemReader[bool]):
    """
    Reader for files containing boolean values.

    This class reads a file where each line contains either "0" or "1", representing
    `False` and `True` respectively.
    """

    def __init__(self, path: Path):
        """
        Initializes the reader for files containing boolean values.

        Args:
            path: The file path.
        """
        self.path = path

    def read_items(self) -> Iterable[bool]:
        """
        Reads boolean values from the file.

        Returns:
            Iterable[bool]: An iterator over boolean values.
        """
        with open(self.path, encoding="utf-8") as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line == "0":
                    yield False
                elif stripped_line == "1":
                    yield True
                else:
                    raise ValueError(f"Invalid line content for {BooleanFileReader.__name__}: {stripped_line}")


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


class EntityListJsonlWriter(ItemWriter[list[Entity]]):
    """
    Writer for JSONL files containing lists of `Entity` objects.

    This class writes a JSONL file, where each line contains a serialized list of `Entity` objects.
    """

    def __init__(self, path: Path):
        """
        Initializes the JSONL writer for entity lists.

        Args:
            path: The file path where the JSONL file will be written.
        """
        self.path = path

    def write_items(self, items: Iterable[list[Entity]]) -> None:
        """
        Writes lists of `Entity` objects to a JSONL file.

        Args:
            items: An iterable of lists of `Entity` objects to be written to the file.
        """
        with open(self.path, "w", encoding="utf-8") as file:
            for entity_list in items:
                json_line = json.dumps(
                    entity_list,
                    ensure_ascii=False,
                    cls=CustomEncoder,
                    separators=(",", ":"),
                )
                file.write(json_line + "\n")


class BooleanFileWriter(ItemWriter[bool]):
    """
    Writer for files containing boolean values.

    This class writes a file where each line contains "0" or "1", representing
    `False` and `True` respectively.
    """

    def __init__(self, path: Path):
        """
        Initializes the writer for boolean values.

        Args:
            path: The file path where the boolean values will be written.
        """
        self.path = path

    def write_items(self, items: Iterable[bool]) -> None:
        """
        Writes boolean values to a file.

        Args:
            items: An iterable of boolean values to be written to the file.
        """
        with open(self.path, "w", encoding="utf-8") as file:
            for item in items:
                file.write("1\n" if item else "0\n")
