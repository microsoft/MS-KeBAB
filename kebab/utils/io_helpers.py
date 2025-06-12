# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from kebab.contracts.document import Document, DocumentUtilities
from kebab.contracts.entity import DataType, Entity, Property, PropertySchema, ValueType


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


def load_dict_from_json(file_path: Path) -> dict:
    """
    Loads data from a JSON file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        dict: The deserialized data.
    """
    with open(file_path, encoding="utf-8") as file:
        return json.load(file)


class ItemReader[DataItemType](ABC):
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


class StringLineReader(ItemReader[str]):
    """
    Line by line reader for text files.

    This class reads a text file, and outputs a line at a time.
    """

    def __init__(self, path: Path):
        """
        Initializes the line by line reader for text files.

        Args:
            path: The file path to the text file.
        """
        self.path = path

    def read_items(self) -> Iterable[str]:
        """
        Reads the text file and yields lines.

        Returns:
            Iterable[str]: An iterable of lines from the document.
        """
        with open(self.path, encoding="utf-8", newline="\n") as f:
            yield from f


class ItemJsonlReader[DataItemType](ItemReader[DataItemType]):
    """
    Reader for JSONL files containing DataItemType objects.
    This is aimed to be used with simple objects that can be serialized to JSON, such as integers, strings, or booleans.

    This class reads a JSONL file, where each line contains a serialized object.
    """

    def __init__(
        self,
        path: Path,
        converter: Callable[[Any], DataItemType] | None = None,
    ):
        """
        Initializes the JSONL reader for objects.

        Args:
            path: The file path to the JSONL file.
            converter: An optional converter function to convert the deserialized objects to the target type.
        """
        self.path = path
        self.converter = converter

    def read_items(self) -> Iterable[DataItemType]:
        """
        Reads the JSONL file and yields objects.

        Returns:
            Iterable[DataItemType]: An iterable of objects.
        """
        with open(self.path, encoding="utf-8") as file:
            for line in file:
                obj = json.loads(line)

                if self.converter is not None:
                    obj = self.converter(obj)

                yield obj


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
        self.schemas = DocumentUtilities.load_schemas(Path(__file__).parents[1] / "configs" / "doc_schemas")
        self.path = path

    def read_items(self) -> Iterable[Document]:
        """
        Reads the JSONL file and yields `Document` objects.

        Returns:
            Iterable[Document]: An iterable of `Document` objects.
        """
        return DocumentUtilities.load_documents(self.path, self.schemas)


class EntityJsonlReader(ItemReader[Entity]):
    """
    Reader for JSONL files containing `Entity` objects.

    This class reads a JSONL file, where each line contains a serialized `Entity` object.
    """

    def __init__(self, path: Path):
        """
        Initializes the JSONL reader for entities.

        Args:
            path: The file path to the JSONL file.
        """
        self.path = path

    def read_items(self) -> Iterable[Entity]:
        """
        Reads the JSONL file and yields `Entity` objects.

        Returns:
            Iterable[Entity]: An iterable of `Entity` objects.
        """
        with open(self.path, encoding="utf-8") as file:
            for line in file:
                yield Entity.from_dict(json.loads(line))


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
            yield entities[0], entities[1]


class ItemWriter[DataItemType](ABC):
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


class StringLineWriter(ItemWriter[str]):
    """
    Writer for text files.

    This class writes a text file line by line.
    """

    def __init__(self, path: Path):
        """
        Initializes the writer.

        Args:
            path: The file path where the text file will be written.
        """
        self.path = path

    def write_items(self, items: Iterable[str]) -> None:
        """
        Writes lines to a text file.

        Args:
            items: An iterable of lines to be written to the file.
        """
        with open(self.path, "w", encoding="utf-8", newline="\n") as file:
            for line in items:
                file.write(line + "\n")


class ItemJsonlWriter[DataItemType](ItemWriter[DataItemType]):
    """
    Writer for JSONL files containing DataItemType objects.
    This is aimed to be used with simple objects that can be serialized to JSON, such as integers, strings, or booleans.

    This class writes a JSONL file, where each line contains a serialized object.
    """

    def __init__(self, path: Path):
        """
        Initializes the JSONL writer for objects.

        Args:
            path: The file path where the JSONL file will be written.
        """
        self.path = path

    def write_items(self, items: Iterable[DataItemType]) -> None:
        """
        Writes objects to a JSONL file.

        Args:
            items: An iterable of objects to be written to the file.
        """
        with open(self.path, "w", encoding="utf-8", newline="\n") as file:
            for obj in items:
                json_line = json.dumps(obj)
                file.write(json_line + "\n")


class EntityJsonlWriter(ItemWriter[Entity]):
    """
    Writer for JSONL files containing `Entity` objects.

    This class writes a JSONL file, where each line contains a serialized `Entity` object.
    """

    def __init__(self, path: Path):
        """
        Initializes the JSONL writer for entities.

        Args:
            path: The file path where the JSONL file will be written.
        """
        self.path = path

    def write_items(self, items: Iterable[Entity]) -> None:
        """
        Writes `Entity` objects to a JSONL file.

        Args:
            items: An iterable of `Entity` objects to be written to the file.
        """
        with open(self.path, "w", encoding="utf-8", newline="\n") as file:
            for entity in items:
                json_line = json.dumps(
                    entity,
                    ensure_ascii=False,
                    cls=CustomEncoder,
                    separators=(",", ":"),
                )
                file.write(json_line + "\n")


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
        with open(self.path, "w", encoding="utf-8", newline="\n") as file:
            for entity_list in items:
                json_line = json.dumps(
                    entity_list,
                    ensure_ascii=False,
                    cls=CustomEncoder,
                    separators=(",", ":"),
                )
                file.write(json_line + "\n")


def generate_draft_property_schema_from_data(paths: Iterable[Path], output_path: Path) -> None:
    """
    Generates a property schema from data.

    Args:
        paths: paths to files with entities
        output_path: path to save the generated property schema
    """
    properties = {}
    data_types = {}

    def get_value_type(values: list[Any]) -> ValueType:
        if len(values) == 0:
            return ValueType.TEXT
        if isinstance(values[0], bool):
            return ValueType.BOOLEAN
        if isinstance(values[0], int | float):
            return ValueType.NUMERIC
        if isinstance(values[0], str) and ("date" in key or "time" in key):
            return ValueType.DATE
        return ValueType.TEXT

    for path in paths:
        entities = EntityListJsonlReader(path).read_items()
        for entity_list in entities:
            for entity in entity_list:
                for key, values in entity.properties.items():
                    value_type = get_value_type(values)
                    data_type_id = value_type.name.lower()
                    is_collection = len(values) > 1
                    if data_type_id not in data_types:
                        data_types[data_type_id] = DataType.from_dict(
                            {"data_type_id": data_type_id, "value_type": value_type.value, "description": data_type_id}
                        )
                    if key not in properties:
                        properties[key] = Property.from_dict(
                            {
                                "property_id": key,
                                "data_type_id": data_type_id,
                                "description": key,
                                "is_collection": is_collection,
                            },
                            data_types,
                        )
                    elif properties[key].data_type.data_type_id != data_type_id:
                        raise ValueError(
                            f"Data type mismatch for property id '{key}': found '{data_type_id}' in entity '{entity.entity_id}', expected '{properties[key].data_type_id}'."
                        )
                    else:
                        properties[key].is_collection = properties[key].is_collection or is_collection
    property_schema = PropertySchema.from_dict(
        {
            "name": "schema",
            "properties": [prop.to_dict() for prop in properties.values()],
            "data_types": [x.to_dict() for x in data_types.values()],
        }
    )
    property_schema.to_file(output_path)


def compare_files_ignore_linebreaks(file_1_path: Path, file_2_path: Path) -> bool:
    """
    Compares two files on disk while ignoring line breaks.

    Args:
        file_1_path: Path to the first file.
        file_2_path: Path to the second file.

    Returns:
        bool: True if the files are identical (ignoring line breaks), False otherwise.
    """
    with open(file_1_path, encoding="utf-8") as f1, open(file_2_path, encoding="utf-8") as f2:
        lines1 = f1.read().splitlines()
        lines2 = f2.read().splitlines()

    return lines1 == lines2


def _shortcut_target(lnk: Path) -> Path:
    """
    Return the target path stored in a Windows *.lnk file.

    Requires either `pywin32` (win32com) **or** `winshell`.
    Raises RuntimeError if neither is available.
    """
    # try pywin32 (preferred)
    try:
        import win32com.client  # type: ignore

        shell = win32com.client.Dispatch("WScript.Shell")
        target = shell.CreateShortcut(str(lnk)).TargetPath
        return Path(target)
    except ImportError:
        pass

    # fall back to winshell
    try:
        import winshell  # type: ignore

        with winshell.shortcut(str(lnk)) as sc:
            return Path(sc.path)
    except ImportError:
        raise RuntimeError("Resolving *.lnk files requires either 'pywin32' or 'winshell'.") from None


def resolve_path(path: str | Path) -> Path:
    """
    Expand "~" to user home directory and resolve all Windows shortcuts (*.lnk).

    Replace every '*.lnk' segment that occurs in `path` with the shortcut's real target,
    preserving any tail that comes after it.
    """
    p = Path(path)
    cur = Path()  # empty base (relative) or will grow absolute

    for part in p.parts:
        next_part = cur / part if cur != Path() else Path(part)

        # use an absolute path (if needed) to test for existence & read target
        abs_path = next_part if next_part.is_absolute() else Path.cwd() / next_part

        # see if it's actually a shortcut
        if not abs_path.exists() and abs_path.suffix.lower() != ".lnk":
            abs_r_path = abs_path.with_suffix(".lnk")
            if abs_r_path.exists() and abs_r_path.is_file():
                abs_path = abs_r_path
            else:
                raise FileNotFoundError(f"Cannot resolve path: {abs_path}")

        if abs_path.suffix.lower() == ".lnk" and abs_path.is_file():
            cur = _shortcut_target(abs_path)
            continue

        cur = next_part

    return cur
