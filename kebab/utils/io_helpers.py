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
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


DataItemType = TypeVar('DataItemType')


class ItemReader(ABC, Generic[DataItemType]):
    """Abstract base class for data item readers."""

    @abstractmethod
    def read_items(self) -> Iterable[DataItemType]:
        pass


class DocumentJsonlReader(ItemReader[Document]):
    def __init__(self, path: Path):
        self.schemas = DocumentUtilities.load_schemas(Path(__file__).parent / 'configs' / 'schemas')
        self.path = path

    def read_items(self) -> Iterable[Document]:
        return DocumentUtilities.load_documents(self.path, self.schemas)


class EntityListJsonlReader(ItemReader[List[Entity]]):
    def __init__(self, path: Path):
        self.path = path

    def read_items(self) -> Iterable[List[Entity]]:
        with open(self.path, encoding='utf-8') as file:
            for line in file:
                yield [Entity.from_json(entity_json) for entity_json in json.loads(line)]


class EntityPairJsonlReader(ItemReader[Tuple[Entity, Entity]]):
    def __init__(self, path: Path):
        self.path = path

    def read_items(self) -> Iterable[Tuple[Entity, Entity]]:
        for entities in EntityListJsonlReader(self.path).read_items():
            if len(entities) != 2:
                raise ValueError(f'Expected exactly two entities in line: {entities}')
            yield (entities[0], entities[1])


class BooleanReader(ItemReader[bool]):
    def __init__(self, path: Path):
        self.path = path

    def read_items(self) -> Iterable[bool]:
        with open(self.path, encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line == '0':
                    yield False
                elif line == '1':
                    yield True
                else:
                    raise ValueError(f'Invalid line content for BooleanReader: {line}')
