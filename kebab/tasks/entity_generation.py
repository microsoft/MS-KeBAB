# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskInstance
from kebab.utils.io_helpers import EntityJsonlReader


class EntityGenerationTaskInstance(TaskInstance):
    """Represents an entity generation benchmark task instance with its data files."""

    __data_entities: Path

    @property
    def data_entities(self) -> Path:
        """Return path to entities."""
        return self.__data_entities

    def __init__(
        self,
        name: str,
        task: Task,
        entities: str,
    ):
        """Initialize an entity generation task instance."""
        super().__init__(name, task)
        self.__data_entities = Path(entities)

    def read_items(self) -> Iterable[Entity]:
        """
        Read data items with entities.

        Returns:
            Iterable[Entity]: An iterable of entities.
        """
        entities = EntityJsonlReader(self.data_entities).read_items()
        return entities

    def write_items(self, path: Path, items: Iterable[Any]) -> None:
        """
        Write items to the specified path.

        Args:
            path: The file path where the items should be written.
            items: An iterable of items to be written to the file.
        """
        raise NotImplementedError("write_items is not implemented for EntityGenerationTaskInstance")

    def evaluate(
        self,
        output_to_evaluate: Path,  # noqa: ARG002
        eval_result_path: Path | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the entity generation task instance."""
        raise NotImplementedError("evaluate is not implemented for EntityGenerationTaskInstance")
