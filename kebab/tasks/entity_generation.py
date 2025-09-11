# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections.abc import Iterable
from logging import Logger
from pathlib import Path
from typing import Any

from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.utils.io_helpers import EntityJsonlReader, resolve_path


class EntityGenerationTask(Task):
    """Represents an entity generation benchmark task with its data files."""

    __entities: Path

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.EntityGeneration

    @property
    def entities(self) -> Path:
        """Return path to entities."""
        return self.__entities

    def __init__(
        self,
        name: str,
        entities: Path,
        schema: Path | None = None,
        data: Path | None = None,
    ):
        """Initialize an entity generation task."""
        super().__init__(name, schema, data=data)
        self.__entities = resolve_path(entities, data)

    def read_items(self) -> Iterable[Entity]:
        """
        Read data items with entities.

        Returns:
            Iterable[Entity]: An iterable of entities.
        """
        entities = EntityJsonlReader(self.entities).read_items()
        return entities

    def write_items(self, path: Path, items: Iterable[Any]) -> None:
        """
        Write items to the specified path.

        Args:
            path: The file path where the items should be written.
            items: An iterable of items to be written to the file.
        """
        raise NotImplementedError("write_items is not implemented for EntityGenerationTask")

    def evaluate(
        self,
        output_to_evaluate: Path,
        eval_result_path: Path | None = None,
        logger: Logger | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the entity generation task."""
        raise NotImplementedError("evaluate is not implemented for EntityGenerationTask")
