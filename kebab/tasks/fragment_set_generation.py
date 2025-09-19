# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.utils.io_helpers import EntityListJsonlReader, resolve_path


class FragmentSetGenerationTask(Task):
    """Represents a task for generating sets of entity fragments."""

    __entity_fragment_sets: Path

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.EntityGeneration  # Or define a new TaskType if needed

    @property
    def entity_fragment_sets(self) -> Path:
        """Return path to entity fragment sets."""
        return self.__entity_fragment_sets

    def __init__(
        self,
        name: str,
        entity_fragment_sets: Path,
        schema: Path | None = None,
        root_for_relative_paths: Path | None = None,
    ):
        """Initialize a fragment set generation task."""
        super().__init__(name, schema, root_for_relative_paths=root_for_relative_paths)
        self.__entity_fragment_sets = resolve_path(entity_fragment_sets, root_for_relative_paths)

    def read_items(self) -> Iterable[list[Entity]]:
        """
        Read items as lists of entity fragments.

        Returns:
            Iterable[list[Entity]]: An iterable of lists of Entity objects.
        """
        return EntityListJsonlReader(self.entity_fragment_sets).read_items()

    def write_items(self, path: Path, items: Iterable[Any]) -> None:
        """
        Write items to the specified path.

        Args:
            path: The file path where the items should be written.
            items: An iterable of items to be written to the file.
        """
        raise NotImplementedError("write_items is not implemented for FragmentSetGenerationTask")
