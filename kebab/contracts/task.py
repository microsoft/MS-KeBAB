# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Task interface implementation for MS-KeBAB benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Any

from kebab.contracts.entity import PropertySchema
from kebab.utils.io_helpers import resolve_path


class TaskType(Enum):
    """Allowed task types."""

    EntityGeneration = 1
    Extraction = 2
    Linking = 3
    Clustering = 4
    TextCompletionUsingDocuments = 5
    TextCompletionUsingKB = 6
    QuestionAnsweringUsingDocuments = 7
    QuestionAnsweringUsingKB = 8
    FragmentSetGeneration = 9


class Task(ABC):
    """Represents a benchmark task with its data files."""

    __name: str
    __schema: Path | None
    __root_for_relative_paths: Path | None

    def __init__(self, name: str, schema: Path | None = None, root_for_relative_paths: Path | None = None):
        """Initialize a task."""
        self.__name = name
        self.__root_for_relative_paths = resolve_path(root_for_relative_paths) if root_for_relative_paths else None
        self.__schema = resolve_path(schema, root_for_relative_paths) if schema is not None else None

    @property
    def name(self) -> str:
        """Return task name."""
        return self.__name

    @property
    def path_safe_name(self) -> str:
        """Return task name with path-safe characters."""
        return self.name.translate(str.maketrans("", "", r'<>:"/\\|?*'))

    @property
    def root_for_relative_paths(self) -> Path | None:
        """Return the path to the data root."""
        return self.__root_for_relative_paths

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Return task type."""

    @abstractmethod
    def read_items(self) -> Iterable[Any]:
        """
        Read data items from a data source.

        Returns:
            Iterable[Any]: An iterable of items read from the data source. The specific type of
            items depends on the implementation in derived classes.
        """

    @abstractmethod
    def write_items(self, path: Path, items: Iterable[Any]) -> None:
        """
        Write output items to a specified file.

        Args:
            path: The file path where the items should be written.
            items: An iterable of items to write to the file. The specific type of items depends on
            the implementation in derived classes.
        """

    @abstractmethod
    def evaluate(
        self, predictions: Path, result_output_path: Path | None = None, logger: Logger | None = None
    ) -> dict[str, float]:
        """
        Evaluate an output for the task.

        Args:
            predictions: The path to the predictions file that needs to be evaluated.
            result_output_path: Optional path to save the evaluation results to.
            logger: Optional logger to log the evaluation process.

        Returns:
            dict[str, float]: A dictionary containing the evaluation metrics.
        """

    def read_schema(self) -> PropertySchema | None:
        """
        Read property schema from a file.

        Returns:
            PropertySchema: A PropertySchema read from a file.
        """
        if not self.__schema or not self.__schema.exists():
            return None

        schema = PropertySchema.from_file(self.__schema)
        return schema
