# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Task interface implementation for MS-KeBAB benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

from kebab.contracts.entity import PropertySchema


class TaskType(Enum):
    """Allowed task types."""

    EntityGeneration = 1
    Extraction = 2
    Linking = 3
    Clustering = 4


class Task(ABC):
    """Represents a benchmark task with its data files."""

    __name: str
    __schema: Path

    def __init__(self, name: str, schema: str):
        """Initialize a task."""
        self.__name = name
        self.__schema = Path(schema)

    @property
    def name(self) -> str:
        """Return task name."""
        return self.__name

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
    def evaluate(self, output_to_evaluate: Path, eval_result_path: Path | None = None) -> dict[str, float]:
        """
        Evaluate an output for the task.

        Args:
            output_to_evaluate: The output that needs to be evaluated.
            eval_result_path: Optional path to save the evaluation result.

        Returns:
            dict[str, float]: A dictionary containing the evaluation metrics.
        """

    def read_schema(self) -> PropertySchema:
        """
        Read property schema from a file.

        Returns:
            PropertySchema: A PropertySchema read from a file.
        """
        schema = PropertySchema.from_file(self.__schema)
        return schema
