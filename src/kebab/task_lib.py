# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Task and instance interface implementation for MS-KeBAB benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import ClassVar, Self


class TaskType(Enum):
    """Allowed task types."""

    Extraction = 1
    Linking = 2


class Task:
    """Represents a benchmark task with its task instances."""

    __task_type: TaskType
    __instances: dict[str, TaskInstance]
    __created_task_types: ClassVar[set[TaskType]] = set()

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return self.__task_type

    @property
    def instances(self) -> dict[str, TaskInstance]:
        """Return task instances."""
        return self.__instances.copy()  # To disallow modifying this dictionary from outside of this class

    def __new__(cls, task_type: TaskType) -> Self:
        """Disallow instantiating multiple Task objects for same task type."""
        if task_type in cls.__created_task_types:
            raise ValueError("Not allowed to create multiple task objects of type {task_type}")
        cls.__created_task_types.add(task_type)
        return super().__new__(cls)

    def __init__(self, task_type: TaskType):
        """Initialize a task."""
        self.__task_type = task_type
        self.__instances = {}

    def add_instance(self, instance: TaskInstance):
        """Add a task instance."""
        if instance.name in self.__instances:
            raise ValueError(f"Instance with name '{instance.name}' already exists.")
        self.__instances[instance.name] = instance


class TaskInstance(ABC):
    """Represents a benchmark task instance with its data files."""

    __name: str
    __task: Task

    def __init__(self, name: str, task: Task):
        """Initialize a task instance."""
        self.__name = name
        self.__task = task
        self.__task.add_instance(self)

    @property
    def name(self) -> str:
        """Return task instance name."""
        return self.__name

    @property
    def task(self) -> Task:
        """Return task."""
        return self.__task

    @classmethod
    def create_task_instance(cls, instance__name: str, task: Task, data_dict: dict[str, str]) -> TaskInstance:
        """Create appropriate task instance."""
        match task.task_type:
            case TaskType.Extraction:
                instance = ExtractionTaskInstance(instance__name, task, **data_dict)
            case TaskType.Linking:
                instance = LinkingTaskInstance(instance__name, task, **data_dict)
        return instance

    @abstractmethod
    def evaluate(self, output_to_evaluate: Path) -> dict[str, float]:
        """Evaluate an output for the task instance."""


class ExtractionTaskInstance(TaskInstance):
    """Represents an extraction benchmark task instance with its data files."""

    __data_extracts: Path
    __data_ground_truth_extracted_entities: Path

    @property
    def data_extracts(self) -> Path:
        """Return path to extracts."""
        return self.__data_extracts

    @property
    def data_ground_truth_extracted_entities(self) -> Path:
        """Return path to ground truth extracted entities."""
        return self.__data_ground_truth_extracted_entities

    def __init__(
        self,
        name: str,
        task: Task,
        extracts: str,
        ground_truth_extracted_entities: str | None = None,
    ):
        """Initialize an extraction task instance."""
        super().__init__(name, task)
        self.__data_extracts = Path(extracts)
        if ground_truth_extracted_entities is not None:
            self.__data_ground_truth_extracted_entities = Path(ground_truth_extracted_entities)

    def evaluate(
        self,
        output_to_evaluate: Path,  # noqa: ARG002
    ) -> dict[str, float]:
        """Evaluate an output for the extraction task instance."""
        if hasattr(self, "__data_ground_truth_extracted_entities"):
            raise ValueError("Can not evaluate on heldout Extraction task instance")

        # TODO(bmitra): Implement actual metric computation
        return {"primary_extraction_metric": 0.8, "secondary_extraction_metric": 0.6}


class LinkingTaskInstance(TaskInstance):
    """Represents an linking benchmark task instance with its data files."""

    __data_entity_fragment_pairs: Path
    __data_ground_truth_boolean: Path

    @property
    def data_entity_fragment_pairs(self) -> Path:
        """Return path to entity fragment pairs."""
        return self.__data_entity_fragment_pairs

    @property
    def data_ground_truth_boolean(self) -> Path:
        """Return path to ground truth boolean."""
        return self.__data_ground_truth_boolean

    def __init__(
        self,
        name: str,
        task: Task,
        entity_fragment_pairs: str,
        ground_truth_boolean: str | None = None,
    ):
        """Initialize an linking task instance."""
        super().__init__(name, task)
        self.__data_entity_fragment_pairs = Path(entity_fragment_pairs)
        if ground_truth_boolean is not None:
            self.__data_ground_truth_boolean = Path(ground_truth_boolean)

    def evaluate(
        self,
        output_to_evaluate: Path,  # noqa: ARG002
    ) -> dict[str, float]:
        """Evaluate an output for the linking task instance."""
        if hasattr(self, "__data_ground_truth_boolean"):
            raise ValueError("Can not evaluate on heldout Linking task instance")

        # TODO(bmitra): Implement actual metric computation
        return {"primary_linking_metric": 0.8, "secondary_linking_metric": 0.6}
