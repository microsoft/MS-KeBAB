# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Task and instance interface implementation for MS-KeBAB benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Iterable, Self


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
    __task_type_registry: ClassVar[dict[TaskType, type[TaskInstance]]] = {}

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
    def register_task_type(cls, task_type: TaskType, task_instance_class: type[TaskInstance]):
        """Register a task type with its corresponding class."""
        cls.__task_type_registry[task_type] = task_instance_class

    @classmethod
    def create_task_instance(cls, instance_name: str, task: Task, data_dict: dict[str, str]) -> TaskInstance:
        """Create appropriate task instance."""
        task_class = cls.__task_type_registry.get(task.task_type)
        if task_class is None:
            raise ValueError(f"No registered class for task type: {task.task_type}.")
        return task_class(instance_name, task, **data_dict)

    @abstractmethod
    def read_items(self) -> Iterable[Any]:
        pass

    @abstractmethod
    def evaluate(self, output_to_evaluate: Path, eval_result_path: Path | None = None) -> dict[str, float]:
        """
        Evaluate an output for the task instance.

        Args:
        output_to_evaluate: The output that needs to be evaluated.
        eval_result_path: Optional path to save the evaluation result.

        Returns:
        dict[str, float]: A dictionary containing the evaluation metrics.
        """
