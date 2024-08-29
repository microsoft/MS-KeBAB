# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Task and instance interface implementation for MS-KeBAB benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Self


class TaskType(Enum):
    """Allowed task types."""
    Extraction = 1
    Linking = 2

class SetType(Enum):
    """Allowed set types."""
    Train = 1
    Test = 2

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
    __parent: Task

    def __init__(self, name: str, parent: Task):
        """Initialize a task instance."""
        self.__name = name
        self.__parent = parent
        self.__parent.add_instance(self)

    @property
    def name(self) -> str:
        """Return task instance name."""
        return self.__name

    @property
    def parent(self) -> Task:
        """Return parent task."""
        return self.__parent

    @classmethod
    def create_task_instance(cls, instance__name: str, instance_config: dict[str, Any], parent: Task) -> TaskInstance:
        """Create appropriate task instance and optionally parent task, if not provided."""
        task_type = TaskType[instance_config["task"]]
        match task_type:
            case TaskType.Extraction:
                instance = ExtractionTaskInstance(instance__name, parent, **instance_config["data"])
            case TaskType.Linking:
                instance = LinkingTaskInstance(instance__name, parent, **instance_config["data"])
        return instance

    @abstractmethod
    def evaluate(self, output_to_evaluate: Path, set_type: SetType) -> dict[str, float]:
        """Evaluate an output for the task instance."""


class ExtractionTaskInstance(TaskInstance):
    """Represents an extraction benchmark task instance with its data files."""

    __data_train_extracts: Path
    __data_train_ground_truth_extracted_entities: Path
    __data_test_extracts: Path
    __data_test_ground_truth_extracted_entities: Path

    @property
    def data_train_extracts(self) -> Path:
        """Return path to train extracts."""
        return self.__data_train_extracts

    @property
    def data_train_ground_truth_extracted_entities(self) -> Path:
        """Return path to train ground truth extracted entities."""
        return self.__data_train_ground_truth_extracted_entities

    @property
    def data_test_extracts(self) -> Path:
        """Return path to test extracts."""
        return self.__data_test_extracts

    @property
    def data_test_ground_truth_extracted_entities(self) -> Path:
        """Return path to test ground truth extracted entities."""
        return self.__data_test_ground_truth_extracted_entities

    def __init__(
        self,
        name: str,
        parent: Task,
        train_extracts: str,
        train_ground_truth_extracted_entities: str,
        test_extracts: str,
        test_ground_truth_extracted_entities: str | None = None
    ):
        """Initialize an extraction task instance."""
        super().__init__(name, parent)
        self.__data_train_extracts = Path(train_extracts)
        self.__data_train_ground_truth_extracted_entities = Path(train_ground_truth_extracted_entities)
        self.__data_test_extracts = Path(test_extracts)
        if test_ground_truth_extracted_entities is not None:
            self.__data_test_ground_truth_extracted_entities = Path(test_ground_truth_extracted_entities)

    def evaluate(
        self,
        output_to_evaluate: Path,  # noqa: ARG002
        set_type: SetType,
    ) -> dict[str, float]:
        """Evaluate an output for the extraction task instance."""
        match set_type:
            case SetType.Train:
                ground_truth_extracted_entities = self.__data_train_ground_truth_extracted_entities
            case SetType.Test:
                ground_truth_extracted_entities = self.__data_test_ground_truth_extracted_entities  # noqa: F841

        # TODO(bmitra): Implement actual metric computation
        return {"primary_extraction_metric": 0.8, "secondary_extraction_metric": 0.6}


class LinkingTaskInstance(TaskInstance):
    """Represents an linking benchmark task instance with its data files."""

    __data_train_entity_fragment_pairs: Path
    __data_train_ground_truth_boolean: Path
    __data_test_entity_fragment_pairs: Path
    __data_test_ground_truth_boolean: Path

    @property
    def data_train_entity_fragment_pairs(self) -> Path:
        """Return path to train entity fragment pairs."""
        return self.__data_train_entity_fragment_pairs

    @property
    def data_train_ground_truth_boolean(self) -> Path:
        """Return path to train ground truth boolean."""
        return self.__data_train_ground_truth_boolean

    @property
    def data_test_entity_fragment_pairs(self) -> Path:
        """Return path to test entity fragment pairs."""
        return self.__data_test_entity_fragment_pairs

    @property
    def data_test_ground_truth_boolean(self) -> Path:
        """Return path to test ground truth boolean."""
        return self.__data_test_ground_truth_boolean

    def __init__(
        self,
        name: str,
        parent: Task,
        train_entity_fragment_pairs: str,
        train_ground_truth_boolean: str,
        test_entity_fragment_pairs: str,
        test_ground_truth_boolean: str | None = None
    ):
        """Initialize an linking task instance."""
        super().__init__(name, parent)
        self.__data_train_entity_fragment_pairs = Path(train_entity_fragment_pairs)
        self.__data_train_ground_truth_boolean = Path(train_ground_truth_boolean)
        self.__data_test_entity_fragment_pairs = Path(test_entity_fragment_pairs)
        if test_ground_truth_boolean is not None:
            self.__data_test_ground_truth_boolean = Path(test_ground_truth_boolean)

    def evaluate(
        self,
        output_to_evaluate: Path,  # noqa: ARG002
        set_type: SetType,
    ) -> dict[str, float]:
        """Evaluate an output for the linking task instance."""
        match set_type:
            case SetType.Train:
                ground_truth_boolean = self.__data_train_ground_truth_boolean
            case SetType.Test:
                ground_truth_boolean = self.__data_test_ground_truth_boolean  # noqa: F841

        # TODO(bmitra): Implement actual metric computation
        return {"primary_linking_metric": 0.8, "secondary_linking_metric": 0.6}
