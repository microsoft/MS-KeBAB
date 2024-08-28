# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Task and instance interface implementation for MS-KeBAB benchmark."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path


class Task:
    """Represents a benchmark task with its task instances."""

    _name: str
    _instances: dict[str, TaskInstance]

    @property
    def name(self) -> str:
        """Return task name."""
        return self._name

    @property
    def instances(self) -> dict[str, TaskInstance]:
        """Return task instances."""
        return (
            self._instances.copy()
        )  # To disallow modifying this dictionary from outside of this class

    def __init__(self, name: str):
        """Initialize a task."""
        self._name = name
        self._instances = {}

    def add_instance(self, instance: TaskInstance):
        """Add a task instance."""
        if instance.name in self._instances:
            raise ValueError(f"Instance with name '{instance.name}' already exists.")
        self._instances[instance.name] = instance


class TaskInstance:
    """Represents a benchmark task instance with its data files."""

    _name: str
    _parent: Task

    @property
    def name(self) -> str:
        """Return task instance name."""
        return self._name

    @property
    def parent(self) -> Task:
        """Return parent task."""
        return self._parent

    @abstractmethod
    def evaluate(self, output_to_evaluate: Path, set_type: str) -> dict[str, float]:
        """Evaluate an output for the task instance."""
        pass


class ExtractionTaskInstance(TaskInstance):
    """Represents an extraction benchmark task instance with its data files."""

    _data_train_extracts: Path
    _data_train_ground_truth_extracted_entities: Path
    _data_test_extracts: Path
    _data_test_ground_truth_extracted_entities: Path

    @property
    def data_train_extracts(self) -> Path:
        """Return path to train extracts."""
        return self._data_train_extracts

    @property
    def data_train_ground_truth_extracted_entities(self) -> Path:
        """Return path to train ground truth extracted entities."""
        return self._data_train_ground_truth_extracted_entities

    @property
    def data_test_extracts(self) -> Path:
        """Return path to test extracts."""
        return self._data_test_extracts

    @property
    def data_test_ground_truth_extracted_entities(self) -> Path:
        """Return path to test ground truth extracted entities."""
        return self._data_test_ground_truth_extracted_entities

    def __init__(
        self,
        name: str,
        parent: Task,
        train_extracts: str,
        train_ground_truth_extracted_entities: str,
        test_extracts: str,
        test_ground_truth_extracted_entities: str = "",
    ):
        """Initialize an extraction task instance."""
        self._name = name
        self._parent = parent
        self._data_train_extracts = Path(train_extracts)
        self._data_train_ground_truth_extracted_entities = Path(
            train_ground_truth_extracted_entities
        )
        self._data_test_extracts = Path(test_extracts)
        if len(test_ground_truth_extracted_entities) > 0:
            self._data_test_ground_truth_extracted_entities = Path(
                test_ground_truth_extracted_entities
            )

    def evaluate(
        self, output_to_evaluate: Path, set_type: str  # noqa: ARG002
    ) -> dict[str, float]:
        """Evaluate an output for the extraction task instance."""
        if set_type == "train":
            ground_truth_extracted_entities = (
                self._data_train_ground_truth_extracted_entities
            )
        elif set_type == "test":
            ground_truth_extracted_entities = (  # noqa: F841
                self._data_test_ground_truth_extracted_entities
            )
        else:
            raise ValueError("Unknown set type: {set_type}")

        # TODO(bmitra): Implement actual metric computation
        return {"primary_extraction_metric": 0.8, "secondary_extraction_metric": 0.6}


class LinkingTaskInstance(TaskInstance):
    """Represents an linking benchmark task instance with its data files."""

    _data_train_entity_fragment_pairs: Path
    _data_train_ground_truth_boolean: Path
    _data_test_entity_fragment_pairs: Path
    _data_test_ground_truth_boolean: Path

    @property
    def data_train_entity_fragment_pairs(self) -> Path:
        """Return path to train entity fragment pairs."""
        return self._data_train_entity_fragment_pairs

    @property
    def data_train_ground_truth_boolean(self) -> Path:
        """Return path to train ground truth boolean."""
        return self._data_train_ground_truth_boolean

    @property
    def data_test_entity_fragment_pairs(self) -> Path:
        """Return path to test entity fragment pairs."""
        return self._data_test_entity_fragment_pairs

    @property
    def data_test_ground_truth_boolean(self) -> Path:
        """Return path to test ground truth boolean."""
        return self._data_test_ground_truth_boolean

    def __init__(
        self,
        name: str,
        parent: Task,
        train_entity_fragment_pairs: str,
        train_ground_truth_boolean: str,
        test_entity_fragment_pairs: str,
        test_ground_truth_boolean: str = "",
    ):
        """Initialize an linking task instance."""
        self._name = name
        self._parent = parent
        self._data_train_entity_fragment_pairs = Path(train_entity_fragment_pairs)
        self._data_train_ground_truth_boolean = Path(train_ground_truth_boolean)
        self._data_test_entity_fragment_pairs = Path(test_entity_fragment_pairs)
        if len(test_ground_truth_boolean) > 0:
            self._data_test_ground_truth_boolean = Path(test_ground_truth_boolean)

    def evaluate(
        self, output_to_evaluate: Path, set_type: str  # noqa: ARG002
    ) -> dict[str, float]:
        """Evaluate an output for the linking task instance."""
        if set_type == "train":
            ground_truth_boolean = self._data_train_ground_truth_boolean
        elif set_type == "test":
            ground_truth_boolean = self._data_test_ground_truth_boolean  # noqa: F841
        else:
            raise ValueError("Unknown set type: {set_type}")

        # TODO(bmitra): Implement actual metric computation
        return {"primary_linking_metric": 0.8, "secondary_linking_metric": 0.6}
