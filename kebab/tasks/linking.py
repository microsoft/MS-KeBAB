# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections.abc import Iterable
from itertools import zip_longest
from pathlib import Path

from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskInstance
from kebab.utils.io_helpers import BooleanFileReader, BooleanFileWriter, EntityPairJsonlReader, save_dict_to_json


class LinkingTaskInstance(TaskInstance):
    """Represents an linking benchmark task instance with its data files."""

    __data_entity_fragment_pairs: Path
    __data_ground_truth_boolean: Path | None

    @property
    def data_entity_fragment_pairs(self) -> Path:
        """Return path to entity fragment pairs."""
        return self.__data_entity_fragment_pairs

    @property
    def data_ground_truth_boolean(self) -> Path | None:
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

    def read_items(self) -> Iterable[tuple[tuple[Entity, Entity], bool | None]]:
        """
        Read data items with optional ground-truth entity fragment pairs.

        Returns:
            Iterable[Tuple[Tuple[Entity, Entity], bool | None]]: An iterable of tuples, where each
            tuple contains:
                - A pair of `Entity` objects.
                - An optional boolean value providing the ground-truth labels.
        """
        entity_pairs = EntityPairJsonlReader(self.data_entity_fragment_pairs).read_items()
        booleans = (
            BooleanFileReader(self.data_ground_truth_boolean).read_items()
            if self.data_ground_truth_boolean is not None
            else iter([])
        )
        return zip_longest(entity_pairs, booleans)

    def write_items(self, path: Path, items: Iterable[bool]) -> None:
        """
        Write items, i.e. boolean labels, to the specified path.

        Args:
            path: The file path where the boolean labels should be written.
            items: An iterable of boolean labels to be written to the file.
        """
        BooleanFileWriter(path).write_items(items)

    def evaluate(
        self,
        output_to_evaluate: Path,  # noqa: ARG002
        eval_result_path: Path | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the linking task instance."""
        if hasattr(self, "__data_ground_truth_boolean"):
            raise ValueError("Can not evaluate on heldout Linking task instance")

        # TODO(bmitra): Implement actual metric computation
        eval_result = {"primary_linking_metric": 0.8, "secondary_linking_metric": 0.6}

        if eval_result_path:
            save_dict_to_json(eval_result, eval_result_path)

        return eval_result
