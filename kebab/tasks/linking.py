# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from kebab.contracts.task import Task, TaskInstance
from kebab.utils.io_helpers import save_to_json


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

    def read_items(self) -> Iterable[Any]:
        # TODO (allenwang)
        yield None

    def evaluate(
        self,
        output_to_evaluate: Path,  # noqa: ARG002
        eval_result_path: Path | None = None
    ) -> dict[str, float]:
        """Evaluate an output for the linking task instance."""
        if hasattr(self, "__data_ground_truth_boolean"):
            raise ValueError("Can not evaluate on heldout Linking task instance")

        # TODO(bmitra): Implement actual metric computation
        eval_result = {"primary_linking_metric": 0.8, "secondary_linking_metric": 0.6}

        if eval_result_path:
            save_to_json(eval_result, eval_result_path)

        return eval_result
