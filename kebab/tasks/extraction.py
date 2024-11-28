# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pathlib import Path

from kebab.contracts.task import Task, TaskInstance


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
