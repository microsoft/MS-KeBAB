# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections.abc import Iterable
from itertools import zip_longest
from pathlib import Path
from typing import Any, ClassVar

from kebab.contracts.entity import Entity, PropertySchema
from kebab.contracts.task import Task, TaskInstance
from kebab.tasks.extraction.metrics.aesop.calculator import AesopMetricCalculator, make_default_aesop_config
from kebab.tasks.extraction.metrics.calculator import ExtractionOutput
from kebab.utils.io_helpers import DocumentJsonlReader, EntityListJsonlReader, EntityListJsonlWriter, save_dict_to_json


class ExtractionTaskInstance(TaskInstance):
    """Represents an extraction benchmark task instance with its data files."""

    __data_extracts: Path
    __data_ground_truth_extracted_entities: Path | None
    __metric_calculator_cls: ClassVar[dict[str, Any]] = {
        "aesop": AesopMetricCalculator
    }

    @property
    def data_extracts(self) -> Path:
        """Return path to extracts."""
        return self.__data_extracts

    @property
    def data_ground_truth_extracted_entities(self) -> Path | None:
        """Return path to ground truth extracted entities."""
        return self.__data_ground_truth_extracted_entities

    def __init__(
        self,
        name: str,
        task: Task,
        extracts: str,
        property_schema: str,
        ground_truth_extracted_entities: str | None = None,
        metric_configs: dict[str, Any] | None = None,
    ):
        """Initialize an extraction task instance."""
        super().__init__(name, task, property_schema)
        self.__data_extracts = Path(extracts)
        if ground_truth_extracted_entities is not None:
            self.__data_ground_truth_extracted_entities = Path(ground_truth_extracted_entities)
        self.property_schema = property_schema
        self.metric_configs = metric_configs


    def read_items(self) -> Iterable[ExtractionOutput]:
        """
        Read data items, with optional ground-truth extracted entities.

        Returns:
            Iterable[ExtractionOutputs]: An iterable of ExtractionOutput, each containing a
            `Document` and an optional list of `Entity` objects.
        """
        # TODO (allenwang): Pass `Cache` into this instance to resolve dataset links to local paths
        # after instances.json is updated with valid links.
        extracts = DocumentJsonlReader(self.data_extracts).read_items()
        entity_lists = (
            EntityListJsonlReader(self.data_ground_truth_extracted_entities).read_items()
            if self.data_ground_truth_extracted_entities is not None
            else iter([])
        )
        return (ExtractionOutput(document=extract, entities=entity_list) for extract, entity_list in zip_longest(extracts, entity_lists))

    def write_items(self, path: Path, items: Iterable[list[Entity]]) -> None:
        """
        Write output items, i.e. extracted entities, to the specified path.

        Args:
            path: The file path where the items should be written.
            items: An iterable of lists of extracted `Entity` objects to be written to the file.
        """
        EntityListJsonlWriter(path).write_items(items)

    def evaluate(
        self,
        output_to_evaluate: Path,
        eval_result_path: Path | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the extraction task instance."""
        if hasattr(self, "__data_ground_truth_extracted_entities"):
            raise ValueError("Can not evaluate on heldout Extraction task instance")

        if self.metric_configs is None:
            self.metric_configs = {
                "aesop": make_default_aesop_config(PropertySchema.from_file(Path(self.property_schema)), matching_threshold=0.5)
            }

        pred_entity_lists = EntityListJsonlReader(output_to_evaluate).read_items()

        gt_extractions = list(self.read_items())

        pred_extractions = [
            ExtractionOutput(document=document, entities=entity_list)
            for document, entity_list in zip((item.document for item in gt_extractions), pred_entity_lists, strict=True)]

        metrics = {}

        for metric_name, metric_config in self.metric_configs.items():
            metric_results = self.__metric_calculator_cls[metric_name](metric_config).run(pred_extractions, gt_extractions)
            metrics[metric_name] = metric_results
        if eval_result_path:
            save_dict_to_json(metrics, eval_result_path)

        return metrics
