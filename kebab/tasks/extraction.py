# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
from collections.abc import Iterable
from itertools import zip_longest
from pathlib import Path
from typing import ClassVar, cast

from kebab.contracts.entity import Entity, PropertySchema
from kebab.contracts.task import Task, TaskType
from kebab.tasks.metrics.extraction.calculator import ExtractionOutput, MetricCalculator, MetricConfig
from kebab.utils.io_helpers import (
    DocumentJsonlReader,
    EntityListJsonlReader,
    EntityListJsonlWriter,
    load_dict_from_json,
    resolve_path,
    save_dict_to_json,
)


class ExtractionTask(Task):
    """Represents an extraction benchmark task with its data files."""

    __extracts: Path
    __ground_truth: Path | None = None
    __default_metrics_config_path: Path = (
        Path(__file__).parents[1] / "configs" / "extraction" / "default_metrics_config.json"
    )
    __metric_calculator_cls: ClassVar[dict[str, type[MetricCalculator]]] = {}
    __metric_config_cls: ClassVar[dict[str, type[MetricConfig]]] = {}

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.Extraction

    @property
    def extracts(self) -> Path:
        """Return path to extracts."""
        return self.__extracts

    @property
    def ground_truth(self) -> Path | None:
        """Return path to ground truth extracted entities."""
        return self.__ground_truth

    def __init__(
        self,
        name: str,
        extracts: Path,
        schema: Path,
        ground_truth: Path | None = None,
        metrics_config: str | None = None,
        data: Path | None = None,
    ):
        """Initialize an extraction task."""
        super().__init__(name, schema, data=data)
        self.__extracts = resolve_path(extracts)
        if ground_truth is not None:
            self.__ground_truth = resolve_path(ground_truth, data)
        self.metrics_config = load_dict_from_json(
            resolve_path(metrics_config or self.__default_metrics_config_path, data)
        )

    def read_items(self) -> Iterable[ExtractionOutput]:
        """
        Read data items, with optional ground-truth extracted entities.

        Returns:
            Iterable[ExtractionOutputs]: An iterable of ExtractionOutput, each containing a
            `Document` and an optional list of `Entity` objects.
        """
        # TODO (allenwang): Pass `Cache` into this instance to resolve dataset links to local paths
        # after instances.json is updated with valid links.
        extracts = DocumentJsonlReader(self.extracts).read_items()
        entity_lists = (
            EntityListJsonlReader(self.ground_truth).read_items() if self.ground_truth is not None else iter([])
        )
        return (
            ExtractionOutput(document=extract, entities=entity_list)
            for extract, entity_list in zip_longest(extracts, entity_lists)
        )

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
        logger: logging.Logger | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the extraction task."""
        if hasattr(self, "__ground_truth"):
            raise ValueError("Can not evaluate on heldout Extraction task")

        pred_extractions = (
            ExtractionOutput(document=document, entities=entity_list)
            for document, entity_list in zip(
                (item.document for item in self.read_items()),
                EntityListJsonlReader(output_to_evaluate).read_items(),
                strict=True,
            )
        )

        metrics = {}

        from kebab.tasks.metrics.extraction.aesop.calculator import (
            ValueAveragedAesopConfig,
            ValueAveragedAesopMetricCalculator,
        )

        self.__metric_calculator_cls["aesop"] = ValueAveragedAesopMetricCalculator
        self.__metric_config_cls["aesop"] = ValueAveragedAesopConfig

        for metric_name, metric_config_dict in self.metrics_config.items():
            if logger is not None:
                logger.info(f"Evaluating metric: {metric_name}")
                logger.info(metric_config_dict)
            metric_calculator_cls = self.__metric_calculator_cls[metric_name]
            metric_config = self.__metric_config_cls[metric_name].from_dict(
                metric_config_dict,
                cast(PropertySchema, self.read_schema()),  # in extraction, schema is not optional
            )
            metric_results = metric_calculator_cls(metric_config, logger).run(pred_extractions, self.read_items())  # type: ignore
            metrics[metric_name] = metric_results
        if eval_result_path:
            save_dict_to_json(metrics, eval_result_path)

        return metrics
