# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from itertools import zip_longest
from pathlib import Path

from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.utils.io_helpers import EntityPairJsonlReader, ItemJsonlReader, ItemJsonlWriter, save_dict_to_json


class LinkingTask(Task):
    """Represents a linking benchmark task with its data files."""

    __data_entity_fragment_pairs: Path
    __data_ground_truth_boolean: Path | None

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.Linking

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
        entity_fragment_pairs: str,
        schema: str,
        ground_truth_boolean: str | None = None,
    ):
        """Initialize a linking task."""
        super().__init__(name, schema)
        self.__data_entity_fragment_pairs = Path(entity_fragment_pairs)
        if ground_truth_boolean is not None:
            self.__data_ground_truth_boolean = Path(ground_truth_boolean)

    def read_items(self) -> Iterable[tuple[tuple[Entity, Entity], bool | None]]:
        """
        Read data items with optional ground-truth entity fragment pairs.

        Returns:
            Iterable[tuple[tuple[Entity, Entity], bool | None]]: An iterable of tuples, where each
            tuple contains:
                - A pair of `Entity` objects.
                - An optional boolean value providing the ground-truth labels.
        """
        entity_pairs = EntityPairJsonlReader(self.data_entity_fragment_pairs).read_items()
        labels = (
            ItemJsonlReader[bool](self.data_ground_truth_boolean, converter=bool).read_items()
            if self.data_ground_truth_boolean is not None
            else iter([])
        )
        return zip_longest(entity_pairs, labels)

    def write_items(self, path: Path, items: Iterable[bool]) -> None:
        """
        Write items, i.e. boolean labels, to the specified path.

        Args:
            path: The file path where the boolean labels should be written.
            items: An iterable of boolean labels to be written to the file.
        """
        ItemJsonlWriter[bool](path).write_items(items)

    def evaluate(
        self,
        output_to_evaluate: Path,
        eval_result_path: Path | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the linking task."""
        if self.data_ground_truth_boolean is None:
            raise ValueError("Ground truth data is required for evaluation.")

        predictions = list(ItemJsonlReader[float](output_to_evaluate, converter=float).read_items())

        # The predictions are either true/false (exactly 0/1 floats), or we treat them as log-odds
        probabilistic = sorted(set(predictions)) != [0, 1]

        # TODO(pmyshkov): For now, we simply use a threshold of 0 to convert the predictions to boolean labels
        pred_labels = [pred > 0 for pred in predictions]
        gt_labels = list(ItemJsonlReader[bool](self.data_ground_truth_boolean, converter=bool).read_items())

        metrics = defaultdict(float)
        for log_odds, pred_label, gt_label in zip(predictions, pred_labels, gt_labels, strict=False):
            metrics["total"] += 1

            if pred_label == gt_label:
                if pred_label:
                    metrics["true_positive"] += 1
                else:
                    metrics["true_negative"] += 1
            else:
                if pred_label:
                    metrics["false_positive"] += 1
                else:
                    metrics["false_negative"] += 1

            if probabilistic:
                metrics["log_prob"] += (
                    math.log(1 / (1 + math.exp(-log_odds))) if gt_label else math.log(1 / (1 + math.exp(log_odds)))
                )

        # add an empirical log-prob estimated from the confusion matrix
        if metrics["true_positive"] > 0 and metrics["true_negative"] > 0:
            metrics["precision"] = metrics["true_positive"] / (metrics["true_positive"] + metrics["false_positive"])
            metrics["recall"] = metrics["true_positive"] / (metrics["true_positive"] + metrics["false_negative"])
            metrics["tpr"] = metrics["precision"]
            metrics["tnr"] = metrics["true_negative"] / (metrics["true_negative"] + metrics["false_negative"])

            if 0 < metrics["tpr"] < 1 and 0 < metrics["tnr"] < 1:
                log_prob = metrics["true_positive"] * math.log(metrics["tpr"])
                log_prob += metrics["false_positive"] * math.log(1 - metrics["tpr"])
                log_prob += metrics["true_negative"] * math.log(metrics["tnr"])
                log_prob += metrics["false_negative"] * math.log(1 - metrics["tnr"])
                metrics["empirical_log_prob"] = log_prob

        metrics = dict(metrics)

        if eval_result_path:
            save_dict_to_json(metrics, eval_result_path)

        return metrics
