# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from itertools import zip_longest
from logging import Logger
from pathlib import Path

from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.utils.io_helpers import (
    EntityPairJsonlReader,
    ItemJsonlReader,
    ItemJsonlWriter,
    resolve_path,
    save_dict_to_json,
)


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
        data_path: Path | None = None,
    ):
        """Initialize a linking task."""
        super().__init__(name, schema, data_path=data_path)
        self.__data_entity_fragment_pairs = resolve_path(entity_fragment_pairs, data_path)
        if ground_truth_boolean is not None:
            self.__data_ground_truth_boolean = resolve_path(ground_truth_boolean, data_path)

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
        threshold: float = 0,
        baseline_model_output: Path | None = None,
        logger: Logger | None = None,
    ) -> dict[str, float]:
        """
        Evaluate an output for the linking task.

        Args:
            output_to_evaluate: Path to model output predictions.
            eval_result_path: Optional path to save evaluation metrics as JSON.
            threshold: Threshold to convert predictions to boolean labels.
            baseline_model_output: Optional path to baseline model predictions.
            logger: Optional logger for logging evaluation summaries.

        Returns:
            dict[str, float]: Evaluation metrics dictionary.
        """
        if self.data_ground_truth_boolean is None:
            raise ValueError("Ground truth data is required for evaluation.")

        predictions = list(ItemJsonlReader[float](output_to_evaluate, converter=float).read_items())

        # The predictions are either true/false (exactly 0/1 floats), or we treat them as log-odds
        probabilistic = sorted(set(predictions)) != [0, 1]

        # TODO(pmyshkov): For now, we simply use the provided threshold to convert the predictions to boolean labels
        predicted_labels = [pred > threshold for pred in predictions]

        pairs, gt_labels = zip(*self.read_items(), strict=False)

        if len(predicted_labels) != len(gt_labels):
            raise ValueError("The number of predicted labels does not match the number of ground truth labels.")

        baseline_predictions = []
        baseline_predicted_labels = []

        if baseline_model_output and baseline_model_output.exists():
            baseline_predictions = list(ItemJsonlReader[float](baseline_model_output, converter=float).read_items())
            if len(predictions) != len(baseline_predictions):
                raise ValueError("The number of predictions does not match the number of baseline predictions.")

            baseline_predicted_labels = [pred > threshold for pred in baseline_predictions]
            if len(baseline_predicted_labels) != len(gt_labels):
                raise ValueError(
                    "The number of baseline predicted labels does not match the number of ground truth labels."
                )

        metrics = defaultdict(float)
        detail_records = []

        for pair, gt_label, log_odds, pred_label, b_log_odds, b_pred_label in zip_longest(
            pairs,
            gt_labels,
            predictions,
            predicted_labels,
            baseline_predictions,
            baseline_predicted_labels,
        ):
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

            left = pair[0]
            right = pair[1]
            ent_type = set()
            ent_type = ent_type.union(left.metadata.get("type", set()))
            ent_type = ent_type.union(right.metadata.get("type", set()))

            prop_pattern = tuple(sorted([tuple(sorted(left.properties)), tuple(sorted(right.properties))]))

            left_props = set(left.properties.keys())
            right_props = set(right.properties.keys())
            overlap_props = sorted(left_props.intersection(right_props))
            prop_overlap_num = len(overlap_props)

            left_name = set(n.lower() for n in left.properties["name"])
            right_name = set(n.lower() for n in right.properties["name"])
            name_overlap = len(left_name.intersection(right_name)) >= 1

            details = {
                "left": dict(left.properties),
                "right": dict(right.properties),
                "entity_type": sorted(ent_type),
                "overlap_props": overlap_props,
                "prop_overlap_num": prop_overlap_num,
                "prop_pattern": prop_pattern,
                "name_overlap": name_overlap,
                "label": gt_label,
                "cal_log_odds": log_odds - threshold,
                "predicted_label": pred_label,
                "baseline_log_odds": b_log_odds if baseline_predictions else None,
                "baseline_predicted_label": b_pred_label if baseline_predictions else None,
                "left_entity_id": left.entity_id,
                "right_entity_id": right.entity_id,
            }

            detail_records.append(details)

        # add an empirical log-prob estimated from the confusion matrix
        if metrics["true_positive"] > 0 and metrics["true_negative"] > 0:
            metrics["precision"] = metrics["true_positive"] / (metrics["true_positive"] + metrics["false_positive"])
            metrics["recall"] = metrics["true_positive"] / (metrics["true_positive"] + metrics["false_negative"])
            metrics["ppv"] = metrics["precision"]
            metrics["npv"] = metrics["true_negative"] / (metrics["true_negative"] + metrics["false_negative"])

            if 0 < metrics["ppv"] < 1 and 0 < metrics["npv"] < 1:
                log_prob = metrics["true_positive"] * math.log(metrics["ppv"])
                log_prob += metrics["false_positive"] * math.log(1 - metrics["ppv"])
                log_prob += metrics["true_negative"] * math.log(metrics["npv"])
                log_prob += metrics["false_negative"] * math.log(1 - metrics["npv"])
                metrics["empirical_log_prob"] = log_prob

        metrics = dict(metrics)

        if "log_prob" not in metrics:
            metrics["log_prob"] = float("nan")

        metrics_order = (
            "log_prob",
            "empirical_log_prob",
            "precision",
            "recall",
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
            "ppv",
            "npv",
            "total",
        )
        metrics = {key: metrics[key] for key in metrics_order if key in metrics}

        if eval_result_path:
            save_dict_to_json(metrics, eval_result_path)

        headers = "\t".join(metrics.keys())
        values = "\t".join(f"{value:.4f}" for value in metrics.values())
        if logger:
            logger.info(f"{headers}\n{values}")
        else:
            print(f"{headers}\n{values}")

        # write detailed records as jsonl
        detailed_output_path = output_to_evaluate.with_suffix(".detailed.jsonl")
        ItemJsonlWriter[dict](detailed_output_path).write_items(detail_records)

        # write detailed records as csv
        csv_output_path = output_to_evaluate.with_suffix(".detailed.csv")
        with csv_output_path.open("w", encoding="utf-8") as f:
            headers = detail_records[0].keys()
            f.write("\t".join(headers) + "\n")
            for item in detail_records:
                f.write("\t".join(str(item.get(header, "")) for header in headers) + "\n")

        return metrics
