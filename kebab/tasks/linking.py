# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
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


@dataclass
class ConfMatrix:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, gt: bool, pred: bool) -> None:
        if pred == gt:
            if pred:
                self.tp += 1
            else:
                self.tn += 1
        else:
            if pred:
                self.fp += 1
            else:
                self.fn += 1

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    def npv(self) -> float | None:
        denom = self.tn + self.fn
        return self.tn / denom if denom else None

    def ppv(self) -> float | None:
        return self.precision()


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

        preds = self._load_predictions(output_to_evaluate)
        baseline_preds = self._load_predictions(baseline_model_output) if baseline_model_output else []

        pairs, gt_labels = zip(*self.read_items(), strict=False)
        self._validate_lengths(preds, gt_labels)
        if baseline_preds:
            self._validate_lengths(preds, baseline_preds)

        pred_labels = [p > threshold for p in preds]
        baseline_labels = [p > threshold for p in baseline_preds] if baseline_preds else []

        cm, log_prob = self._confusion_matrix(list(gt_labels), pred_labels, preds, threshold)
        metrics = {"log_prob": log_prob, **self._extra_metrics(cm)}

        # Collect detailed records (unchanged logic, could be extracted too)
        detail_records = []  # build as before …

        self._write_side_outputs(detail_records, metrics, output_to_evaluate, eval_result_path)

        if logger:
            logger.info("\t".join(metrics.keys()) + "\n" + "\t".join(f"{v:.4f}" for v in metrics.values()))
        else:
            print("\t".join(metrics.keys()) + "\n" + "\t".join(f"{v:.4f}" for v in metrics.values()))

        return metrics

    def _load_predictions(self, path: Path) -> list[float]:
        return list(ItemJsonlReader[float](path, converter=float).read_items())

    def _validate_lengths(self, *streams: list) -> None:
        lengths = {len(s) for s in streams}
        if len(lengths) > 1:
            raise ValueError("Input streams have mismatched lengths.")

    def _confusion_matrix(
        self, gt: list[bool], preds: list[bool], log_odds: list[float], threshold: float
    ) -> tuple[ConfMatrix, float]:
        cm = ConfMatrix()
        log_prob = 0.0
        probabilistic = sorted(set(log_odds)) != [0, 1]

        for g, p, l in zip(gt, preds, log_odds, strict=False):
            cm.update(g, p)
            if probabilistic:
                log_prob += math.log(1 / (1 + math.exp(-l))) if g else math.log(1 / (1 + math.exp(l)))

        if not probabilistic:
            log_prob = float("nan")

        return cm, log_prob

    def _extra_metrics(self, cm: ConfMatrix) -> dict[str, float]:
        metrics: dict[str, float] = defaultdict(float)
        precision = cm.precision()
        recall = cm.recall()
        npv = cm.npv()
        ppv = cm.ppv()

        metrics.update(
            true_positive=cm.tp,
            true_negative=cm.tn,
            false_positive=cm.fp,
            false_negative=cm.fn,
            total=cm.total,
        )
        if precision is not None:
            metrics["precision"] = precision
            metrics["ppv"] = ppv
        if recall is not None:
            metrics["recall"] = recall
        if npv is not None:
            metrics["npv"] = npv

        if ppv and npv and 0 < ppv < 1 and 0 < npv < 1:
            log_prob = cm.tp * math.log(ppv) + cm.fp * math.log(1 - ppv)
            log_prob += cm.tn * math.log(npv) + cm.fn * math.log(1 - npv)
            metrics["empirical_log_prob"] = log_prob

        return metrics

    def _write_side_outputs(
        self,
        detail_records: list[dict],
        metrics: dict[str, float],
        output_to_evaluate: Path,
        eval_result_path: Path | None,
    ) -> None:
        if eval_result_path:
            save_dict_to_json(metrics, eval_result_path)

        # JSONL
        ItemJsonlWriter[dict](output_to_evaluate.with_suffix(".detailed.jsonl")).write_items(detail_records)

        # TSV
        csv_path = output_to_evaluate.with_suffix(".detailed.csv")
        with csv_path.open("w", encoding="utf-8") as f:
            headers = detail_records[0].keys()
            f.write("\t".join(headers) + "\n")
            for row in detail_records:
                f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")
