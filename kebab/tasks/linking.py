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
        logger: Logger | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, float]:
        """
        Evaluate an output for the linking task.

        Args:
            output_to_evaluate: Path to model output predictions.
            eval_result_path: Optional path to save evaluation metrics as JSON.
            logger: Optional logger for logging evaluation summaries.
            output_dir: Optional directory for saving metrics and evaluation outputs.

        Returns:
            dict[str, float]: Evaluation metrics dictionary.
        """
        if self.data_ground_truth_boolean is None:
            raise ValueError("Ground truth data is required for evaluation.")

        preds = self._load_predictions(output_to_evaluate)
        pairs, gt_labels = zip(*self.read_items(), strict=False)
        self._validate_lengths(preds, gt_labels)

        output_dir = output_dir or Path.cwd() / "output" / self.path_safe_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # TODO(pmyshkov): this will be determined by calibration in the future
        threshold = 0.0
        pred_labels = [p > threshold for p in preds]

        conf_matrix, log_prob = self._confusion_matrix(list(gt_labels), pred_labels, preds)
        metrics = {"log_prob": log_prob, **self._extra_metrics(conf_matrix)}

        # build detailed records
        detail_records = self._build_detail_records(
            pairs,
            list(gt_labels),
            preds,
            pred_labels,
            threshold,
        )

        self._write_side_outputs(metrics, detail_records, output_dir, eval_result_path)

        # report metrics in console or logger
        self._report_metrics(logger, metrics)

        return metrics

    def _report_metrics(self, logger: Logger | None, metrics: dict[str, float]):
        """Report evaluation metrics in a structured format."""
        order = (
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

        reported_metrics = {key: metrics[key] for key in order if key in metrics}

        if logger:
            logger.info(
                "\t".join(reported_metrics.keys()) + "\n" + "\t".join(f"{v:.4f}" for v in reported_metrics.values())
            )
        else:
            print("\t".join(reported_metrics.keys()) + "\n" + "\t".join(f"{v:.4f}" for v in reported_metrics.values()))

    def _load_predictions(self, path: Path) -> list[float]:
        """Load numeric predictions from a JSONL file."""
        return list(ItemJsonlReader[float](path, converter=float).read_items())

    def _validate_lengths(self, *streams: list | tuple) -> None:
        """Ensure all provided sequences have equal lengths."""
        lengths = {len(s) for s in streams}
        if len(lengths) > 1:
            raise ValueError("Input streams have mismatched lengths.")

    def _confusion_matrix(
        self, gt_labels: list[bool], preds: list[bool], log_odds: list[float]
    ) -> tuple[_ConfMatrix, float]:
        """Compute confusion matrix and log-probability."""
        conf_matrix = _ConfMatrix()
        log_prob = 0.0
        probabilistic = sorted(set(log_odds)) != [0, 1]

        for gt_label, pred_label, l_odds in zip(gt_labels, preds, log_odds, strict=False):
            conf_matrix.update(gt_label, pred_label)
            if probabilistic:
                log_prob += math.log(1 / (1 + math.exp(-l_odds))) if gt_label else math.log(1 / (1 + math.exp(l_odds)))

        if not probabilistic:
            log_prob = float("nan")

        return conf_matrix, log_prob

    def _extra_metrics(self, conf_matrix: _ConfMatrix) -> dict[str, float]:
        """Generate additional evaluation metrics from confusion matrix."""
        metrics: dict[str, float] = defaultdict(float)
        precision = conf_matrix.precision()
        recall = conf_matrix.recall()
        npv = conf_matrix.npv()
        ppv = conf_matrix.ppv()

        metrics.update(
            true_positive=conf_matrix.tp,
            true_negative=conf_matrix.tn,
            false_positive=conf_matrix.fp,
            false_negative=conf_matrix.fn,
            total=conf_matrix.total,
        )

        if precision is not None:
            metrics["precision"] = precision
            metrics["ppv"] = ppv

        if recall is not None:
            metrics["recall"] = recall

        if npv is not None:
            metrics["npv"] = npv

        if ppv and npv and 0 < ppv < 1 and 0 < npv < 1:
            log_prob = conf_matrix.tp * math.log(ppv) + conf_matrix.fp * math.log(1 - ppv)
            log_prob += conf_matrix.tn * math.log(npv) + conf_matrix.fn * math.log(1 - npv)
            metrics["empirical_log_prob"] = log_prob

        return metrics

    def _build_detail_records(
        self,
        pairs: Iterable[tuple[Entity, Entity]],
        gt_labels: list[bool],
        log_odds: list[float],
        pred_labels: list[bool],
        threshold: float,
    ) -> list[dict]:
        """Create a per-sample dictionary describing the prediction outcome."""
        records: list[dict] = []
        for (left, right), gt_label, l_odds, pred_label in zip_longest(
            pairs,
            gt_labels,
            log_odds,
            pred_labels,
        ):
            ent_type: set[str] = set(left.metadata.get("type", [])) | set(right.metadata.get("type", []))

            left_props = set(left.properties.keys())
            right_props = set(right.properties.keys())
            overlap_props = sorted(left_props & right_props)

            prop_pattern = tuple(sorted([tuple(sorted(left.properties)), tuple(sorted(right.properties))]))

            name_overlap = (
                len(
                    {name.lower() for name in left.properties.get("name", [])}
                    & {name.lower() for name in right.properties.get("name", [])}
                )
                >= 1
            )

            records.append(
                {
                    "left": dict(left.properties),
                    "right": dict(right.properties),
                    "entity_type": sorted(ent_type),
                    "overlap_props": overlap_props,
                    "prop_overlap_num": len(overlap_props),
                    "prop_pattern": prop_pattern,
                    "name_overlap": name_overlap,
                    "label": gt_label,
                    "cal_log_odds": l_odds - threshold,
                    "predicted_label": pred_label,
                    "left_entity_id": left.entity_id,
                    "right_entity_id": right.entity_id,
                }
            )

        return records

    def _write_side_outputs(
        self,
        metrics: dict[str, float],
        detail_records: list[dict],
        output_dir: Path,
        eval_result_path: Path | None,
    ) -> None:
        """Save evaluation metrics and detailed prediction records."""
        if eval_result_path:
            save_dict_to_json(metrics, eval_result_path)

        tsv_path = output_dir / "linking_predictions.tsv"
        with tsv_path.open("w", encoding="utf-8") as f:
            headers = detail_records[0].keys()
            f.write("\t".join(headers) + "\n")
            for row in detail_records:
                f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")


@dataclass
class _ConfMatrix:
    """Confusion matrix for binary classification."""

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, gt: bool, pred: bool) -> None:
        """Update confusion matrix counts based on ground truth and prediction."""
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
        """Return total number of samples (sum of TP, TN, FP, FN)."""
        return self.tp + self.tn + self.fp + self.fn

    def precision(self) -> float:
        """Compute precision (positive predictive value).

        Returns:
            float: Precision if (tp+fp)>0, else None.
        """
        denom = self.tp + self.fp
        return self.tp / denom if denom else float("nan")

    def recall(self) -> float:
        """Compute recall (sensitivity).

        Returns:
            float: Recall if (tp+fn)>0, else None.
        """
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")

    def npv(self) -> float:
        """Compute negative predictive value (P(negative | predicted negative)).

        Returns:
            float: NPV if (tn+fn)>0, else None.
        """
        denom = self.tn + self.fn
        return self.tn / denom if denom else float("nan")

    def ppv(self) -> float:
        """Compute positive predictive value (P(positive | predicted positive)).

        Returns:
            float: PPV if (tp+fp)>0, else None.
        """
        return self.precision()
