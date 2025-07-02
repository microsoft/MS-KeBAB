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

import numpy as np
from sklearn.metrics import confusion_matrix

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
        calibrate: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate an output for the linking task.

        Args:
            output_to_evaluate: Path to model output predictions.
            eval_result_path: Optional path to save evaluation metrics as JSON.
            logger: Optional logger for logging evaluation summaries.
            output_dir: Optional directory for saving metrics and evaluation outputs.
            calibrate: If True, calibrate the threshold based on ground truth data.

        Returns:
            dict[str, float]: Evaluation metrics dictionary.
        """
        if self.data_ground_truth_boolean is None:
            raise ValueError("Ground truth data is required for evaluation.")

        log_odds = np.asarray(list(ItemJsonlReader[float](output_to_evaluate, converter=float).read_items()))
        pairs, gt_labels = zip(*self.read_items(), strict=False)
        gt_labels = np.asarray(gt_labels, dtype=bool)
        if log_odds.shape != gt_labels.shape:
            raise ValueError(
                f"Log-odds shape {log_odds.shape} does not match ground truth labels shape {gt_labels.shape}."
            )

        output_dir = output_dir or Path.cwd() / "output" / self.path_safe_name
        output_dir.mkdir(parents=True, exist_ok=True)

        threshold = self._calibrate(gt_labels, log_odds) if calibrate else 0.0
        predictions = log_odds > threshold

        conf_matrix = _ConfMatrix.from_arrays(gt_labels, predictions)
        metrics = self._compute_metrics(conf_matrix, gt_labels, log_odds)

        if calibrate:
            metrics["threshold"] = threshold

        # build detailed records
        detail_records = self._build_detail_records(
            pairs,
            gt_labels,
            log_odds,
            predictions,
            threshold,
        )

        self._write_side_outputs(metrics, detail_records, output_dir, eval_result_path)

        # report metrics in console or logger
        self._report_metrics(logger, metrics)

        return metrics

    def _calibrate(self, gt_labels: np.ndarray, log_odds: np.ndarray, eps: float = 1e-15) -> float:
        """
        Determine the threshold that maximizes the empirical log-probability.

        Args:
            gt_labels: Ground truth boolean labels.
            log_odds: Array of predicted scores (log-odds).

        Returns:
            float: Optimal threshold for classification.
        """
        # start from the highest log-odds, everything is predicted negative
        order = np.argsort(log_odds)[::-1]
        gt_labels = gt_labels[order].astype(int)
        log_odds = log_odds[order]

        n = len(gt_labels)
        n_pos = gt_labels.sum()
        n_neg = n - n_pos

        tp = fp = 0
        fn, tn = n_pos, n_neg

        best_log_prob = -np.inf
        best_threshold = log_odds[0] + eps  # => all predicted as negative

        for k, (score, label) in enumerate(zip(log_odds, gt_labels, strict=True), 1):
            # classify the k-th example as positive
            if label:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1

            ppv = tp / k
            npv = tn / (n - k) if k != n else 0.0

            ppv = np.clip(ppv, eps, 1 - eps)
            npv = np.clip(npv, eps, 1 - eps)

            log_prob = tp * np.log(ppv) + fp * np.log(1 - ppv) + tn * np.log(npv) + fn * np.log(1 - npv)

            if log_prob > best_log_prob:
                best_log_prob = log_prob

                # we want this example to be predicted as positive => threshold is just below the score
                best_threshold = score - eps

        return best_threshold

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
            "threshold",
        )

        reported_metrics = {key: metrics[key] for key in order if key in metrics}

        if logger:
            logger.info(
                "\t".join(reported_metrics.keys()) + "\n" + "\t".join(f"{v:.4f}" for v in reported_metrics.values())
            )
        else:
            print("\t".join(reported_metrics.keys()) + "\n" + "\t".join(f"{v:.4f}" for v in reported_metrics.values()))

    def _compute_metrics(
        self, conf_matrix: _ConfMatrix, gt_labels: np.ndarray, log_odds: np.ndarray
    ) -> dict[str, float]:
        """Compute evaluation metrics based on the confusion matrix and log-odds."""
        # empirical log-prob
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

        # log-prob
        vals = np.unique(log_odds)
        probabilistic = not (vals.size == 2 and np.allclose(np.sort(vals), [0.0, 1.0]))

        if probabilistic:
            probs = 1.0 / (1.0 + np.exp(-log_odds))
            log_prob = float(np.sum(np.where(gt_labels, np.log(probs), np.log(1.0 - probs))))
        else:
            log_prob = float("nan")

        metrics["log_prob"] = log_prob

        return metrics

    def _build_detail_records(
        self,
        pairs: Iterable[tuple[Entity, Entity]],
        gt_labels: np.ndarray,
        log_odds: np.ndarray,
        predictions: np.ndarray,
        threshold: float,
    ) -> list[dict]:
        """Create a per-sample dictionary describing the prediction outcome."""
        records: list[dict] = []
        for (left, right), gt_label, score, prediction in zip_longest(
            pairs,
            gt_labels,
            log_odds,
            predictions,
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
                    "log_odds": score,
                    "cal_log_odds": score - threshold,
                    "predicted_label": prediction,
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

    @staticmethod
    def from_arrays(
        gt_labels: np.ndarray,
        predictions: np.ndarray,
    ) -> _ConfMatrix:
        """
        Build a confusion-matrix instance from ground-truth and prediction arrays.

        Args:
            gt_labels: Ground-truth boolean labels.
            predictions: Predicted boolean labels.

        Returns:
            _ConfMatrix: Populated confusion-matrix object.
        """
        cm = confusion_matrix(gt_labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        return _ConfMatrix(tp=tp, tn=tn, fp=fp, fn=fn)
