# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import zip_longest
from logging import Logger
from pathlib import Path
from typing import cast

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, log_expit
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


# TODO: add seed


class LinkingTask(Task):
    """Represents a linking benchmark task with its data files."""

    __entity_fragment: Path
    __ground_truth: Path | None

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.Linking

    @property
    def data_entity_fragment_pairs(self) -> Path:
        """Return path to entity fragment pairs."""
        return self.__entity_fragment

    @property
    def ground_truth(self) -> Path | None:
        """Return path to ground truth boolean."""
        return self.__ground_truth

    def __init__(
        self,
        name: str,
        entity_fragment_pairs: Path,
        schema: Path | None = None,
        ground_truth: Path | None = None,
        data: Path | None = None,
    ):
        """Initialize a linking task."""
        super().__init__(name, schema, data=data)
        self.__entity_fragment = resolve_path(entity_fragment_pairs, data)
        if ground_truth is not None:
            self.__ground_truth = resolve_path(ground_truth, data)

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
            ItemJsonlReader[bool](self.ground_truth, converter=bool).read_items()
            if self.ground_truth is not None
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
        adjust_to_test_prior: bool = True,
        debugging_info_path: Path | None = None,
    ) -> dict[str, float]:
        """
        Evaluate an output for the linking task.

        Args:
            output_to_evaluate: Path to model output predictions (log-odds or 0/1 predictions).
            eval_result_path: Optional path to save evaluation metrics as JSON.
            logger: Optional logger for logging evaluation summaries.
            output_dir: Optional directory for saving metrics and evaluation outputs.
            adjust_to_test_prior: If True, adjust the log-odds to match the prior on the test set.
            debugging_info_path: Optional path for loading debugging information, where each line
            contains the debugging info for each prediction in the same order. If available, the
            info will be written to the spreadsheet.

        Returns:
            dict[str, float]: Evaluation metrics dictionary.
        """
        if self.ground_truth is None:
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

        # compute probabilistic metrics (and update the log-odds in-place if needed)
        metrics = self._compute_probabilistic_metrics(log_odds, gt_labels, adjust_to_test_prior)

        # compute binary metrics
        predictions = log_odds > 0
        conf_matrix = _ConfMatrix.from_arrays(gt_labels, predictions)
        metrics = {**metrics, **conf_matrix.get_metrics()}

        # no need for optimistic log-prob if log_prob is available
        if not math.isnan(metrics["log_prob"]):
            metrics["optimistic_log_prob"] = float("nan")

        # build detailed records
        detail_records = self._build_detail_records(pairs, gt_labels, log_odds, predictions, debugging_info_path)
        self._write_side_outputs(metrics, detail_records, output_dir, eval_result_path)

        # report metrics in console or logger
        self._report_metrics(logger, metrics)

        return metrics

    def _compute_log_odds_adjustment(self, log_odds: np.ndarray, gt_labels: np.ndarray) -> float:
        """Compute the prior train->test log odds adjustment based on the ground truth labels."""
        log_odds = np.asarray(log_odds, dtype=float)
        gt_labels = np.asarray(gt_labels, dtype=float)

        # ML estimate of the constant log-odds shift:
        # d/dc L(l_odds + c) = 0
        # => sum(y_i) - exp(log_odds + c) = 0
        # i.e. the observed prevalence of the positive class should match the expected prevalence
        def balance(c: float) -> float:
            """Balance function to find the constant c."""
            return gt_labels.sum() - expit(log_odds + c).sum()

        # expand the interval until the balance function changes sign
        lo, hi = -20.0, 20.0
        while balance(lo) * balance(hi) > 0:
            lo *= 2.0
            hi *= 2.0

        c = float(cast(float, brentq(balance, lo, hi)))
        return c

    def _compute_probabilistic_metrics(
        self, log_odds: np.ndarray, gt_labels: np.ndarray, adjust_to_test_prior: bool
    ) -> dict[str, float | int]:
        """Compute the probabilistic metrics based on log-odds and ground truth labels."""
        metrics = {"log_prob": float("nan"), "log_odds_adjustment": float("nan")}

        # determine if the predictions are log-odds or binary
        vals = np.unique(log_odds)
        probabilistic = not (vals.size == 2 and np.allclose(np.sort(vals), [0.0, 1.0]))  # noqa: PLR2004

        if not probabilistic:
            return metrics

        if adjust_to_test_prior:
            log_odds_adj = self._compute_log_odds_adjustment(log_odds, gt_labels)
            log_odds += log_odds_adj
            metrics["log_odds_adjustment"] = log_odds_adj

        log_prob = float(np.sum(np.where(gt_labels, log_expit(log_odds), log_expit(-log_odds))))
        log_prob /= len(gt_labels)

        metrics["log_prob"] = log_prob

        return metrics

    def _report_metrics(self, logger: Logger | None, metrics: dict[str, float]):
        """Report evaluation metrics in a structured format."""
        order = (
            "log_prob",
            "optimistic_log_prob",
            "precision",
            "recall",
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
            "ppv",
            "npv",
            "total",
            "log_odds_adjustment",
        )

        reported_metrics = {key: metrics[key] for key in order if key in metrics}

        if logger:
            logger.info("Metrics:")
            for k, v in reported_metrics.items():
                logger.info(f"{k}: {v}")

    def _build_detail_records(
        self,
        pairs: Iterable[tuple[Entity, Entity]],
        gt_labels: np.ndarray,
        log_odds: np.ndarray,
        predictions: np.ndarray,
        debugging_info_path: Path | None = None,
    ) -> list[dict]:
        """Create a per-sample dictionary describing the prediction outcome."""
        if debugging_info_path:
            debugging_infos = ItemJsonlReader[str](debugging_info_path, converter=str).read_items()
        else:
            debugging_infos = []

        records: list[dict] = []
        for (left, right), gt_label, score, prediction, debugging_info in zip_longest(
            pairs,
            gt_labels,
            log_odds,
            predictions,
            debugging_infos,
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
                    "predicted_label": prediction,
                    "left_entity_id": left.entity_id,
                    "right_entity_id": right.entity_id,
                    "debugging_info": debugging_info if debugging_info else "",
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

    def optimistic_log_prob(self) -> float:
        """Compute the optimistic log-probability based on the confusion matrix."""
        ppv = self.precision()
        npv = self.npv()

        if ppv is None or npv is None:
            return float("nan")

        log_prob = 0

        if self.tp > 0:
            log_prob += self.tp * math.log(ppv)

        if self.fp > 0:
            log_prob += self.fp * math.log(1 - ppv)

        if self.tn > 0:
            log_prob += self.tn * math.log(npv)

        if self.fn > 0:
            log_prob += self.fn * math.log(1 - npv)

        return log_prob / self.total

    def get_metrics(self) -> dict[str, float]:
        """Get all metrics as a dictionary."""
        return {
            "true_positive": self.tp,
            "true_negative": self.tn,
            "false_positive": self.fp,
            "false_negative": self.fn,
            "total": self.total,
            "precision": self.precision(),
            "recall": self.recall(),
            "npv": self.npv(),
            "ppv": self.ppv(),
            "optimistic_log_prob": self.optimistic_log_prob(),
        }

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

        return _ConfMatrix(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))
