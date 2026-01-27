# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from logging import Logger
from pathlib import Path

import numpy as np

from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.tasks.metrics.extraction.aesop.metric_helpers import safe_division
from kebab.utils.io_helpers import EntityJsonlReader, ItemJsonlReader, ItemJsonlWriter, resolve_path, save_dict_to_json


class ClusteringTask(Task):
    """Represents a clustering benchmark task with its data files."""

    __entity_fragments: Path
    __ground_truth: Path | None

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.Clustering

    @property
    def entity_fragments(self) -> Path:
        """Return the path to the file containing entity fragments."""
        return self.__entity_fragments

    @property
    def ground_truth(self) -> Path | None:
        """Return the path to the ground truth labels representing cluster IDs."""
        return self.__ground_truth

    def __init__(
        self,
        name: str,
        entity_fragments: Path,
        schema: Path | None = None,
        ground_truth: Path | None = None,
        root_for_relative_paths: Path | None = None,
    ):
        """Initialize a new clustering task."""
        super().__init__(name, schema, root_for_relative_paths=root_for_relative_paths)
        self.__entity_fragments = resolve_path(entity_fragments, root_for_relative_paths)
        if ground_truth is not None:
            self.__ground_truth = resolve_path(ground_truth, root_for_relative_paths)

    def read_items(self) -> Iterable[tuple[Entity, str | None]]:
        """
        Read entity fragments with optional ground-truth labels.

        Returns:
            Iterable[tuple[Entity, str | None]]: An iterable of tuples, where each
            tuple contains:
                - An `Entity` object describing an entity fragment.
                - A string value providing the ground-truth label (cluster ID).
        """
        entities = EntityJsonlReader(self.entity_fragments).read_items()
        labels = (
            ItemJsonlReader[str](self.ground_truth, converter=str).read_items()
            if self.ground_truth is not None
            else iter([])
        )
        return zip(entities, labels, strict=True)

    def write_items(self, path: Path, items: Iterable[str]) -> None:
        """
        Write items, i.e. str labels, to the specified path.

        Args:
            path: The file path where the str labels should be written.
            items: An iterable of str labels to be written to the file.
        """
        ItemJsonlWriter[str](path).write_items(items)

    def evaluate(
        self,
        predictions: Path,
        result_output_path: Path | None = None,
        logger: Logger | None = None,  # noqa: ARG002
    ) -> dict[str, float]:
        """Evaluate an output for the clustering task."""
        if self.ground_truth is None:
            raise ValueError("Ground truth data is required for evaluation.")

        predicted_vals = list(ItemJsonlReader[str](predictions, converter=str).read_items())
        ground_truth = list(ItemJsonlReader[str](self.ground_truth).read_items())

        fragment_count = len(predicted_vals)
        metrics = {}
        metrics["fragments"] = fragment_count

        # construct the predicted {element_idx -> set of element_idx} map
        pred_clusters = defaultdict(set)
        pred_cluster_map = {}
        for i, cluster_id in enumerate(predicted_vals):
            cluster = pred_clusters[cluster_id]
            cluster.add(i)
            pred_cluster_map[i] = cluster

        metrics["predicted_clusters"] = len(pred_clusters)

        # construct the ground truth {element_idx -> set of element_idx} map
        gt_clusters = defaultdict(set)
        gt_cluster_map = {}
        for i, cluster_id in enumerate(ground_truth):
            cluster = gt_clusters[cluster_id]
            cluster.add(i)
            gt_cluster_map[i] = cluster

        metrics["ground_truth_clusters"] = len(gt_clusters)

        # Compute BCubed P,R and F1
        # See "Entity-Based Cross-Document Coreferencing Using the Vector Space Model" https://aclanthology.org/P98-1012.pdf
        precisions = []
        recalls = []
        f1s = []
        for i in range(fragment_count):
            tp = len(pred_cluster_map[i].intersection(gt_cluster_map[i]))
            fp = len(pred_cluster_map[i].difference(gt_cluster_map[i]))
            fn = len(gt_cluster_map[i].difference(pred_cluster_map[i]))

            precision = tp / len(pred_cluster_map[i])
            recall = tp / len(gt_cluster_map[i])
            f1 = tp / (tp + 0.5 * (fp + fn))

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        metrics["precision"] = float(np.mean(precisions, dtype=np.float64))
        metrics["recall"] = float(np.mean(recalls, dtype=np.float64))
        metrics["f1"] = float(np.mean(f1s, dtype=np.float64))

        # Compute MUC precision, recall, F1
        # See "A Model-Theoretic Coreference Scoring Scheme" https://aclanthology.org/M95-1005.pdf
        cluster_precision_numerator = 0
        cluster_precision_denominator = 0
        for cluster in pred_clusters.values():
            ground_truth_clusters_intersecting = {ground_truth[element_idx] for element_idx in cluster}
            cluster_precision_numerator += len(cluster) - len(ground_truth_clusters_intersecting)
            cluster_precision_denominator += len(cluster) - 1
        muc_precision = (
            cluster_precision_numerator / cluster_precision_denominator if cluster_precision_denominator > 0 else 1.0
        )
        metrics["muc_precision"] = muc_precision
        cluster_recall_numerator = 0
        cluster_recall_denominator = 0
        for cluster in gt_clusters.values():
            predicted_clusters_intersecting = {predicted_vals[element_idx] for element_idx in cluster}
            cluster_recall_numerator += len(cluster) - len(predicted_clusters_intersecting)
            cluster_recall_denominator += len(cluster) - 1
        muc_recall = cluster_recall_numerator / cluster_recall_denominator if cluster_recall_denominator > 0 else 1.0
        metrics["muc_recall"] = muc_recall
        metrics["muc_f1"] = safe_division(2 * muc_recall * muc_precision, muc_recall + muc_precision)

        metrics = dict(metrics)

        if result_output_path:
            save_dict_to_json(metrics, result_output_path)

        return metrics
