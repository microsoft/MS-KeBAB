# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import zip_longest
from pathlib import Path

import numpy as np

from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskInstance
from kebab.utils.io_helpers import EntityJsonlReader, ItemJsonlReader, ItemJsonlWriter, save_dict_to_json


class ClusteringTaskInstance(TaskInstance):
    """Represents a clustering benchmark task instance with its data files."""

    __data_entity_fragments: Path
    __data_ground_truth_labels: Path | None

    @property
    def data_entity_fragments(self) -> Path:
        """Return the path to the file containing entity fragments."""
        return self.__data_entity_fragments

    @property
    def data_ground_truth_labels(self) -> Path | None:
        """Return the path to the ground truth labels representing cluster IDs."""
        return self.__data_ground_truth_labels

    def __init__(
        self,
        name: str,
        task: Task,
        entity_fragments: str | Path,
        schema: str,
        ground_truth_labels: str | Path | None = None,
    ):
        """Initialize a new clustering task instance."""
        super().__init__(name, task, schema)
        self.__data_entity_fragments = Path(entity_fragments)
        if ground_truth_labels is not None:
            self.__data_ground_truth_labels = Path(ground_truth_labels)

    def read_items(self) -> Iterable[tuple[Entity, str | None]]:
        """
        Read entity fragments with optional ground-truth labels.

        Returns:
            Iterable[tuple[Entity, str | None]]: An iterable of tuples, where each
            tuple contains:
                - An `Entity` object describing an entity fragment.
                - A string value providing the ground-truth label (cluster ID).
        """
        entities = EntityJsonlReader(self.data_entity_fragments).read_items()
        labels = (
            ItemJsonlReader[str](self.data_ground_truth_labels).read_items()
            if self.data_ground_truth_labels is not None
            else iter([])
        )
        return zip_longest(entities, labels)

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
        output_to_evaluate: Path,
        eval_result_path: Path | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the clustering task instance."""
        if self.data_ground_truth_labels is None:
            raise ValueError("Ground truth data is required for evaluation.")

        predictions = list(ItemJsonlReader[str](output_to_evaluate).read_items())
        ground_truth = list(ItemJsonlReader[str](self.data_ground_truth_labels).read_items())

        fragment_count = len(predictions)
        metrics = defaultdict(float)
        metrics["fragments"] = fragment_count

        # construct the predicted {element_idx -> set of element_idx} map
        pred_clusters = defaultdict(set)
        pred_cluster_map = {}
        for i, cluster_id in enumerate(predictions):
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

        # compute BCubed P,R and F1
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

        metrics["precision"] = np.mean(precisions, dtype=np.float64)
        metrics["recall"] = np.mean(recalls, dtype=np.float64)
        metrics["f1"] = np.mean(f1s, dtype=np.float64)

        metrics = dict(metrics)

        if eval_result_path:
            save_dict_to_json(metrics, eval_result_path)

        return metrics
