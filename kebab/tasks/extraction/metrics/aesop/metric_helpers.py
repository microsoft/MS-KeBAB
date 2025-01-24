# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from kebab.contracts.entity import Entity, Property, PropertySchema


# from kebab.tasks.extraction.metrics.aesop.distances import PropertyScoreValue


@dataclass
class MatchedPairInfo:
    """
    Stores information about matched pairs of entities.

    Attributes:
        left_ind: the indices of the matched entities in the ground truth
        right_ind: the indices of the matched entities in the predictions
        left_unmatched: the indices of the unmatched entities in the ground truth
        right_unmatched: the indices of the unmatched entities in the predictions
        distances: the distances between the matched entities

    """

    left_ind: list[int]
    right_ind: list[int]
    left_unmatched: list[int]
    right_unmatched: list[int]
    distances: np.ndarray


class EntityMatcher:
    """Matches entities in two disjoint sets based on a distance function and a matching score threshold."""

    def __init__(self, entities_gt: list[Entity], entities_pred: list[Entity]) -> None:
        """Initialize the EntityMatcher."""
        self.entities_gt = entities_gt
        self.entities_pred = entities_pred

    def match(self, distance_function: Callable[[Entity, Entity], float], threshold: float = 1.0) -> MatchedPairInfo:
        """Match entities based on a distance function and a matching score threshold."""
        distances = self._compute_distances(distance_function)
        print(distances)
        matched_indices = match_items(distances, threshold)
        return MatchedPairInfo(
            matched_indices.left_ind,
            matched_indices.right_ind,
            matched_indices.left_unmatched,
            matched_indices.right_unmatched,
            distances,
        )

    def _compute_distances(self, distance_function: Callable[[Entity, Entity], float]) -> np.ndarray:
        distances = np.empty((len(self.entities_gt), len(self.entities_pred)))
        for i, e1 in enumerate(self.entities_gt):
            for j, e2 in enumerate(self.entities_pred):
                distances[i, j] = distance_function(e1, e2)
        return distances


class EvaluatedPropertyMetrics:
    """
    Stores intermediate metrics for a single property key.

    Attributes:
        relevant_pair_scores: the scores for the property in the matched entities where each element is a list of scores for matched values
        property_value_count_gt: the number of values for the property in the ground truth
        unmatched_count: the number of unmatched values for the property in the predictions
        zero_out_recall: whether to zero out the recall for the property for cases when the prediction is missing the property
        zero_out_prec: whether to zero out the precision for the property for cases when the ground truth is missing the property

    """

    def __init__(self, gt_length: int) -> None:
        """Initialize EvaluatedPropertyMetrics."""
        self.relevant_pair_scores = defaultdict(lambda: defaultdict(lambda: [0.0]))
        self.property_value_count_gt = [0 for _ in range(gt_length)]
        self.unmatched_count = 0

    def __repr__(self: EvaluatedPropertyMetrics) -> str:
        """String representation of EvaluatedPropertyMetrics."""
        return (
            f"EvaluatedPropertyMetrics(\n"
            f"relevant_pair_scores={self.relevant_pair_scores},\n"
            f"property_value_count_gt={self.property_value_count_gt},\n"
            f"unmatched_count={self.unmatched_count})"
        )


@dataclass
class MetricsAccumulator:
    """Accumulates scores for metrics computation across all files."""

    def __init__(self) -> None:
        """Initialize MetricsAccumulator."""
        self.total_scores_per_doc: dict = defaultdict(list)
        self.total_unmatched_counts_per_doc: dict = defaultdict(int)
        self.total_gt_property_value_counts_per_doc: dict = defaultdict(list)
        self.total_gt_entities: int = 0
        self.total_pred_entities: int = 0
        self.matched_count: int = 0
        self.unmatched_extra_gt_count: int = 0
        self.unmatched_extra_pred_count: int = 0
        self.unmatched_pair_count: int = 0

    def update(self, other: MetricsAccumulator) -> None:
        """Update the metrics accumulator with the metrics from another accumulator."""
        self.matched_count += other.matched_count
        self.unmatched_extra_gt_count += other.unmatched_extra_gt_count
        self.unmatched_extra_pred_count += other.unmatched_extra_pred_count
        self.unmatched_pair_count += other.unmatched_pair_count
        self.total_gt_entities += other.total_gt_entities
        self.total_pred_entities += other.total_pred_entities
        for key in other.total_scores_per_doc:
            self.total_scores_per_doc[key].extend(other.total_scores_per_doc[key])
            self.total_unmatched_counts_per_doc[key] += other.total_unmatched_counts_per_doc[key]
            self.total_gt_property_value_counts_per_doc[key].extend(other.total_gt_property_value_counts_per_doc[key])

    def accumulate_metrics(self) -> dict:
        """Accumulate metrics across all files."""
        metrics = {
            "property_precision": {},
            "property_recall": {},
        }
        for key, score_values in self.total_scores_per_doc.items():
            scores = np.hstack(score_values) if len(score_values) > 0 else np.array([])
            total_score = np.sum(scores)
            precision_denominator = len(scores) + np.sum(self.total_unmatched_counts_per_doc[key])
            recall_denominator = np.sum(self.total_gt_property_value_counts_per_doc[key])
            metrics["property_precision"][key] = total_score / precision_denominator if precision_denominator > 0 else 0
            metrics["property_recall"][key] = total_score / recall_denominator if recall_denominator > 0 else None
            if len(scores) == 0 and self.total_unmatched_counts_per_doc[key] == 0:
                metrics["property_precision"][key] = None
        metrics["matched_count"] = self.matched_count  # type: ignore
        metrics["unmatched_counts"] = {
            "extra_gt_entities": self.unmatched_extra_gt_count,
            "extra_pred_entities": self.unmatched_extra_pred_count,
            "pairs": self.unmatched_pair_count,
        }
        metrics["unmatched_fractions"] = {
            "pairs": self.unmatched_pair_count / (self.unmatched_pair_count + self.matched_count),
            "extra_pred_entities": self.unmatched_extra_pred_count / self.total_pred_entities,
            "extra_gt_entities": self.unmatched_extra_gt_count / self.total_gt_entities,
        }

        return metrics

    def __repr__(self: MetricsAccumulator) -> str:
        """String representation of MetricsAccumulator."""
        return (
            f"MetricsAccumulator(\n"
            f"total_scores_per_doc={self.total_scores_per_doc},\n"
            f"total_unmatched_counts_per_doc={self.total_unmatched_counts_per_doc},\n"
            f"total_gt_property_value_counts_per_doc={self.total_gt_property_value_counts_per_doc},\n"
            f"total_gt_entities={self.total_gt_entities},\n"
            f"total_pred_entities={self.total_pred_entities},\n"
            f"matched_count={self.matched_count},\n"
            f"unmatched_extra_gt_count={self.unmatched_extra_gt_count},\n"
            f"unmatched_extra_pred_count={self.unmatched_extra_pred_count},\n"
            f"unmatched_pair_count={self.unmatched_pair_count})"
        )


@dataclass
class MatchedIndices:
    """Stores the indices of matched items and unmatched items in ground truth and prediction."""

    left_ind: list[int]
    right_ind: list[int]
    left_unmatched: list[int]
    right_unmatched: list[int]


class MatchedEntitiesScorer:
    """Computes scores for a given property for matched entities."""

    def __init__(
        self,
        entities_gt: list[Entity],
        entities_pred: list[Entity],
        left_ind: list[int],
        right_ind: list[int],
    ) -> None:
        """Initialize the MatchedEntitiesScorer."""
        self.entities_gt = entities_gt
        self.entities_pred = entities_pred

        if len(left_ind) != len(right_ind):
            raise ValueError("left_ind and right_ind must have the same length.")

        self.gt_ind = left_ind
        self.pred_ind = right_ind

    def score(
        self, score_function: Callable[[Entity, Entity, Property], PropertyScoreValue], property_: Property
    ) -> EvaluatedPropertyMetrics:
        """Compute scores for a given property for matched entities."""
        evaluated_property_metrics = EvaluatedPropertyMetrics(len(self.entities_gt))

        for gt_idx, pred_idx in zip(self.gt_ind, self.pred_ind, strict=True):
            if property_ in self.entities_gt[gt_idx]:
                if property_ in self.entities_pred[pred_idx]:
                    scores, unmatched_pred_count = score_function(
                        self.entities_gt[gt_idx], self.entities_pred[pred_idx], property_
                    )
                    evaluated_property_metrics.relevant_pair_scores[gt_idx][pred_idx] = scores
                    evaluated_property_metrics.unmatched_count += unmatched_pred_count
                else:
                    evaluated_property_metrics.relevant_pair_scores[gt_idx][pred_idx] = []
                evaluated_property_metrics.property_value_count_gt[gt_idx] += len(self.entities_gt[gt_idx][property_])
        return evaluated_property_metrics


class MetricsComputer:
    """Computes metrics for a given set of ground truth and prediction entities."""

    def __init__(
        self,
        ground_truth: list[Entity],
        predictions: list[Entity],
        property_schema: PropertySchema,
    ) -> None:
        """Initialize the MetricsComputer."""
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.property_schema = property_schema

    def compute_bipartite_metrics(
        self,
        matched_pair: MatchedPairInfo,
        keys_to_score: dict[str, Callable[[Entity, Entity, Property], tuple[list[float], int]]],
    ) -> MetricsAccumulator:
        """Compute provided property scores for the matched entities."""
        metrics_accumulator = MetricsAccumulator()
        metrics_accumulator.matched_count = len(matched_pair.left_ind)
        metrics_accumulator.unmatched_pair_count = len(matched_pair.left_unmatched)
        metrics_accumulator.total_gt_entities = len(self.ground_truth)
        metrics_accumulator.total_pred_entities = len(self.predictions)

        diff = len(self.ground_truth) - len(self.predictions)
        if diff > 0:
            metrics_accumulator.unmatched_extra_gt_count = diff
        elif diff < 0:
            metrics_accumulator.unmatched_extra_pred_count = -diff

        properties_union = compute_properties_union(self.ground_truth, self.predictions, matched_pair)
        entities_scorer = MatchedEntitiesScorer(
            self.ground_truth, self.predictions, matched_pair.left_ind, matched_pair.right_ind
        )

        for key in properties_union:
            score_func = keys_to_score[key]

            evaluated_property_metrics = entities_scorer.score(score_func, self.property_schema.properties[key])

            metrics_accumulator.total_scores_per_doc[key].extend(
                [
                    evaluated_property_metrics.relevant_pair_scores[gt_idx][pred_idx]
                    for gt_idx in evaluated_property_metrics.relevant_pair_scores
                    for pred_idx in evaluated_property_metrics.relevant_pair_scores[gt_idx]
                ]
                or [[0.0]]
            )
            metrics_accumulator.total_unmatched_counts_per_doc[key] += evaluated_property_metrics.unmatched_count
            metrics_accumulator.total_gt_property_value_counts_per_doc[key] += (
                evaluated_property_metrics.property_value_count_gt
            )

        return metrics_accumulator


def match_items(distances: np.ndarray, threshold: float = 1.0) -> MatchedIndices:
    """
    Match items from two sets of items based on a distance matrix if the distance is within a threshold.
    Returns the indices of the matched items.
    """
    epsilon = 0.01
    left_ind, right_ind = min_weight_full_bipartite_matching(csr_matrix(distances + epsilon))

    matched_dists = distances[left_ind, right_ind]
    indices_within_threshold = np.where(matched_dists <= threshold)
    unmatched_ind = np.where(matched_dists > threshold)

    left_ind_matched = left_ind[indices_within_threshold]
    right_ind_matched = right_ind[indices_within_threshold]

    left_ind_unmatched = left_ind[unmatched_ind]
    right_ind_unmatched = right_ind[unmatched_ind]

    return MatchedIndices(left_ind_matched, right_ind_matched, left_ind_unmatched, right_ind_unmatched)


def compute_properties_union(
    ground_truth: list[Entity],
    predictions: list[Entity],
    matched_pair: MatchedPairInfo,
) -> set[str]:
    """Compute the union of the properties of the ground truth and the predictions ignoring properties only present in unmatched entities."""
    ground_truth_keys = {
        key for idx, entity in enumerate(ground_truth) if idx in matched_pair.left_ind for key in entity.properties
    }
    predictions_keys = {
        key for idx, entity in enumerate(predictions) if idx in matched_pair.right_ind for key in entity.properties
    }
    return ground_truth_keys.union(predictions_keys)
