# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# ruff: noqa: D107, FA102, D102, E721, D101, C401
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


PropertyValue = str | list[str]
Entity = dict[str, PropertyValue]


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
        self.entities_gt = entities_gt
        self.entities_pred = entities_pred

    def match(
        self, distance_function: Callable[[str | Any, str | Any], float], threshold: float = 1.0
    ) -> MatchedPairInfo:
        distances = self._compute_distances(distance_function)
        matched_indices = match_items(distances, threshold)
        return MatchedPairInfo(
            matched_indices.left_ind,
            matched_indices.right_ind,
            matched_indices.left_unmatched,
            matched_indices.right_unmatched,
            distances,
        )

    def _compute_distances(self, distance_function: Callable[[str | Any, str | Any], float]) -> np.ndarray:
        distances = np.empty((len(self.entities_gt), len(self.entities_pred)))
        for i, e1 in enumerate(self.entities_gt):
            for j, e2 in enumerate(self.entities_pred):
                distances[i, j] = distance_function(e1, e2)
        return distances


class EvaluatedPropertyMetrics:
    """
    Stores intermediate metrics for a single property key.

    Attributes:
        score_lists: the scores for the property in the matched entities where each element is a list of scores for matched values
        property_value_count: the number of values for the property in the ground truth
        unmatched_count: the number of unmatched values for the property in the predictions
        zero_out_recall: whether to zero out the recall for the property for cases when the prediction is missing the property
        zero_out_prec: whether to zero out the precision for the property for cases when the ground truth is missing the property

    """

    def __init__(self, length: int) -> None:
        self.score_lists = []
        self.property_value_count = [0 for _ in range(length)]
        self.unmatched_count = 0
        self.zero_out_recall = [False] * length
        self.zero_out_prec = [False] * length


@dataclass
class MetricsAccumulator:
    """Accumulates scores for metrics computation across all files."""

    def __init__(self) -> None:
        self.total_scores_per_doc: dict = defaultdict(list)
        self.total_unmatched_counts_per_doc: dict = defaultdict(int)
        self.total_property_value_counts_per_doc: dict = defaultdict(list)
        self.total_zero_out_recall: dict = defaultdict(list)
        self.total_zero_out_precision: dict = defaultdict(list)
        self.total_gt_entities: int = 0
        self.total_pred_entities: int = 0
        self.matched_count: int = 0
        self.unmatched_extra_gt_count: int = 0
        self.unmatched_extra_pred_count: int = 0
        self.unmatched_pair_count: int = 0

    def update(self, other: "MetricsAccumulator") -> None:
        self.matched_count += other.matched_count
        self.unmatched_extra_gt_count += other.unmatched_extra_gt_count
        self.unmatched_extra_pred_count += other.unmatched_extra_pred_count
        self.unmatched_pair_count += other.unmatched_pair_count
        self.total_gt_entities += other.total_gt_entities
        self.total_pred_entities += other.total_pred_entities
        for key in other.total_scores_per_doc:
            self.total_scores_per_doc[key].extend(other.total_scores_per_doc[key])
            self.total_unmatched_counts_per_doc[key] += other.total_unmatched_counts_per_doc[key]
            self.total_property_value_counts_per_doc[key].extend(other.total_property_value_counts_per_doc[key])
            self.total_zero_out_recall[key].extend(other.total_zero_out_recall[key])
            self.total_zero_out_precision[key].extend(other.total_zero_out_precision[key])


@dataclass
class MatchedIndices:
    left_ind: list[int]
    right_ind: list[int]
    left_unmatched: list[int]
    right_unmatched: list[int]


class MatchedEntitiesScorer:
    """Computes scores for a given property key for matched entities."""

    def __init__(
        self,
        entities_gt: list[Entity],
        entities_pred: list[Entity],
        left_ind: list[int],
        right_ind: list[int],
    ) -> None:
        self.entities_gt = entities_gt
        self.entities_pred = entities_pred

        if len(left_ind) != len(right_ind):
            raise ValueError("left_ind and right_ind must have the same length.")

        self.left_ind = left_ind
        self.right_ind = right_ind

    def score(
        self, score_function: Callable[[str | Any, str | Any, str], list[float]], key: str
    ) -> EvaluatedPropertyMetrics:
        """Compute scores for a given property key for matched entities."""
        evaluated_property_metrics = EvaluatedPropertyMetrics(len(self.left_ind))

        for i, j, count in zip(self.left_ind, self.right_ind, range(len(self.left_ind))):
            if key not in self.entities_gt[i]:
                evaluated_property_metrics.zero_out_prec[count] = True
            else:
                evaluated_property_metrics.score_lists.append(
                    score_function(self.entities_gt[i], self.entities_pred[j], key)
                )
                evaluated_property_metrics.property_value_count[count] += (
                    len(self.entities_gt[i][key]) if type(self.entities_gt[i][key]) is list else 1
                )

                if key not in self.entities_pred[j]:
                    evaluated_property_metrics.zero_out_recall[count] = True
                    continue

                values_are_in_lists = (
                    type(self.entities_gt[i][key]) is list and type(self.entities_pred[j][key]) is list
                )
                if not values_are_in_lists:
                    continue

                # Only count unmatched values if predictions have unmatched values
                if len(self.entities_pred[j][key]) > len(self.entities_gt[i][key]):
                    evaluated_property_metrics.unmatched_count += len(self.entities_pred[j][key]) - len(
                        self.entities_gt[i][key]
                    )

        return evaluated_property_metrics


class MetricsComputer:
    def __init__(
        self,
        ground_truth: list[Entity],
        predictions: list[Entity],
    ) -> None:
        self.ground_truth = ground_truth
        self.predictions = predictions

    def compute_bipartite_metrics(
        self,
        matched_pair: MatchedPairInfo,
        keys_to_score: dict[str, Callable[[str | Any, str | Any, str], list[float]]],
    ) -> MetricsAccumulator:
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

            evaluated_property_metrics = entities_scorer.score(score_func, key)

            metrics_accumulator.total_scores_per_doc[key].extend(
                evaluated_property_metrics.score_lists if evaluated_property_metrics.score_lists else [[0]]
            )
            metrics_accumulator.total_unmatched_counts_per_doc[key] += evaluated_property_metrics.unmatched_count
            metrics_accumulator.total_property_value_counts_per_doc[key] += (
                evaluated_property_metrics.property_value_count
            )
            metrics_accumulator.total_zero_out_recall[key] += evaluated_property_metrics.zero_out_recall
            metrics_accumulator.total_zero_out_precision[key] += evaluated_property_metrics.zero_out_prec

        return metrics_accumulator


def match_items(distances: np.ndarray, threshold: float = 1.0) -> MatchedIndices:
    """
    Match items from two sets of items based on a distance matrix if the distance is within a threshold.
    Returns the indices of the matched items.
    """
    left_ind, right_ind = min_weight_full_bipartite_matching(csr_matrix(distances + 0.01))

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
    unmatched_gt = [ground_truth[i] for i in range(len(ground_truth)) if i not in matched_pair.left_ind]
    unmatched_pred = [predictions[i] for i in range(len(predictions)) if i not in matched_pair.right_ind]
    ground_truth_keys = set(key for entity in ground_truth if entity not in unmatched_gt for key in entity)
    predictions_keys = set(key for entity in predictions if entity not in unmatched_pred for key in entity)
    return ground_truth_keys.union(predictions_keys)
