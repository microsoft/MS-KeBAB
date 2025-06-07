# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from kebab.contracts.entity import Entity, Property, PropertySchema
from kebab.tasks.metrics.extraction.aesop.distances import ValueMatchingRecordWihScores, match_items


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


def safe_division(a: float, b: float, default: float | None = 0.0) -> float | None:
    """Safely divide two numbers."""
    return a / b if b != 0 else default


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
            metrics["property_precision"][key] = safe_division(total_score, precision_denominator)
            metrics["property_recall"][key] = safe_division(total_score, recall_denominator, None)
            if len(scores) == 0 and self.total_unmatched_counts_per_doc[key] == 0:
                metrics["property_precision"][key] = None

        # sort for convenience
        metrics["property_precision"] = dict(
            sorted(metrics["property_precision"].items(), key=lambda key: -key[1] if key[1] else 0.0)
        )
        metrics["property_recall"] = dict(
            sorted(metrics["property_recall"].items(), key=lambda key: -key[1] if key[1] else 0.0)
        )

        metrics["avg_property_precision"] = np.mean(  # type: ignore
            [value for value in metrics["property_precision"].values() if value is not None]
        )
        metrics["avg_property_recall"] = np.mean(  # type: ignore
            [value for value in metrics["property_recall"].values() if value is not None]
        )
        metrics["matched_count"] = self.matched_count  # type: ignore
        metrics["unmatched_counts"] = {
            "extra_gt_entities": self.unmatched_extra_gt_count,
            "extra_pred_entities": self.unmatched_extra_pred_count,
            "pairs": self.unmatched_pair_count,
        }
        metrics["unmatched_fractions"] = {
            "pairs": safe_division(self.unmatched_pair_count, self.unmatched_pair_count + self.matched_count),
            "extra_pred_entities": safe_division(self.unmatched_extra_pred_count, self.total_pred_entities),
            "extra_gt_entities": safe_division(self.unmatched_extra_gt_count, self.total_gt_entities),
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
class PropertyEvaluationRecord:
    """Stores property evaluation information for 2 specific entities."""

    property_id: str
    property_values_pred: list[Any]
    property_values_gt: list[Any]
    property_value_scores: list[float]
    unmatched_count: int


def get_original_property_id(entity: Entity, property_id: str) -> str:
    """
    Get the original property ID from the entity's evidence map.

    Args:
        entity: The entity containing the property.
        property_id: The property ID to look for.

    Returns:
        The original property ID if found, otherwise an empty string.
    """
    if property_id in entity.evidence_map:
        evidence_idx = entity.evidence_map[property_id][0][0]
        return entity.source_ids[evidence_idx]
    return ""


class MatchedEntitiesScorer:
    """Computes scores for a given property for matched entities."""

    def __init__(
        self,
        entities_gt: list[Entity],
        entities_pred: list[Entity],
        matched_pair: MatchedPairInfo,
        property_schema: PropertySchema,
    ) -> None:
        """Initialize the MatchedEntitiesScorer."""
        self.entities_gt = entities_gt
        self.entities_pred = entities_pred

        if len(matched_pair.left_ind) != len(matched_pair.right_ind):
            raise ValueError("left_ind and right_ind must have the same length.")

        self.gt_ind = matched_pair.left_ind
        self.pred_ind = matched_pair.right_ind
        self.distances = matched_pair.distances
        self.property_schema = property_schema
        self.debug_info: defaultdict[int, defaultdict[int, dict[str, ValueMatchingRecordWihScores]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.unmapped_property_ids: set[str] = set()

    def score(
        self,
        score_function: Callable[[Entity, Entity, Property], ValueMatchingRecordWihScores],
        property_id: str,
    ) -> EvaluatedPropertyMetrics:
        """Compute scores for a given property for matched entities."""
        evaluated_property_metrics = EvaluatedPropertyMetrics(len(self.entities_gt))

        property_ = self.property_schema.properties.get(property_id)

        if property_ is None:
            # property is not in schema, but we still need to create debug info for predictions
            for gt_idx, pred_idx in zip(self.gt_ind, self.pred_ind, strict=True):
                if property_id in self.entities_pred[pred_idx].properties:
                    pred_values = self.entities_pred[pred_idx].properties[property_id]
                    self.debug_info[gt_idx][pred_idx][property_id] = ValueMatchingRecordWihScores(
                        [], [], [], [], pred_values
                    )
                    self.unmapped_property_ids.add(property_id)
            return evaluated_property_metrics

        for gt_idx, pred_idx in zip(self.gt_ind, self.pred_ind, strict=True):
            if property_ in self.entities_gt[gt_idx]:
                if property_ in self.entities_pred[pred_idx]:
                    value_matching_record = score_function(
                        self.entities_gt[gt_idx], self.entities_pred[pred_idx], property_
                    )
                    self.debug_info[gt_idx][pred_idx][property_.property_id] = value_matching_record
                    evaluated_property_metrics.relevant_pair_scores[gt_idx][pred_idx] = (
                        value_matching_record.matched_scores
                    )
                    evaluated_property_metrics.unmatched_count += len(value_matching_record.unmatched_pred_values)
                else:
                    evaluated_property_metrics.relevant_pair_scores[gt_idx][pred_idx] = []
                    self.debug_info[gt_idx][pred_idx][property_.property_id] = ValueMatchingRecordWihScores(
                        [], [], [], self.entities_gt[gt_idx][property_], []
                    )

                evaluated_property_metrics.property_value_count_gt[gt_idx] += len(self.entities_gt[gt_idx][property_])
            elif property_ in self.entities_pred[pred_idx]:
                pred_values = self.entities_pred[pred_idx][property_]
                self.debug_info[gt_idx][pred_idx][property_.property_id] = ValueMatchingRecordWihScores(
                    [], [], [], [], pred_values
                )
        return evaluated_property_metrics

    def get_debug_info(self) -> pd.DataFrame:
        """Collect debug information into a table."""
        dfs = []
        for gt_idx, pred_dict in self.debug_info.items():
            gt_entity_id = self.entities_gt[gt_idx].entity_id
            for pred_idx, property_dict in pred_dict.items():
                pred_entity_id = self.entities_pred[pred_idx].entity_id
                records = []
                entity_match_distance = self.distances[gt_idx, pred_idx]
                for property_id, value_matching_record in property_dict.items():
                    num_gt_values = len(self.entities_gt[gt_idx].properties[property_id])
                    num_pred_values = len(self.entities_pred[pred_idx].properties[property_id])
                    first_property_record = True
                    for value_gt, value_pred, score in zip(
                        value_matching_record.gt_values,
                        value_matching_record.pred_values,
                        value_matching_record.matched_scores,
                        strict=False,
                    ):
                        record, first_property_record = create_property_record(
                            gt_entity_id,
                            pred_entity_id,
                            get_original_property_id(self.entities_pred[pred_idx], property_id),
                            property_id,
                            value_gt,
                            value_pred,
                            score,
                            num_gt_values,
                            num_pred_values,
                            first_property_record,
                        )
                        records.append(record)
                    for unmatched_gt_value in value_matching_record.unmatched_gt_values:
                        record, first_property_record = create_property_record(
                            gt_entity_id,
                            pred_entity_id,
                            "",
                            property_id,
                            unmatched_gt_value,
                            "",
                            None,
                            num_gt_values,
                            num_pred_values,
                            first_property_record,
                        )
                        records.append(record)
                    for unmatched_pred_value in value_matching_record.unmatched_pred_values:
                        original_property_id = get_original_property_id(self.entities_pred[pred_idx], property_id)
                        property_id_ = property_id
                        if property_id in self.unmapped_property_ids:
                            property_id_ = ""  # use empty string for unmapped properties
                            original_property_id = property_id
                        record, first_property_record = create_property_record(
                            gt_entity_id,
                            pred_entity_id,
                            original_property_id,
                            property_id_,
                            "",
                            unmatched_pred_value,
                            None,
                            num_gt_values,
                            num_pred_values,
                            first_property_record,
                        )
                        records.append(record)
                records[0]["entity_match_distance"] = entity_match_distance
                entity_debug_info = pd.DataFrame.from_records(records)
                dfs.append(entity_debug_info)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def create_property_record(
    gt_entity_id: str,
    pred_entity_id: str,
    original_property_id: str,
    property_id: str,
    gt_value: Any,  # noqa: ANN401
    pred_value: Any,  # noqa: ANN401
    score: float | None,
    total_gt_values: int,
    total_pred_values: int,
    first_property_record: bool = False,
) -> tuple[dict, bool]:
    """Create a property evaluation record."""
    record = {
        "gt_entity_id": gt_entity_id,
        "pred_entity_id": pred_entity_id,
        "original_pred_property_id": original_property_id,
        "property_id": property_id,
        "gt_values": gt_value,
        "pred_values": pred_value,
        "matched_scores": score,
    }
    if first_property_record:
        record["num_gt_values"] = total_gt_values
        record["num_pred_values"] = total_pred_values
        first_property_record = False
    return record, first_property_record


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
        keys_to_score: dict[str, Callable[[Entity, Entity, Property], ValueMatchingRecordWihScores]],
    ) -> tuple[MetricsAccumulator, pd.DataFrame]:
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
        # properties_union = properties_union.intersection(self.property_schema.properties.keys())
        entities_scorer = MatchedEntitiesScorer(self.ground_truth, self.predictions, matched_pair, self.property_schema)

        for key in properties_union:
            score_func = keys_to_score[key]

            evaluated_property_metrics = entities_scorer.score(score_func, key)

            if key in self.property_schema.properties:
                # only accumulate metrics for properties that are in the schema
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
        matched_entities_debug_info = entities_scorer.get_debug_info()
        unmatched_gt_records = []
        unmatched_gt_indices = set(range(len(self.ground_truth))) - set(matched_pair.left_ind)
        for idx in unmatched_gt_indices:
            entity = self.ground_truth[idx]
            for property_id, values in entity.properties.items():
                first_property_record = True
                for value in values:
                    record, first_property_record = create_property_record(
                        entity.entity_id,
                        "",  # No pred_entity_id for unmatched ground truth entities
                        "",  # No original_property_id for unmatched ground truth entities
                        property_id,
                        value,
                        "",
                        None,
                        len(entity.properties[property_id]),
                        0,
                        first_property_record,
                    )
                    unmatched_gt_records.append(record)
        unmatched_gt_entities_debug_info = pd.DataFrame.from_records(unmatched_gt_records)

        unmatched_pred_records = []
        unmatched_pred_indices = set(range(len(self.predictions))) - set(matched_pair.right_ind)
        for idx in unmatched_pred_indices:
            entity = self.predictions[idx]
            for property_id, values in entity.properties.items():
                first_property_record = True
                original_property_id = ""
                property_evidence = entity.evidence_map.get(property_id)
                if property_evidence:
                    evidence_idx = property_evidence[0][0]
                    original_property_id = entity.source_ids[evidence_idx]
                for value in values:
                    record, first_property_record = create_property_record(
                        "",
                        entity.entity_id,  # No gt_entity_id for unmatched prediction entities
                        original_property_id,
                        property_id,
                        "",
                        value,
                        None,
                        0,
                        len(entity.properties[property_id]),
                        first_property_record,
                    )
                    unmatched_pred_records.append(record)
        unmatched_pred_entities_debug_info = pd.DataFrame.from_records(unmatched_pred_records)

        return metrics_accumulator, pd.concat(
            [matched_entities_debug_info, unmatched_gt_entities_debug_info, unmatched_pred_entities_debug_info],
            ignore_index=True,
        )


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
