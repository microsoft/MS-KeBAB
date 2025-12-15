# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
from collections import defaultdict, Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from kebab.contracts.entity import Entity, Property, PropertySchema
from kebab.tasks.metrics.extraction.aesop.distances import ValueMatchingRecordWithScores, match_items
from kebab.tasks.metrics.extraction.aesop.config import ValueAveragedAesopConfig

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


def get_full_class_name(obj: object) -> str:
    """
    Get the class name of an object.

    Args:
        obj: The object to get the class name for.

    Returns:
        The class name of the object.
    """
    if obj is None:
        return "NoneType"
    return f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"


class EntityMatcher:
    """Matches entities in two disjoint sets based on a distance function and a matching score threshold."""

    def __init__(
        self, entities_gt: list[Entity], entities_pred: list[Entity], logger: logging.Logger | None = None
    ) -> None:
        """Initialize the EntityMatcher.

        Args:
            entities_gt: List of ground truth entities to match.
            entities_pred: List of predicted entities to match.
            logger: Optional logger instance for logging matching progress.
        """
        self.entities_gt = entities_gt
        self.entities_pred = entities_pred
        self.logger = logger or logging.getLogger(get_full_class_name(self))

    def match(self, distance_function: Callable[[Entity, Entity], float], threshold: float = 1.0) -> MatchedPairInfo:
        """Match entities based on a distance function and a matching score threshold.

        Args:
            distance_function: Function that computes distance between two entities.
            threshold: Maximum distance threshold for considering entities as matched.

        Returns:
            MatchedPairInfo containing indices of matched and unmatched entities.
        """
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
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the MatchedEntitiesScorer.

        Args:
            entities_gt: List of ground truth entities.
            entities_pred: List of predicted entities.
            matched_pair: Information about matched entity pairs.
            property_schema: Schema defining properties and their data types.
            logger: Optional logger instance for logging scoring progress.
        """
        self.entities_gt = entities_gt
        self.entities_pred = entities_pred

        if len(matched_pair.left_ind) != len(matched_pair.right_ind):
            raise ValueError("left_ind and right_ind must have the same length.")

        self.gt_ind = matched_pair.left_ind
        self.pred_ind = matched_pair.right_ind
        self.distances = matched_pair.distances
        self.property_schema = property_schema
        self.debug_info: defaultdict[int, defaultdict[int, dict[str, ValueMatchingRecordWithScores]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.unmapped_property_ids: set[str] = set()
        self.logger = logger or logging.getLogger(get_full_class_name(self))

    def score(
        self,
        score_function: Callable[[Entity, Entity, Property], ValueMatchingRecordWithScores],
        property_id: str,
    ) -> EvaluatedPropertyMetrics:
        """Compute scores for a given property for matched entities.

        Args:
            score_function: Function that computes scores for matched property values of 2 entities.
            property_id: ID of the property to evaluate.

        Returns:
            EvaluatedPropertyMetrics containing scores for the property.
        """
        evaluated_property_metrics = EvaluatedPropertyMetrics(len(self.entities_gt))

        property_ = self.property_schema.properties.get(property_id)

        if property_ is None:
            # property is not in schema, but we still need to create debug info for predictions
            for gt_idx, pred_idx in zip(self.gt_ind, self.pred_ind, strict=True):
                if property_id in self.entities_pred[pred_idx].properties:
                    pred_values = self.entities_pred[pred_idx].properties[property_id]
                    self.debug_info[gt_idx][pred_idx][property_id] = ValueMatchingRecordWithScores(
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
                    self.debug_info[gt_idx][pred_idx][property_.property_id] = ValueMatchingRecordWithScores(
                        [], [], [], self.entities_gt[gt_idx][property_], []
                    )

                evaluated_property_metrics.property_value_count_gt[gt_idx] += len(self.entities_gt[gt_idx][property_])
            elif property_ in self.entities_pred[pred_idx]:
                pred_values = self.entities_pred[pred_idx][property_]
                self.debug_info[gt_idx][pred_idx][property_.property_id] = ValueMatchingRecordWithScores(
                    [], [], [], [], pred_values
                )
        return evaluated_property_metrics

    def get_debug_info(self) -> pd.DataFrame | None:
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
                if len(records) > 0:
                    records[0]["entity_match_distance"] = entity_match_distance
                    entity_debug_info = pd.DataFrame.from_records(records)
                    if entity_debug_info.isna().to_numpy().all():
                        self.logger.warning(
                            f"All values are null for matched entities: {gt_entity_id} and {pred_entity_id}"
                        )
                    else:
                        dfs.append(entity_debug_info)
                else:
                    self.logger.warning(f"No records found for matched entities: {gt_entity_id} and {pred_entity_id}")
        if len(dfs) == 0:
            self.logger.warning("No debug information collected for matched entities.")
            return None
        return pd.concat(dfs, ignore_index=True)


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
    """Create a property evaluation record.

    Args:
        gt_entity_id: ID of the ground truth entity.
        pred_entity_id: ID of the predicted entity.
        original_property_id: Original property ID from the prediction.
        property_id: Standardized property ID.
        gt_value: Ground truth property value.
        pred_value: Predicted property value.
        score: Similarity score between the values.
        total_gt_values: Total number of ground truth values for this property.
        total_pred_values: Total number of predicted values for this property.
        first_property_record: Whether this is the first record for this property.

    Returns:
        Tuple of the property record dictionary and updated first_property_record flag.
    """
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
        config: ValueAveragedAesopConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the MetricsComputer.

        Args:
            logger: Optional logger instance for logging computation progress.
        """
        self.config = config
        self.logger = logger or logging.getLogger(get_full_class_name(self))

    def compute_bipartite_metrics(self, context: dict[str, Any]) -> tuple[MetricsAccumulator, pd.DataFrame | None]:
        """Compute provided property scores for the matched entities.

        Args:
            matched_pair: Information about matched and unmatched entity pairs.
            keys_to_score: Dictionary mapping property IDs to scoring functions.

        Returns:
            Tuple of MetricsAccumulator with computed scores and optional debug DataFrame.
        """
        matched_pair = context["matched_pair"]
        ground_truth = context["gt_entities"]
        predictions = context["pred_entities"]
        metrics_accumulator = MetricsAccumulator()
        metrics_accumulator.matched_count = len(matched_pair.left_ind)
        metrics_accumulator.unmatched_pair_count = len(matched_pair.left_unmatched)
        metrics_accumulator.total_gt_entities = len(ground_truth)
        metrics_accumulator.total_pred_entities = len(predictions)

        diff = len(ground_truth) - len(predictions)
        if diff > 0:
            metrics_accumulator.unmatched_extra_gt_count = diff
        elif diff < 0:
            metrics_accumulator.unmatched_extra_pred_count = -diff

        properties_union = compute_properties_union(ground_truth, predictions, matched_pair)
        entities_scorer = MatchedEntitiesScorer(ground_truth, predictions, matched_pair, self.config.property_schema)

        for key in properties_union:
            score_func = self.config.get_score_for_property(key, context)
            if score_func is None:
                raise ValueError(f"No scoring function found for property {key}")

            evaluated_property_metrics = entities_scorer.score(score_func, key)

            if key in self.config.property_schema.properties:
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
        unmatched_gt_indices = set(range(len(ground_truth))) - set(matched_pair.left_ind)
        for idx in unmatched_gt_indices:
            entity = ground_truth[idx]
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
        unmatched_gt_entities_debug_info = (
            pd.DataFrame.from_records(unmatched_gt_records) if unmatched_gt_records else None
        )

        if unmatched_gt_entities_debug_info is None or unmatched_gt_entities_debug_info.isna().to_numpy().all():
            self.logger.warning("All values are null for unmatched ground truth entities.")
            unmatched_gt_entities_debug_info = None

        unmatched_pred_records = []
        unmatched_pred_indices = set(range(len(predictions))) - set(matched_pair.right_ind)
        for idx in unmatched_pred_indices:
            entity = predictions[idx]
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
        unmatched_pred_entities_debug_info = (
            pd.DataFrame.from_records(unmatched_pred_records) if unmatched_pred_records else None
        )

        if unmatched_pred_entities_debug_info is None or unmatched_pred_entities_debug_info.isna().to_numpy().all():
            self.logger.warning("All values are null for unmatched prediction entities.")
            unmatched_pred_entities_debug_info = None

        debug_info_parts = [
            matched_entities_debug_info,
            unmatched_gt_entities_debug_info,
            unmatched_pred_entities_debug_info,
        ]
        debug_info_parts = [df for df in debug_info_parts if df is not None]
        return metrics_accumulator, pd.concat(debug_info_parts, ignore_index=True) if debug_info_parts else None


def compute_properties_union(
    ground_truth: list[Entity],
    predictions: list[Entity],
    matched_pair: MatchedPairInfo,
) -> set[str]:
    """Compute the union of the properties occuring in matched entities from both ground truth and predictions.

    Args:
        ground_truth: List of ground truth entities.
        predictions: List of predicted entities.
        matched_pair: Information about matched entity pairs.

    Returns:
        Set of property IDs that appear in matched entities from either ground truth or predictions.
    """
    ground_truth_keys = {
        key for idx, entity in enumerate(ground_truth) if idx in matched_pair.left_ind for key in entity.properties
    }
    predictions_keys = {
        key for idx, entity in enumerate(predictions) if idx in matched_pair.right_ind for key in entity.properties
    }
    return ground_truth_keys.union(predictions_keys)