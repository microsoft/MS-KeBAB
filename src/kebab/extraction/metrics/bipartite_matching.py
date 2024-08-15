# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pyright: reportArgumentType=false, reportAssignmentType=false, reportOperatorIssue=false
# ruff: noqa: D103, E731
from __future__ import annotations

import copy
from collections import defaultdict
from logging import Logger
from typing import Any, Callable

import nltk
import numpy as np
import tiktoken
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

from kebab.extraction.metrics.metric_helpers import (
    Entity,
    EntityMatcher,
    MetricsAccumulator,
    MetricsComputer,
    match_items,
)
from kebab.extraction.metrics.property_statistics import normalize_value


PropertyMetricDict = dict[str, float | None]


def jaccard_distance(tokens1: list[int], tokens2: list[int]) -> float:
    set1 = set(tokens1)
    set2 = set(tokens2)
    return 1 - len(set1.intersection(set2)) / len(set1.union(set2))


def token_distance(
    gt_entity: Entity | str,
    pred_entity: Entity | str,
    key: str = "name",
    encoder: tiktoken.Encoding | None = None,
    logger: Logger | None = None,
) -> float | None:
    if encoder is None:
        encoder = tiktoken.get_encoding("cl100k_base")

    if isinstance(gt_entity, (str, int, float)) and isinstance(pred_entity, (str, int, float)):
        return token_distance_between_strings(str(gt_entity), str(pred_entity), encoder)

    if not key or key not in gt_entity or key not in pred_entity:
        return 1

    # no precision for properties that are missing from the predictions
    if key in gt_entity and key not in pred_entity:
        return None

    if not (isinstance(gt_entity[key], str) and isinstance(pred_entity[key], str)):
        if logger:
            logger.info(f"Skipping non-string property {key} in {gt_entity} and {pred_entity}")
        return 1

    return token_distance_between_strings(gt_entity[key], pred_entity[key], encoder)


def embedding_distance(
    gt_entity: Entity | str,
    pred_entity: Entity | str,
    model: SentenceTransformer,
    key: str = "name",
    logger: Logger | None = None,
) -> float | None:
    if isinstance(gt_entity, str) and isinstance(pred_entity, str):
        return embeddings_distance_between_strings(gt_entity, pred_entity, model)

    if not key or key not in gt_entity or key not in pred_entity:
        return 1.0

    # no precision for properties that are missing from the predictions
    if key in gt_entity and key not in pred_entity:
        return None

    if not (isinstance(gt_entity[key], str) and isinstance(pred_entity[key], str)):
        if logger:
            logger.info(
                f"Skipping non-string property {key} in {gt_entity} and {pred_entity}. Comparison of collection values is not currently supported."
            )
        return 1.0

    return embeddings_distance_between_strings(gt_entity[key], pred_entity[key], model)


def token_distance_between_strings(value1: str, value2: str, encoder: tiktoken.Encoding) -> float:
    """Compute token distance between two string values."""
    tokens1 = encoder.encode(normalize_value(value1))
    tokens2 = encoder.encode(normalize_value(value2))
    return jaccard_distance(tokens1, tokens2)


def embeddings_distance_between_strings(value1: str, value2: str, model: SentenceTransformer) -> float:
    """Compute embedding distance between two string values."""
    embeds = model.encode([value1, value2])
    return max(0, cosine(embeds[0], embeds[1]))


def str_match_distance(entity1: Entity, entity2: Entity, key: str = "type", normalize: bool = False) -> float:
    """Computes a string distance between two entities for a given property key. Binary output of 0 if the values match, 1 otherwise."""
    if key not in entity1 or key not in entity2:
        return 1
    value1 = entity1[key]
    value2 = entity2[key]
    if normalize:
        value1 = normalize_value(value1)
        value2 = normalize_value(value2)
    return 0 if entity1[key] == entity2[key] else 1


def edit_distance(entity1: Entity, entity2: Entity, key: str = "type", normalize: bool = False) -> float:
    """Computes an edit distance between two entities for a given property key."""
    if key not in entity1 or key not in entity2:
        return 1
    value1 = entity1[key]
    value2 = entity2[key]
    if normalize:
        value1 = normalize_value(value1)
        value2 = normalize_value(value2)
    return normalized_edit_distance(value1, value2)


def normalized_edit_distance(value1: str, value2: str) -> float:
    return abs(nltk.edit_distance(value1, value2)) / (max(len(value1), len(value2)))


def set_distance(
    gt_entity: Entity,
    pred_entity: Entity,
    key: str,
    element_distance: Callable[[str | Any, str | Any, Any], float],
) -> list[float | None]:
    """Computes the average distance between two sets of values for a given key."""
    if key not in gt_entity or key not in pred_entity:
        return [1]

    gt_values = gt_entity[key]
    pred_values = pred_entity[key]

    if not isinstance(gt_values, list):
        gt_values = [gt_values]
    if not isinstance(pred_values, list):
        pred_values = [pred_values]

    distances = np.zeros((len(gt_values), len(pred_values)))
    for i, value1 in enumerate(gt_values):
        for j, value2 in enumerate(pred_values):
            distances[i, j] = element_distance(value1, value2, key)
    matched_indices = match_items(distances)
    return list(distances[matched_indices.left_ind, matched_indices.right_ind])


def weighted_distance(
    entity1: Entity,
    entity2: Entity,
    keys: list[str],
    element_distances: list[Callable[[str | Any, str | Any, Any], float]],
    weights: np.ndarray,
) -> float:
    weights = np.array(weights)
    weights /= weights.sum()
    distances = np.zeros(len(keys))
    for i, (key, element_distance) in enumerate(zip(keys, element_distances)):
        distances[i] = element_distance(entity1, entity2, key)
    return weights.dot(distances)


def normalize_property_names(entity: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Normalize the property names of an entity and convert them to lists."""
    properties_to_combine = {
        "descriptions": ["description", "definitions", "definition"],
        "risks": ["risk"],
        "types": ["type"],
        "employees": ["employee"],
        "topics": ["topic"],
        "teams": ["team"],
        "objectives": ["objective"],
        "part of": ["part_of"],
        "issues": ["issue", "limitation", "limitations"],
        "fax": ["fax number"],
        "phone": ["mobile phone", "number", "phone number", "mobile"],
    }

    for new_key, keys in properties_to_combine.items():
        for key in keys:
            value = entity.pop(key, None)
            if not value:
                continue

            if new_key in entity:
                if isinstance(value, str):
                    entity[new_key] = (
                        entity[new_key] + [value] if isinstance(entity[new_key], list) else [entity[new_key], value]
                    )
                elif isinstance(value, list):
                    entity[new_key] = (
                        entity[new_key] + value if isinstance(entity[new_key], list) else [entity[new_key], *value]
                    )

            else:
                entity[new_key] = value if isinstance(value, list) else [value]

    return entity


def get_set_score(
    e1: Entity, e2: Entity, key: str, element_distance: Callable[[str | Any, str | Any, Any], float]
) -> list[float]:
    """Compute the set score for a given key where set score is an array of scores between matched pairs of values in the sets."""
    distance = set_distance(e1, e2, key, element_distance)
    return [1.0 - d for d in distance]


def compute_bipartite_metrics(
    ground_truth: list[Entity],
    predictions: list[Entity],
    normalize: bool = False,
    embed_model: SentenceTransformer | None = None,
    matching_threshold: float = 1.0,
) -> MetricsAccumulator:
    if normalize:
        ground_truth = [normalize_property_names(entity) for entity in ground_truth]
        predictions = [normalize_property_names(entity) for entity in predictions]

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(lambda e1, e2: token_distance(e1, e2, key="name"), matching_threshold)

    if not embed_model:
        embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    set_token_score = lambda e1, e2, key: get_set_score(e1, e2, key, token_distance)
    str_score = lambda e1, e2, key: [1 - str_match_distance(e1, e2, key)]
    embedding_score = lambda e1, e2, key: [1 - embedding_distance(e1, e2, embed_model, key)]
    set_embedding_distance = lambda e1, e2, key: embedding_distance(e1, e2, embed_model, key)
    set_embedding_score = lambda e1, e2, key: get_set_score(e1, e2, key, set_embedding_distance)
    edit_score = lambda e1, e2, key: [1 - edit_distance(e1, e2, key)]

    keys_to_score = defaultdict(lambda: set_token_score)

    # Set score functions are used for properties for which we expect multiple values. Purpose is not observed to have multiple values.
    # This should be updated if we observe multiple values for purpose.
    # Score function always returns a list of scores, even if there is only one value.
    keys_to_score.update(
        {
            "descriptions": set_embedding_score,
            "definitions": set_embedding_score,
            "objectives": set_embedding_score,
            "risks": set_embedding_score,
            "purpose": embedding_score,
            "url": str_score,
            "email": str_score,
            "password": str_score,
            "name": edit_score,
        }
    )

    metrics_computer = MetricsComputer(ground_truth, predictions)
    metrics_accumulator = metrics_computer.compute_bipartite_metrics(matched_pair, keys_to_score)

    return metrics_accumulator


def compute_scores_across_documents(
    metrics_accumulator: MetricsAccumulator,
) -> tuple[PropertyMetricDict, PropertyMetricDict]:
    """Compute the precision and recall for all properties across all documents according to the given metrics accumulator.
    Returns precision and recall dictionaries, where the keys are the property names and the values are the corresponding scores.
    """
    property_precision = {}
    property_recall = {}
    total_scores_for_precision = copy.deepcopy(metrics_accumulator.total_scores_per_doc)

    for key, score_values in metrics_accumulator.total_scores_per_doc.items():
        for i in range(len(score_values)):
            if (
                not metrics_accumulator.total_zero_out_precision[key][i]
                and metrics_accumulator.total_zero_out_recall[key][i]
            ):
                # no information about the precision of the current example
                total_scores_for_precision[key][i] = None
            if (
                not metrics_accumulator.total_zero_out_recall[key][i]
                and metrics_accumulator.total_zero_out_precision[key][i]
            ):
                # no information about the current recall
                metrics_accumulator.total_property_value_counts_per_doc[key][i] = None

        valid_precision_scores = []
        for scorelist in total_scores_for_precision[key]:
            if scorelist is None:
                continue
            for score in scorelist:
                if score is None:
                    continue
                valid_precision_scores.append(score)

        valid_recall_scores = [
            score for score in metrics_accumulator.total_property_value_counts_per_doc[key] if score is not None
        ]

        property_precision[key] = (
            np.mean(valid_precision_scores + [0] * metrics_accumulator.total_unmatched_counts_per_doc[key])
            if len(valid_precision_scores)
            else None
        )

        flattened_total_scores_per_doc = [
            score for scorelist in metrics_accumulator.total_scores_per_doc[key] for score in scorelist
        ]
        property_recall[key] = (
            np.sum(flattened_total_scores_per_doc) / np.sum(valid_recall_scores)
            if np.sum(valid_recall_scores) != 0
            else None
        )

    return property_precision, property_recall


def compute_umatched_statistics(metrics_accumulator: MetricsAccumulator) -> tuple[float, float, float]:
    """Compute the fractions of unmatched pairs and extra entities in the ground truth and predictions."""
    unmatched_pair_fraction = metrics_accumulator.unmatched_pair_count / (
        metrics_accumulator.unmatched_pair_count + metrics_accumulator.matched_count
    )

    extra_pred_entities_fraction = (
        metrics_accumulator.unmatched_extra_pred_count / metrics_accumulator.total_pred_entities
    )

    extra_gt_entities_fraction = metrics_accumulator.unmatched_extra_gt_count / metrics_accumulator.total_gt_entities

    return unmatched_pair_fraction, extra_pred_entities_fraction, extra_gt_entities_fraction
