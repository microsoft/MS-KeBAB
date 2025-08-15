# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# ruff: noqa: ANN401
from __future__ import annotations

import abc
import sys
from dataclasses import dataclass
from typing import Any

import nltk
import numpy as np
import tiktoken
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

from kebab.contracts.entity import Entity, Property, PropertySchema, ValueType
from kebab.tasks.metrics.extraction.utils import normalize_property_value, normalize_string


class ElementDistance(abc.ABC):
    """Element distance class.
    Element distance values should be between 0 and 1.
    """

    @abc.abstractmethod
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for element distance."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self, value1: Any, value2: Any, property_: Property) -> float:
        """Compute distance between property values of ground truth and prediction entity."""
        raise NotImplementedError

    def __call__(self, value1: Any, value2: Any, property_: Property) -> float:
        """Compute distance between property values of ground truth and prediction entity."""
        self.check_constraints(property_)
        return self.compute(value1, value2, property_)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ElementDistance:
        """Create element distance from dictionary."""
        raise NotImplementedError


class TokenDistance(ElementDistance):
    """Token distance class."""

    __default_encoding: str = "cl100k_base"

    def __init__(self, encoder: tiktoken.Encoding | None = None):
        """Initialize token distance.

        Args:
            encoder: Optional tiktoken encoding instance for tokenization.
        """
        if encoder is None:
            encoder = tiktoken.get_encoding(self.__default_encoding)
        self.encoder = encoder

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for token distance."""
        if property_.data_type.value_type not in {ValueType.TEXT, ValueType.DATE}:
            raise ValueError("Only ValueType.TEXT and ValueType.DATE properties are supported.")

    def compute(self, value1: Any, value2: Any, property_: Property) -> float:  # noqa: ARG002
        """Compute token distance between property values of ground truth and prediction entity."""
        return token_distance_between_strings(str(value1), str(value2), self.encoder)

    @classmethod
    def from_dict(cls, config: dict[str, str]) -> TokenDistance:
        """Create token distance from dictionary."""
        return TokenDistance(encoder=tiktoken.get_encoding(config.get("encoding", cls.__default_encoding)))


class EmbeddingDistance(ElementDistance):
    """Embedding distance class."""

    __default_model: str = "paraphrase-MiniLM-L6-v2"

    def __init__(self, model: SentenceTransformer | None = None):
        """Initialize embedding distance.

        Args:
            model: Optional pre-trained SentenceTransformer model for generating embeddings.
        """
        if model is None:
            model = SentenceTransformer(self.__default_model)  # type: ignore
        self.model = model

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for embedding distance."""
        if property_.data_type.value_type is not ValueType.TEXT:
            raise ValueError("Only ValueType.TEXT properties are supported.")

    def compute(self, value1: Any, value2: Any, property_: Property) -> float:  # noqa: ARG002
        """Compute distance between string property values using sentence embeddings."""
        return embeddings_distance_between_strings(str(value1), str(value2), self.model)

    @classmethod
    def from_dict(cls, config: dict[str, str]) -> EmbeddingDistance:
        """Create embedding distance from dictionary."""
        return EmbeddingDistance(model=SentenceTransformer(config.get("model", cls.__default_model)))  # type: ignore


class BinaryMatchDistance(ElementDistance):
    """Binary match distance class."""

    def check_constraints(self, property_: Property) -> None:  # noqa: ARG002
        """Check constraints for binary match distance."""
        return None

    def compute(self, value1: Any, value2: Any, property_: Property) -> float:
        """Computes a distance between two entities for a given property id. Binary output of 0 if the values match, 1 otherwise."""
        value1 = normalize_property_value(value1, property_)
        value2 = normalize_property_value(value2, property_)
        return 0 if value1 == value2 else 1

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> BinaryMatchDistance:  # noqa: ARG003
        """Create binary match distance from dictionary."""
        return BinaryMatchDistance()


class EditDistance(ElementDistance):
    """Edit distance class."""

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for edit distance."""
        if property_.data_type.value_type is not ValueType.TEXT:
            raise ValueError("Only ValueType.TEXT properties are supported.")

    def compute(self, value1: Any, value2: Any, property_: Property) -> float:  # noqa: ARG002
        """Compute edit distance between two string property values."""
        return normalized_edit_distance(str(value1), str(value2))

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> EditDistance:  # noqa: ARG003
        """Create edit distance from dictionary."""
        return EditDistance()


PropertyDistanceValue = tuple[list[float], int]
PropertyScoreValue = tuple[list[float], int]


class PropertyDistance(abc.ABC):
    """Property distance class."""

    @abc.abstractmethod
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for property distance."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> ValueMatchingRecordWithDistances:
        """Compute distance between two entities for a given property id.

        Returns:
            - list of distances between matched property values of ground truth and prediction entity
            - number of unmatched property values in prediction entity"
        """
        raise NotImplementedError

    def __call__(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> ValueMatchingRecordWithDistances:
        """Compute distance between two entities for a given property id.

        Returns:
            - list of distances between matched property values of ground truth and prediction entity
            - number of unmatched property values in prediction entity"
        """
        self.check_constraints(property_)
        return self.compute(gt_entity, pred_entity, property_)


@dataclass
class ValueMatchingRecordWithDistances:
    """Value matching record containing distances between matched values.
    Distances are between 0 and 1.
    The lower the distance, the more similar the values are.
    """

    gt_values: list[Any]
    pred_values: list[Any]
    matched_distances: list[float]
    unmatched_gt_values: list[Any]
    unmatched_pred_values: list[Any]


@dataclass
class ValueMatchingRecordWithScores:
    """Value matching record with scores for pairs of matched values.
    Scores are between 0 and 1.
    The higher the score, the more similar the values are.
    """

    gt_values: list[Any]
    pred_values: list[Any]
    matched_scores: list[float]
    unmatched_gt_values: list[Any]
    unmatched_pred_values: list[Any]


class SingleValuePropertyDistance(PropertyDistance):
    """Single value property distance class."""

    def __init__(self, element_distance: ElementDistance):
        """Initialize single value property distance."""
        self.element_distance = element_distance

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for single value property distance."""
        if property_.is_collection:
            raise ValueError("Collection properties are not supported.")

    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> ValueMatchingRecordWithDistances:
        """Compute distance between two entities for a given property id.

        Args:
            gt_entity: Ground truth entity.
            pred_entity: Predicted entity.
            property_: Property to compute distance for.

        Returns:
            ValueMatchingRecordWithDistances containing list of distances between matched property values
        """
        gt_values = gt_entity[property_] if property_ in gt_entity else []  # noqa: SIM401
        pred_values = pred_entity[property_] if property_ in pred_entity else []  # noqa: SIM401
        if property_ not in gt_entity or property_ not in pred_entity:
            score = 1
        else:
            score = self.element_distance(gt_entity[property_][0], pred_entity[property_][0], property_)
        return ValueMatchingRecordWithDistances(gt_values, pred_values, [score], [], [])


class SetPropertyDistance(PropertyDistance):
    """Set distance class."""

    def __init__(self, element_distance: ElementDistance):
        """Initialize set distance."""
        self.element_distance = element_distance

    def check_constraints(self, property_: Property) -> None:  # noqa: ARG002
        """Check constraints for set distance."""
        return None

    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> ValueMatchingRecordWithDistances:
        """Compute distances between two sets of values for a given property.

        Args:
            gt_entity: Ground truth entity.
            pred_entity: Predicted entity.
            property_: Property to compute distances for.

        Returns:
            ValueMatchingRecordWithDistances containing distances between matched property values
        """
        gt_values = np.array(gt_entity[property_] if property_ in gt_entity else [])  # noqa: SIM401
        pred_values = np.array(pred_entity[property_] if property_ in pred_entity else [])  # noqa: SIM401
        if property_ not in gt_entity or property_ not in pred_entity:
            return ValueMatchingRecordWithDistances([], [], [1], list(gt_values), list(pred_values))

        distances = np.zeros((len(gt_values), len(pred_values)))
        for i, value1 in enumerate(gt_values):
            for j, value2 in enumerate(pred_values):
                distances[i, j] = self.element_distance(value1, value2, property_)
        result = match_items(distances)
        matched_scores = list(distances[result.left_ind, result.right_ind])
        unmatched_gt_indices = list(set(range(len(gt_values))) - set(result.left_ind))
        unmatched_pred_indices = list(set(range(len(pred_values))) - set(result.right_ind))
        return ValueMatchingRecordWithDistances(
            list(gt_values[result.left_ind]),
            list(pred_values[result.right_ind]),
            list(matched_scores),
            list(gt_values[unmatched_gt_indices]),
            list(pred_values[unmatched_pred_indices]),
        )


class PropertyScore:
    """Property score class."""

    def __init__(self, property_distance: PropertyDistance):
        """Initialize property score."""
        self.property_distance = property_distance

    def __call__(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> ValueMatchingRecordWithScores:
        """Compute the score for a given property.

        Args:
            gt_entity: Ground truth entity.
            pred_entity: Predicted entity.
            property_: Property to compute score for.

        Returns:
            ValueMatchingRecordWithScores containing scores for matched property values
        """
        value_matching_record = self.property_distance(gt_entity, pred_entity, property_)
        for distance in value_matching_record.matched_distances:
            if distance < 0 or distance > 1:
                raise ValueError(f"Distance value {distance} is out of range [0, 1].")
        return ValueMatchingRecordWithScores(
            value_matching_record.gt_values,
            value_matching_record.pred_values,
            [1.0 - d for d in value_matching_record.matched_distances],
            value_matching_record.unmatched_gt_values,
            value_matching_record.unmatched_pred_values,
        )


class EntityDistance:
    """Distance between two entities.

    Computed as a weighted sum of distances between properties.
    """

    DEFAULT_ELEMENT_DISTANCE_CLS = TokenDistance
    DEFAULT_PROPERTY_WEIGHT = 1.0

    def __init__(
        self,
        property_schema: PropertySchema,
        property_to_distance_function_and_weight: dict[str, tuple[PropertyDistance, float]],
        default_property_distance: PropertyDistance,
        default_property_weight: float = DEFAULT_PROPERTY_WEIGHT,
    ):
        """Initialize entity distance.

        Args:
            property_schema: Schema defining properties and their data types.
            property_to_distance_function_and_weight: Mapping of property names to distance functions and weights.
            default_property_distance: Default distance function for properties not explicitly configured.
            default_property_weight: Default weight for properties not explicitly configured.
        """
        self.property_to_distance_function_and_weight = property_to_distance_function_and_weight
        self.property_schema = property_schema
        self.default_property_distance = default_property_distance
        self.default_property_weight = default_property_weight
        self.__normalize_weights()

    def __call__(self, gt_entity: Entity, pred_entity: Entity) -> float:
        """Compute distance between two entities."""
        distance = 0.0
        for property_id, (property_distance, weight) in self.property_to_distance_function_and_weight.items():
            value_matching_record = property_distance(
                gt_entity, pred_entity, self.property_schema.properties[property_id]
            )
            distance += float(np.mean(value_matching_record.matched_distances)) * weight
        return distance

    def __normalize_weights(self):
        """Normalize weights to sum to 1."""
        weight_sum = 0.0
        for property_id in self.property_schema.properties:
            if property_id not in self.property_to_distance_function_and_weight:
                weight_sum += self.default_property_weight
            else:
                weight_sum += self.property_to_distance_function_and_weight[property_id][1]
        for property_id in self.property_schema.properties:
            if property_id not in self.property_to_distance_function_and_weight:
                self.property_to_distance_function_and_weight[property_id] = (
                    self.default_property_distance,
                    self.default_property_weight / weight_sum,
                )
            else:
                self.property_to_distance_function_and_weight[property_id] = (
                    self.property_to_distance_function_and_weight[property_id][0],
                    self.property_to_distance_function_and_weight[property_id][1] / weight_sum,
                )

    @classmethod
    def from_dict(cls, config: dict[str, Any], property_schema: PropertySchema) -> EntityDistance:
        """Create entity distance from dictionary."""
        property_to_distance_function_and_weight = {}
        default_weight = config.get("default_property_weight", cls.DEFAULT_PROPERTY_WEIGHT)
        for property_id, params in config["property_to_distance"].items():
            if property_id not in property_schema.properties:
                raise ValueError(f"Property '{property_id}' is not in the property schema.")
            element_distance = getattr(sys.modules[__name__], params["distance_function"]["name"]).from_dict(
                params["distance_function"].get("params", {})
            )
            property_distance = (
                SetPropertyDistance(element_distance)
                if property_schema.properties[property_id].is_collection
                else SingleValuePropertyDistance(element_distance)
            )
            weight = params.get("weight", default_weight)
            property_to_distance_function_and_weight[property_id] = (property_distance, weight)
        default_element_distance_cls = (
            getattr(sys.modules[__name__], config["default_property_distance"]["name"])
            if "default_property_distance" in config
            else cls.DEFAULT_ELEMENT_DISTANCE_CLS
        )
        default_property_distance = SetPropertyDistance(
            default_element_distance_cls.from_dict(config["default_property_distance"].get("params", {}))
        )
        return EntityDistance(
            property_schema, property_to_distance_function_and_weight, default_property_distance, default_weight
        )


#######################################################################################################################


def jaccard_distance(tokens1: list[int], tokens2: list[int]) -> float:
    """Jaccard distance."""
    set1 = set(tokens1)
    set2 = set(tokens2)
    return 1 - len(set1.intersection(set2)) / len(set1.union(set2))


def token_distance_between_strings(value1: str, value2: str, encoder: tiktoken.Encoding) -> float:
    """Compute token distance between two string values.
    Distance value is between 0 and 1.

    Args:
        value1: First string value to compare.
        value2: Second string value to compare.
        encoder: Tiktoken encoding instance for tokenization.

    Returns:
        Jaccard distance between token sets of the two strings.
    """
    tokens1 = encoder.encode(normalize_string(value1))
    tokens2 = encoder.encode(normalize_string(value2))
    return jaccard_distance(tokens1, tokens2)


def embeddings_distance_between_strings(value1: str, value2: str, model: SentenceTransformer) -> float:
    """Compute embedding distance between two string values.
    Distance value is between 0 and 1.

    Args:
        value1: First string value to compare.
        value2: Second string value to compare.
        model: Pre-trained SentenceTransformer model for generating embeddings.

    Returns:
        Cosine distance between embeddings of the two strings.
    """
    embeds = model.encode([value1, value2])
    return 0.5 * float(cosine(embeds[0], embeds[1]))  # scipy.spatial.distance.cosine is between 0 and 2


def normalized_edit_distance(value1: str, value2: str) -> float:
    """Compute edit distance between 2 strings and normalize it by length of the longest string.

    Args:
        value1: First string to compare.
        value2: Second string to compare.

    Returns:
        Normalized edit distance between 0 and 1.
    """
    return abs(nltk.edit_distance(value1, value2)) / (max(len(value1), len(value2)))


@dataclass
class MatchedIndices:
    """Stores the indices of matched items and unmatched items in ground truth and prediction."""

    left_ind: list[int]
    right_ind: list[int]
    left_unmatched: list[int]
    right_unmatched: list[int]


def match_items(distances: np.ndarray, threshold: float = 1.0) -> MatchedIndices:
    """Match items from two sets of items based on a distance matrix if the distance is within a threshold.

    Args:
        distances: 2D array where distances[i,j] is the distance between item i and item j.
        threshold: Maximum distance threshold for considering items as matched.

    Returns:
        MatchedIndices object containing indices of matched and unmatched items.
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
