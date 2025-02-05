# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# ruff: noqa: ANN401
from __future__ import annotations

import sys
from typing import Any

import nltk
import numpy as np
import tiktoken
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

from kebab.contracts.entity import Entity, Property, PropertySchema, ValueType
from kebab.tasks.extraction.metrics.aesop.metric_helpers import match_items
from kebab.tasks.extraction.metrics.utils import normalize_property_value, normalize_string


class ElementDistance:
    """Element distance class.
    Element distance values should be between 0 and 1.
    """

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for element distance."""
        raise NotImplementedError

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
        """Initialize token distance."""
        if encoder is None:
            encoder = tiktoken.get_encoding(self.__default_encoding)
        self.encoder = encoder

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for token distance."""
        if property_.data_type.value_type is not ValueType.TEXT:
            raise ValueError("Only ValueType.TEXT properties are supported.")

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
        """Initialize embedding distance."""
        if model is None:
            model = SentenceTransformer(self.__default_model)
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
        return EmbeddingDistance(model=SentenceTransformer(config.get("model", cls.__default_model)))


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


class PropertyDistance:
    """Property distance class."""

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for property distance."""
        raise NotImplementedError

    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> PropertyDistanceValue:
        """Compute distance between two entities for a given property id.

        Returns:
            - list of distances between matched property values of ground truth and prediction entity
            - number of unmatched property values in prediction entity"
        """
        raise NotImplementedError

    def __call__(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> PropertyDistanceValue:
        """Compute distance between two entities for a given property id.

        Returns:
            - list of distances between matched property values of ground truth and prediction entity
            - number of unmatched property values in prediction entity"
        """
        self.check_constraints(property_)
        return self.compute(gt_entity, pred_entity, property_)


class SingleValuePropertyDistance(PropertyDistance):
    """Single value property distance class."""

    def __init__(self, element_distance: ElementDistance):
        """Initialize single value property distance."""
        self.element_distance = element_distance

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for single value property distance."""
        if property_.is_collection:
            raise ValueError("Collection properties are not supported.")

    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> PropertyDistanceValue:
        """Compute distance between two entities for a given property id.

        Returns:
            - list of distances between matched property values of ground truth and prediction entity
            - number of unmatched property values in prediction entity (always 0 for single value properties)"
        """
        if property_ not in gt_entity or property_ not in pred_entity:
            return [1], 0
        return [self.element_distance(gt_entity[property_][0], pred_entity[property_][0], property_)], 0


class SetPropertyDistance(PropertyDistance):
    """Set distance class."""

    def __init__(self, element_distance: ElementDistance):
        """Initialize set distance."""
        self.element_distance = element_distance

    def check_constraints(self, property_: Property) -> None:  # noqa: ARG002
        """Check constraints for set distance."""
        return None

    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> PropertyDistanceValue:
        """Compute distances between two sets of values for a given property.

        Returns:
            - list of distances between matched property values of ground truth
                and prediction entity (needed for aggregation across entities)
            - number of unmatched property values in prediction entity.
        """
        if property_ not in gt_entity or property_ not in pred_entity:
            return [1], 0

        gt_values = gt_entity[property_]
        pred_values = pred_entity[property_]

        distances = np.zeros((len(gt_values), len(pred_values)))
        for i, value1 in enumerate(gt_values):
            for j, value2 in enumerate(pred_values):
                distances[i, j] = self.element_distance(value1, value2, property_)
        matched_indices = match_items(distances)
        matched_scores = list(distances[matched_indices.left_ind, matched_indices.right_ind])
        unmatched_pred_count = len(pred_values) - len(matched_indices.right_ind)
        return matched_scores, unmatched_pred_count


class PropertyScore:
    """Property score class."""

    def __init__(self, property_distance: PropertyDistance):
        """Initialize property score."""
        self.property_distance = property_distance

    def __call__(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> PropertyScoreValue:
        """Compute the score for a given property."""
        distances, unmatched_count = self.property_distance(gt_entity, pred_entity, property_)
        for distance in distances:
            if distance < 0 or distance > 1:
                raise ValueError(f"Distance value {distance} is out of range [0, 1].")
        return [1.0 - d for d in distances], unmatched_count


class EntityDistance:
    """Distance between two entities.

    Computed as a weighted sum of distances between properties.
    """

    def __init__(
        self,
        property_schema: PropertySchema,
        property_to_distance_function_and_weight: dict[str, tuple[PropertyDistance, float]],
    ):
        """Initialize entity distance."""
        self.property_to_distance_function_and_weight = property_to_distance_function_and_weight
        self.property_schema = property_schema

    def __call__(self, gt_entity: Entity, pred_entity: Entity) -> float:
        """Compute distance between two entities."""
        distance = 0.0
        for property_id, (property_distance, weight) in self.property_to_distance_function_and_weight.items():
            value_distances, _ = property_distance(gt_entity, pred_entity, self.property_schema.properties[property_id])
            distance += float(np.mean(value_distances)) * weight
        return distance

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> EntityDistance:
        """Create entity distance from dictionary."""
        property_schema = PropertySchema.from_file(config["property_schema"])
        property_to_distance_function_and_weight = {}
        num_properties = len(config["entity_distance"])
        num_missing_weights = sum("weight" not in params for params in config["entity_distance"].values())
        default_weight = 1.0 / (num_properties - num_missing_weights) if num_missing_weights < num_properties else 1.0
        for property_id, params in config["entity_distance"].items():
            element_distance = getattr(sys.modules[__name__], params["distance_function"]["name"]).from_dict(
                params["distance_function"].get("params", {})
            )
            property_distance = (
                SetPropertyDistance(element_distance)
                if property_schema.properties[property_id].is_collection
                else SingleValuePropertyDistance(element_distance)
            )
            property_to_distance_function_and_weight[property_id] = (
                property_distance,
                params.get("weight", default_weight),
            )
        return EntityDistance(property_schema, property_to_distance_function_and_weight)


#######################################################################################################################


def jaccard_distance(tokens1: list[int], tokens2: list[int]) -> float:
    """Jaccard distance."""
    set1 = set(tokens1)
    set2 = set(tokens2)
    return 1 - len(set1.intersection(set2)) / len(set1.union(set2))


def token_distance_between_strings(value1: str, value2: str, encoder: tiktoken.Encoding) -> float:
    """Compute token distance between two string values.
    Distance value is between 0 and 1.
    """
    tokens1 = encoder.encode(normalize_string(value1))
    tokens2 = encoder.encode(normalize_string(value2))
    return jaccard_distance(tokens1, tokens2)


def embeddings_distance_between_strings(value1: str, value2: str, model: SentenceTransformer) -> float:
    """Compute embedding distance between two string values.
    Distance value is between 0 and 1.
    """
    embeds = model.encode([value1, value2])
    return 0.5 * float(cosine(embeds[0], embeds[1]))  # scipy.spatial.distance.cosine is between 0 and 2


def normalized_edit_distance(value1: str, value2: str) -> float:
    """Compute edit distance between 2 strings and normalize it by length of the longest string."""
    return abs(nltk.edit_distance(value1, value2)) / (max(len(value1), len(value2)))
