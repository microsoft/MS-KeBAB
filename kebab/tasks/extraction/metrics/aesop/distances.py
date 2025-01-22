# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# ruff: noqa: ANN401
from __future__ import annotations

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
    """Element distance class."""

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


class TokenDistance(ElementDistance):
    """Token distance class."""

    def __init__(self, encoder: tiktoken.Encoding | None = None):
        """Initialize token distance."""
        if encoder is None:
            encoder = tiktoken.get_encoding("cl100k_base")
        self.encoder = encoder

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for token distance."""
        if property_.data_type.value_type is not ValueType.TEXT:
            raise ValueError("Only ValueType.TEXT properties are supported.")

    def compute(self, value1: Any, value2: Any, property_: Property) -> float:  # noqa: ARG002
        """Compute token distance between property values of ground truth and prediction entity."""
        return token_distance_between_strings(str(value1), str(value2), self.encoder)


class EmbeddingDistance(ElementDistance):
    """Embedding distance class."""

    def __init__(self, model: SentenceTransformer | None = None):
        """Initialize embedding distance."""
        if model is None:
            model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.model = model

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for embedding distance."""
        if property_.data_type.value_type is not ValueType.TEXT:
            raise ValueError("Only ValueType.TEXT properties are supported.")

    def compute(self, value1: Any, value2: Any, property_: Property) -> float:  # noqa: ARG002
        """Compute distance between string property values using sentence embeddings."""
        return embeddings_distance_between_strings(str(value1), str(value2), self.model)


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


class EditDistance(ElementDistance):
    """Edit distance class."""

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for edit distance."""
        if property_.data_type.value_type is not ValueType.TEXT:
            raise ValueError("Only ValueType.TEXT properties are supported.")

    def compute(self, value1: Any, value2: Any, property_: Property) -> float:  # noqa: ARG002
        """Compute edit distance between two string property values."""
        return normalized_edit_distance(str(value1), str(value2))


class PropertyDistance:
    """Property distance class."""

    def check_constraints(self, property_: Property) -> None:
        """Check constraints for property distance."""
        raise NotImplementedError

    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> list[float]:
        """Compute distance between two entities for a given property id."""
        raise NotImplementedError

    def __call__(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> list[float]:
        """Compute distance between two entities for a given property id."""
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

    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> list[float]:
        """Compute distance between two entities for a given property id."""
        if property_ not in gt_entity or property_ not in pred_entity:
            return [1]
        return [self.element_distance(gt_entity[property_][0], pred_entity[property_][0], property_)]


class SetPropertyDistance(PropertyDistance):
    """Set distance class."""

    def __init__(self, element_distance: ElementDistance):
        """Initialize set distance."""
        self.element_distance = element_distance

    def check_constraints(self, property_: Property) -> None:  # noqa: ARG002
        """Check constraints for set distance."""
        return None

    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> list[float]:
        """Compute the average distance between two sets of values for a given property."""
        if property_ not in gt_entity or property_ not in pred_entity:
            return [1]

        gt_values = gt_entity[property_]
        pred_values = pred_entity[property_]

        distances = np.zeros((len(gt_values), len(pred_values)))
        for i, value1 in enumerate(gt_values):
            for j, value2 in enumerate(pred_values):
                distances[i, j] = self.element_distance(value1, value2, property_)
        matched_indices = match_items(distances)
        return list(distances[matched_indices.left_ind, matched_indices.right_ind])


class PropertyScore:
    """Property score class."""

    def __init__(self, property_distance: PropertyDistance):
        """Initialize property score."""
        self.property_distance = property_distance

    def __call__(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> list[float]:
        """Compute the score for a given property."""
        distances = self.property_distance(gt_entity, pred_entity, property_)
        return [1.0 - d for d in distances]


class EntityDistance:
    """Distance between two entities."""

    def __init__(
        self,
        property_schema: PropertySchema,
        property_to_distance_function: dict[str, PropertyDistance],
        weights: np.ndarray | None = None,
    ):
        """Initialize entity distance."""
        self.property_to_distance_function = property_to_distance_function
        self.property_schema = property_schema
        if weights is None:
            weights = np.ones(len(property_to_distance_function))
            weights /= weights.sum()
        self.weights = weights

    def __call__(self, gt_entity: Entity, pred_entity: Entity) -> float:
        """Compute distance between two entities."""
        distances = np.zeros(len(self.property_to_distance_function))
        for i, (property_id, property_distance) in enumerate(self.property_to_distance_function.items()):
            distances[i] = np.mean(
                property_distance(gt_entity, pred_entity, self.property_schema.properties[property_id])
            )
        return self.weights.dot(distances)  # type: ignore


#######################################################################################################################


def jaccard_distance(tokens1: list[int], tokens2: list[int]) -> float:
    """Jaccard distance."""
    set1 = set(tokens1)
    set2 = set(tokens2)
    return 1 - len(set1.intersection(set2)) / len(set1.union(set2))


def token_distance_between_strings(value1: str, value2: str, encoder: tiktoken.Encoding) -> float:
    """Compute token distance between two string values."""
    tokens1 = encoder.encode(normalize_string(value1))
    tokens2 = encoder.encode(normalize_string(value2))
    return jaccard_distance(tokens1, tokens2)


def embeddings_distance_between_strings(value1: str, value2: str, model: SentenceTransformer) -> float:
    """Compute embedding distance between two string values."""
    embeds = model.encode([value1, value2])
    return max(0.0, float(cosine(embeds[0], embeds[1])))


def normalized_edit_distance(value1: str, value2: str) -> float:
    """Compute edit distance between 2 strings and normalize it by length of the longest string."""
    return abs(nltk.edit_distance(value1, value2)) / (max(len(value1), len(value2)))
