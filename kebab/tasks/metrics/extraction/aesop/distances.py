# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# ruff: noqa: ANN401
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, override

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
    def build(cls, config: dict[str, Any], **kwargs) -> ElementDistance:
        """Build element distance from configuration dictionary and additional arguments."""
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

    @override
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for token distance."""
        if property_.data_type.value_type not in {ValueType.TEXT, ValueType.DATE}:
            raise ValueError("Only ValueType.TEXT and ValueType.DATE properties are supported.")

    @override
    def compute(self, value1: Any, value2: Any, property_: Property) -> float:
        """Compute token distance between property values of ground truth and prediction entity."""
        return token_distance_between_strings(str(value1), str(value2), self.encoder)

    @classmethod
    @override
    def build(cls, config: dict[str, str], **kwargs) -> TokenDistance:
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

    @override
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for embedding distance."""
        if property_.data_type.value_type is not ValueType.TEXT:
            raise ValueError("Only ValueType.TEXT properties are supported.")

    @override
    def compute(self, value1: Any, value2: Any, property_: Property) -> float:
        """Compute distance between string property values using sentence embeddings."""
        return embeddings_distance_between_strings(str(value1), str(value2), self.model)

    @classmethod
    @override
    def build(cls, config: dict[str, str], **kwargs) -> EmbeddingDistance:
        """Create embedding distance from dictionary."""
        return EmbeddingDistance(model=SentenceTransformer(config.get("model", cls.__default_model)))  # type: ignore


class BinaryMatchDistance(ElementDistance):
    """Binary match distance class."""

    @override
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for binary match distance."""
        return None

    @override
    def compute(self, value1: Any, value2: Any, property_: Property) -> float:
        """Computes a distance between two entities for a given property id. Binary output of 0 if the values match, 1 otherwise."""
        value1 = normalize_property_value(value1, property_)
        value2 = normalize_property_value(value2, property_)
        return 0 if value1 == value2 else 1

    @classmethod
    @override
    def build(cls, config: dict[str, Any], **kwargs) -> BinaryMatchDistance:
        """Create binary match distance from dictionary."""
        return BinaryMatchDistance()


class EditDistance(ElementDistance):
    """Edit distance class."""

    @override
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for edit distance."""
        if property_.data_type.value_type is not ValueType.TEXT:
            raise ValueError("Only ValueType.TEXT properties are supported.")

    @override
    def compute(self, value1: Any, value2: Any, property_: Property) -> float:
        """Compute edit distance between two string property values."""
        return normalized_edit_distance(str(value1), str(value2))

    @classmethod
    @override
    def build(cls, config: dict[str, Any], **kwargs) -> EditDistance:
        """Create edit distance from dictionary."""
        return EditDistance()


class TypeNameDistance(ElementDistance):
    """Type name distance class."""

    MISCELLANEOUS_TYPE_NAME = "miscellaneous"
    DEFAULT_MISCELLANEOUS_DISTANCE = 0.0

    def __init__(self, miscellaneous_distance: float | None = None):
        """Initialize type name distance.

        Args:
          miscellaneous_distance: distance value to use when one of the type names is MISCELLANEOUS_TYPE_NAME.
        """
        if miscellaneous_distance is not None:
            if miscellaneous_distance < 0 or miscellaneous_distance > 1:
                raise ValueError("Miscellaneous distance must be between 0 and 1.")
            self.miscellaneous_distance = miscellaneous_distance
        else:
            self.miscellaneous_distance = self.DEFAULT_MISCELLANEOUS_DISTANCE

    @override
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for type name distance."""
        if property_.data_type.value_type is not ValueType.TEXT:
            raise ValueError("Only ValueType.TEXT properties are supported.")

    @override
    def compute(self, value1: Any, value2: Any, property_: Property) -> float:
        """Compute type name distance between two type name property values."""
        if str(value1) == str(value2):
            return 0.0
        if self.MISCELLANEOUS_TYPE_NAME in {str(value1), str(value2)}:
            return self.miscellaneous_distance
        return 1.0

    @classmethod
    @override
    def build(cls, config: dict[str, Any], **kwargs) -> TypeNameDistance:
        """Create type name distance from dictionary."""
        return TypeNameDistance(config.get("miscellaneous_distance"))
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
