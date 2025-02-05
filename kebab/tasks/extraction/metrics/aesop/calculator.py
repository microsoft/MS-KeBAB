# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sentence_transformers import SentenceTransformer

from kebab.contracts.entity import Entity, Property, PropertySchema
from kebab.tasks.extraction.metrics.aesop import distances
from kebab.tasks.extraction.metrics.aesop.distances import (
    BinaryMatchDistance,
    EditDistance,
    ElementDistance,
    EmbeddingDistance,
    EntityDistance,
    PropertyScore,
    PropertyScoreValue,
    SetPropertyDistance,
    SingleValuePropertyDistance,
    TokenDistance,
)
from kebab.tasks.extraction.metrics.aesop.metric_helpers import EntityMatcher, MetricsAccumulator, MetricsComputer
from kebab.tasks.extraction.metrics.calculator import ExtractionOutput, MetricCalculator, MetricConfig


@dataclass
class ValueAveragedAesopConfig(MetricConfig):
    """Value-averaged-AESOP metric configuration."""

    matching_score_function: Callable[[Entity, Entity], float]
    matching_threshold: float
    property_score_functions: dict[str, Callable[[Entity, Entity, Property], PropertyScoreValue]]
    property_schema: PropertySchema

    @staticmethod
    def from_dict(config: dict[str, Any], property_schema: PropertySchema) -> ValueAveragedAesopConfig:
        """Create value-averaged-AESOP metric configuration from dictionary."""

        def get_element_distance(element_distance_params: dict[str, Any]) -> ElementDistance:
            """Get element distance function from parameters."""
            element_distance_cls_name = element_distance_params["name"]
            return getattr(distances, element_distance_cls_name).from_dict(element_distance_params.get("params", {}))

        default_property_score = PropertyScore(
            SetPropertyDistance(get_element_distance(config["default_property_distance"]))
        )

        property_score_functions = defaultdict(lambda: default_property_score)
        for property_name, score_config in config["property_distance_functions"].items():
            property_score_functions[property_name] = PropertyScore(
                SetPropertyDistance(get_element_distance(score_config))
            )
        return ValueAveragedAesopConfig(
            matching_score_function=EntityDistance.from_dict(config),
            matching_threshold=config["matching_threshold"],
            property_score_functions=property_score_functions,
            property_schema=property_schema,
        )


class ValueAveragedAesopMetricCalculator(MetricCalculator):
    """Value-averaged-AESOP metric calculator.
    Computes AESOP metric as described in the paper: https://arxiv.org/pdf/2402.04437
    with the following modification:
    We don't average property scores on entity level to obtain entity similarities.
    Instead, we compute similarity scores for each pair of matched values of that property in the dataset.
    Then, we average these scores to obtain precision for the property.
    Finally, we average precision scores for all properties to obtain the average precision.
    Similarly, we compute recall scores for each property and average them to obtain the average recall.
    """

    def __init__(self, config: ValueAveragedAesopConfig):
        """Configure value-averaged-AESOP metric calculator."""
        self.config = config

    def run(self, prediction: list[ExtractionOutput], ground_truth: list[ExtractionOutput]) -> dict:
        """Calculate AESOP-based metrics."""
        metrics_accumulator = MetricsAccumulator()
        for pred, gt in zip(prediction, ground_truth, strict=True):
            entity_matcher = EntityMatcher(gt.entities, pred.entities)
            matched_pairs = entity_matcher.match(self.config.matching_score_function, self.config.matching_threshold)
            metrics_computer = MetricsComputer(gt.entities, pred.entities, self.config.property_schema)
            metrics_accumulator.update(
                metrics_computer.compute_bipartite_metrics(matched_pairs, self.config.property_score_functions)
            )
        return metrics_accumulator.accumulate_metrics()


def make_default_value_averaged_aesop_config(
    property_schema: PropertySchema, matching_threshold: float = 1.0, embed_model: SentenceTransformer | None = None
) -> ValueAveragedAesopConfig:
    """Create default value-averaged-AESOP metric config."""
    # Set score functions are used for properties for which we expect multiple values.
    # Score function always returns a list of scores, even if there is only one value.
    if embed_model is None:
        embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    set_embedding_score = PropertyScore(SetPropertyDistance(EmbeddingDistance()))
    set_token_score = PropertyScore(SetPropertyDistance(TokenDistance()))
    str_score = PropertyScore(SingleValuePropertyDistance(BinaryMatchDistance()))
    edit_score = PropertyScore(SetPropertyDistance(EditDistance()))
    embedding_score = PropertyScore(SingleValuePropertyDistance(EmbeddingDistance(model=embed_model)))

    property_to_score = defaultdict(lambda: set_token_score)
    property_to_score.update(
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

    return ValueAveragedAesopConfig(
        matching_score_function=EntityDistance(property_schema, {"name": (SetPropertyDistance(TokenDistance()), 1.0)}),
        matching_threshold=matching_threshold,
        property_score_functions=property_to_score,
        property_schema=property_schema,
    )
