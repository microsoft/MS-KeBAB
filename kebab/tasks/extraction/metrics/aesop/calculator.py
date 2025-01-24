# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from kebab.contracts.entity import Entity, Property, PropertySchema
from kebab.tasks.extraction.metrics.aesop.distances import (
    BinaryMatchDistance,
    EditDistance,
    EmbeddingDistance,
    EntityDistance,
    PropertyScore,
    PropertyScoreValue,
    SetPropertyDistance,
    SingleValuePropertyDistance,
    TokenDistance,
)
from kebab.tasks.extraction.metrics.aesop.metric_helpers import EntityMatcher, MetricsAccumulator, MetricsComputer
from kebab.tasks.extraction.metrics.calculator import ExtractionOutput, MetricsCalculator


@dataclass
class AesopConfig:
    """AESOP metric configuration."""

    matching_score_function: Callable[[Entity, Entity], float]
    matching_threshold: float
    property_score_functions: dict[str, Callable[[Entity, Entity, Property], PropertyScoreValue]]
    property_schema: PropertySchema


class AesopMetricCalculator(MetricsCalculator):
    """AESOP metric calculator.
    Computes AESOP metric as described in the paper: https://arxiv.org/pdf/2402.04437
    with the following modification:
    We don't average property scores on entity level to obtain entity similarities.
    Instead, we compute property-level similarities across all entities in the set.
    """

    def __init__(self, config: AesopConfig):
        """Configure AESOP metric calculator."""
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


def make_default_aesop_config(
    property_schema: PropertySchema, matching_threshold: float = 1.0, embed_model: SentenceTransformer | None = None
) -> AesopConfig:
    """Create default aesop metric config."""
    # Set score functions are used for properties for which we expect multiple values.
    # Score function always returns a list of scores, even if there is only one value.
    if embed_model is None:
        embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    set_embedding_score = PropertyScore(SetPropertyDistance(EmbeddingDistance()))
    set_token_score = PropertyScore(SetPropertyDistance(TokenDistance()))
    str_score = PropertyScore(SingleValuePropertyDistance(BinaryMatchDistance()))
    edit_score = PropertyScore(SingleValuePropertyDistance(EditDistance()))
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

    return AesopConfig(
        matching_score_function=EntityDistance(property_schema, {"name": SingleValuePropertyDistance(TokenDistance())}),
        matching_threshold=matching_threshold,
        property_score_functions=property_to_score,
        property_schema=property_schema,
    )
