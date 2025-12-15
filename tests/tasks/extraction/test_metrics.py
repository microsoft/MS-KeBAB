# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false

"""Tests for the extraction metrics module."""

from __future__ import annotations

from typing import Any

import numpy as np
import tiktoken
from kebab.contracts.document import Document, DocumentSchema
from kebab.contracts.entity import Entity, PropertySchema
from kebab.tasks.metrics.extraction.aesop.calculator import (
    ValueAveragedAesopConfig,
    ValueAveragedAesopMetricCalculator,
    make_default_value_averaged_aesop_config,
)
from kebab.tasks.metrics.extraction.aesop.distances import (
    BinaryMatchDistance,
    EditDistance,
    EmbeddingDistance,
    EntityDistance,
    PropertyScore,
    ReferenceResolvingDistance,
    SetPropertyDistance,
    SingleValuePropertyDistance,
    TokenDistance,
)
from kebab.tasks.metrics.extraction.aesop.metric_helpers import (
    EntityMatcher,
    MatchedEntitiesScorer,
    MetricsAccumulator,
    MetricsComputer,
    compute_properties_union,
)
from kebab.tasks.metrics.extraction.calculator import ExtractionOutput
from kebab.tasks.metrics.extraction.utils import normalize_string
from sentence_transformers import SentenceTransformer


embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
encoder = tiktoken.get_encoding("cl100k_base")
token_score = PropertyScore(SingleValuePropertyDistance(TokenDistance(encoder=encoder)))
token_score_ref = PropertyScore(SingleValuePropertyDistance(ReferenceResolvingDistance(TokenDistance(encoder=encoder))))
embedding_score = PropertyScore(SetPropertyDistance(EmbeddingDistance(model=embed_model)))
embedding_score_ref = PropertyScore(SetPropertyDistance(ReferenceResolvingDistance(EmbeddingDistance(model=embed_model))))
str_score = PropertyScore(SingleValuePropertyDistance(BinaryMatchDistance()))
str_score_ref = PropertyScore(SingleValuePropertyDistance(ReferenceResolvingDistance(BinaryMatchDistance())))
edit_score = PropertyScore(SingleValuePropertyDistance(EditDistance()))
edit_score_ref = PropertyScore(SingleValuePropertyDistance(ReferenceResolvingDistance(EditDistance())))
set_token_score = PropertyScore(SetPropertyDistance(TokenDistance(encoder=encoder)))
set_token_score_ref = PropertyScore(SetPropertyDistance(ReferenceResolvingDistance(TokenDistance(encoder=encoder))))


def name_entity_distance(property_schema: PropertySchema) -> EntityDistance:
    return EntityDistance(
        property_schema,
        {"name": (SingleValuePropertyDistance(TokenDistance()), 1),
         "type": (SetPropertyDistance(TokenDistance()), 0),
         "definitions": (SetPropertyDistance(TokenDistance()), 0),},
        default_property_distance=SetPropertyDistance(ReferenceResolvingDistance(TokenDistance(encoder=encoder))),
        default_property_weight=0,
    )


def to_entities(properties: list[dict[str, Any]]) -> list[Entity]:
    for property_dict in properties:
        for k, v in property_dict.items():
            if not isinstance(v, list):
                property_dict[k] = [v]
    entities = [
        Entity.from_dict({"entity_id": str(idx), "properties": property_dict})
        for idx, property_dict in enumerate(properties)
    ]

    name_to_entity_id = {}
    for entity in entities:
        for name in entity.properties["name"]:
            name_to_entity_id[name] = entity.entity_id

    for entity in entities:
        for property_id, values in entity.properties.items():
            if property_id in ("name", "type", "definitions"):
                continue
            new_values = set()
            for value in values:
                new_values.add(name_to_entity_id[value])
            entity.properties[property_id] = list(new_values)
    return entities

def dummy_doc() -> Document:
    return Document.from_dict(
        data={"document_id": "1", "schema_id": "plain_text", "data": {"text": "dummy text"}},
        schemas={
            "plain_text": DocumentSchema.from_dict(
                {"schema_id": "plain_text", "fields": [{"field_id": "text", "description": "text"}]}
            )
        },
    )


def to_extraction_output(entities: list[Entity]) -> list[ExtractionOutput]:
    return [ExtractionOutput(dummy_doc(), entities)]


def test_normalize_string():
    assert normalize_string("ProJect") == "project"
    assert normalize_string("  Project  ") == "project"
    assert normalize_string("kebab project") == "kebab project"
    assert normalize_string("KeBAB\tProject") == "kebab project"
    assert normalize_string("KEBAB\nProject") == "kebab project"


def test_match_entities_in_different_order(property_schema: PropertySchema):
    ground_truth = to_entities([{"name": "London"}, {"name": "United Kingdom"}, {"name": "France"}])

    predictions = to_entities([{"name": "France"}, {"name": "London"}, {"name": "United Kingdom"}])

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(name_entity_distance(property_schema))

    np.testing.assert_allclose(matched_pair.left_ind, [0, 1, 2])
    np.testing.assert_allclose(matched_pair.right_ind, [1, 2, 0])


def test_match_entities_with_larger_ground_truth(property_schema: PropertySchema):
    ground_truth = to_entities(
        [
            {"name": "United Kingdom"},
            {"name": "London"},
            {"name": "France"},
            {"name": "Paris"},
        ]
    )

    predictions = to_entities([{"name": "London"}])

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(name_entity_distance(property_schema))

    np.testing.assert_allclose(matched_pair.left_ind, [1])
    np.testing.assert_allclose(matched_pair.right_ind, [0])
    np.testing.assert_allclose(matched_pair.distances, [[1], [0], [1], [1]])


def test_match_entities_with_larger_predictions(property_schema: PropertySchema):
    ground_truth = to_entities([{"name": "London"}])

    predictions = to_entities(
        [
            {"name": "United Kingdom"},
            {"name": "London"},
            {"name": "France"},
            {"name": "Paris"},
        ]
    )

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(name_entity_distance(property_schema))

    np.testing.assert_allclose(matched_pair.left_ind, [0])
    np.testing.assert_allclose(matched_pair.right_ind, [1])
    np.testing.assert_allclose(matched_pair.distances, [[1, 0, 1, 1]])


def test_match_multiple_different_entities(property_schema: PropertySchema):
    ground_truth = to_entities([{"name": "United Kingdom"}, {"name": "London"}, {"name": "Istanbul"}, {"name": "UK"}])
    predictions = to_entities([{"name": "Paris"}, {"name": "Istanbul"}, {"name": "France"}, {"name": "London"}])

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(name_entity_distance(property_schema), threshold=0.5)

    np.testing.assert_allclose(matched_pair.left_ind, [1, 2])
    np.testing.assert_allclose(matched_pair.right_ind, [3, 1])
    np.testing.assert_allclose(matched_pair.left_unmatched, [0, 3])
    np.testing.assert_allclose(matched_pair.right_unmatched, [2, 0])


def test_match_entities_with_no_overlap(property_schema: PropertySchema):
    ground_truth = to_entities([{"name": "United Kingdom"}, {"name": "London"}])
    predictions = to_entities([{"name": "France"}, {"name": "Paris"}])

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(name_entity_distance(property_schema), threshold=0.5)

    # matching score is lower than the threshold so no entities are matched
    np.testing.assert_allclose(matched_pair.left_ind, [])
    np.testing.assert_allclose(matched_pair.right_ind, [])
    np.testing.assert_allclose(matched_pair.left_unmatched, [0, 1])
    np.testing.assert_allclose(matched_pair.right_unmatched, [1, 0])

    properties_union = compute_properties_union(ground_truth, predictions, matched_pair)

    assert not properties_union

    entities_scorer = MatchedEntitiesScorer(ground_truth, predictions, matched_pair, property_schema)

    evaluated_property_metrics = entities_scorer.score(token_score, "name")

    # no entities are matched so no scores are computed
    assert not evaluated_property_metrics.relevant_pair_scores
    assert not evaluated_property_metrics.unmatched_count
    np.testing.assert_allclose(evaluated_property_metrics.property_value_count_gt, [0, 0])


def test_entity_matches_with_low_matching_scores_should_not_be_evaluated(property_schema: PropertySchema):
    ground_truth = to_extraction_output(
        to_entities([{"name": "United Kingdom"}, {"name": "London"}, {"name": "Istanbul"}])
    )
    predictions = to_extraction_output(to_entities([{"name": "France"}, {"name": "Paris"}]))

    config = make_default_value_averaged_aesop_config(property_schema, matching_threshold=0.5, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    # no entities are matched
    assert metrics["matched_count"] == 0
    assert metrics["unmatched_counts"]["pairs"] == 2
    assert metrics["unmatched_fractions"]["pairs"] == 1.0

    # extra entity in the ground truth
    assert metrics["unmatched_counts"]["extra_gt_entities"] == 1
    assert metrics["unmatched_counts"]["extra_pred_entities"] == 0
    assert metrics["unmatched_fractions"]["extra_gt_entities"] == 1 / 3
    assert metrics["unmatched_fractions"]["extra_pred_entities"] == 0.0

    # no property metrics are computed
    assert not metrics.get("property_precision")
    assert not metrics.get("property_recall")


def test_match_entities_with_different_thresholds(property_schema: PropertySchema):
    ground_truth = to_extraction_output(
        to_entities([{"name": "South Korea", "type": "country"}, {"name": "London"}, {"name": "Istanbul"}])
    )
    predictions = to_extraction_output(
        to_entities(
            [
                {"name": "North Korea", "type": "country"},
                {"name": "London"},
                {"name": "France"},
            ]
        )
    )

    # threshold 1.0
    config = make_default_value_averaged_aesop_config(property_schema, matching_threshold=1.0, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    # no extra entities in the ground truth or predictions
    assert metrics["unmatched_counts"]["extra_gt_entities"] == 0
    assert metrics["unmatched_counts"]["extra_pred_entities"] == 0
    assert metrics["unmatched_fractions"]["extra_pred_entities"] == 0.0
    assert metrics["unmatched_fractions"]["extra_gt_entities"] == 0.0

    # with a threshold of 1, all entities are matched
    assert metrics["matched_count"] == 3
    assert metrics["unmatched_counts"]["pairs"] == 0
    assert metrics["unmatched_fractions"]["pairs"] == 0

    # threshold 0.5
    config = make_default_value_averaged_aesop_config(property_schema, matching_threshold=0.5, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    # with a threshold of 0.5, the first two entities are matched
    assert metrics["matched_count"] == 2
    assert metrics["unmatched_counts"]["pairs"] == 1
    assert metrics["unmatched_fractions"]["pairs"] == 1 / 3
    assert metrics["unmatched_fractions"]["extra_pred_entities"] == 0.0
    assert metrics["unmatched_fractions"]["extra_gt_entities"] == 0.0

    # threshold 0.0
    config = make_default_value_averaged_aesop_config(property_schema, matching_threshold=0.0, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    # with a threshold of 0, one entity is matched

    assert metrics["matched_count"] == 1
    assert metrics["unmatched_counts"]["pairs"] == 2
    assert metrics["unmatched_fractions"]["pairs"] == 2 / 3
    assert metrics["unmatched_fractions"]["extra_pred_entities"] == 0.0
    assert metrics["unmatched_fractions"]["extra_gt_entities"] == 0.0


def assert_score_dicts_close(left: dict, right: dict):
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert value.keys() == right[key].keys()
        for subkey in value:
            np.testing.assert_allclose(value[subkey], right[key][subkey])


def test_compute_scores_of_target_entities_with_themselves(property_schema: PropertySchema):
    ground_truth = to_entities([{"name": "London", "type": "city", "definitions": "Capital of the United Kindgom.", "part of": "United Kingdom"},
                                {"name": "United Kingdom", "type": "country", "definitions": "A country in Europe."}])
    predictions = ground_truth.copy()

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(name_entity_distance(property_schema))

    print(matched_pair)

    properties_union = compute_properties_union(ground_truth, predictions, matched_pair)

    assert properties_union == {"name", "type", "definitions", "part of"}

    property_to_score = {
        "name": token_score,
        "type": set_token_score,
        "definitions": set_token_score,
        "part of": token_score_ref,
    }

    for key in properties_union:
        entities_scorer = MatchedEntitiesScorer(ground_truth, predictions, matched_pair, property_schema)
        evaluated_property_metrics = entities_scorer.score(property_to_score[key], key)

        assert_score_dicts_close(evaluated_property_metrics.relevant_pair_scores, {0: {0: [1.0]}})
        np.testing.assert_allclose(evaluated_property_metrics.unmatched_count, 0)
        np.testing.assert_allclose(evaluated_property_metrics.property_value_count_gt, [1])


def test_evaluate_target_entities_with_themselves(property_schema: PropertySchema):
    ground_truth = to_extraction_output(
        to_entities([{"name": "London", "type": "city", "definitions": ["Capital of the United Kindgom."]}])
    )
    predictions = ground_truth.copy()

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    assert metrics["property_precision"]["name"] == metrics["property_precision"]["type"] == 1.0
    assert metrics["property_recall"]["name"] == metrics["property_recall"]["type"] == 1.0


def test_evaluate_target_entities_with_more_predictions(property_schema: PropertySchema):
    ground_truth = to_extraction_output(to_entities([{"name": "London"}, {"name": "Istanbul"}]))

    predictions = to_extraction_output(
        to_entities(
            [
                {"name": "United Kindgom"},
                {"name": "London"},
                {"name": "Istanbul"},
                {"name": "France"},
            ]
        )
    )

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    assert metrics["matched_count"] == 2
    assert metrics["unmatched_counts"]["extra_pred_entities"] == 2
    assert metrics["unmatched_counts"]["extra_gt_entities"] == 0
    assert metrics["unmatched_fractions"]["pairs"] == 0
    assert metrics["unmatched_fractions"]["extra_gt_entities"] == 0
    assert metrics["unmatched_fractions"]["extra_pred_entities"] == 0.5

    assert metrics["property_precision"]["name"] == 1.0
    assert metrics["property_recall"]["name"] == 1.0


def test_evaluate_target_entities_with_more_ground_truth(property_schema: PropertySchema):
    ground_truth = to_extraction_output(
        to_entities(
            [
                {"name": "United Kindgom"},
                {"name": "London"},
                {"name": "Istanbul"},
                {"name": "France"},
            ]
        )
    )

    predictions = to_extraction_output(to_entities([{"name": "London"}, {"name": "Istanbul"}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    assert metrics["matched_count"] == 2
    assert metrics["unmatched_counts"]["extra_pred_entities"] == 0
    assert metrics["unmatched_counts"]["extra_gt_entities"] == 2
    assert metrics["unmatched_fractions"]["pairs"] == 0
    assert metrics["unmatched_fractions"]["extra_gt_entities"] == 0.5
    assert metrics["unmatched_fractions"]["extra_pred_entities"] == 0

    assert metrics["property_precision"]["name"] == 1.0
    assert metrics["property_recall"]["name"] == 1.0


def test_evaluate_set_properties_with_same_values(property_schema: PropertySchema):
    ground_truth = to_extraction_output(to_entities([{"name": "London", "type": ["city"]}]))
    predictions = to_extraction_output(to_entities([{"name": "London", "type": ["city"]}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    assert metrics["property_precision"]["name"] == metrics["property_precision"]["type"] == 1.0
    assert metrics["property_recall"]["name"] == metrics["property_recall"]["type"] == 1.0


def compute_intermediate_metrics(
    ground_truth: list[ExtractionOutput], predictions: list[ExtractionOutput], config: ValueAveragedAesopConfig
) -> MetricsAccumulator:
    metrics_accumulator = MetricsAccumulator()
    for pred, gt in zip(predictions, ground_truth, strict=True):
        entity_matcher = EntityMatcher(gt.entities, pred.entities)
        matched_pairs = entity_matcher.match(config.matching_score_function, config.matching_threshold)
        metrics_computer = MetricsComputer(gt.entities, pred.entities, config.property_schema)
        metrics, _ = metrics_computer.compute_bipartite_metrics(matched_pairs, config.property_score_functions)
        metrics_accumulator.update(metrics)
    return metrics_accumulator


def test_evaluate_set_properties_with_different_values(property_schema: PropertySchema):
    ground_truth = to_extraction_output(to_entities([{"type": ["project", "team"]}]))
    predictions = to_extraction_output(to_entities([{"type": ["project"]}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)

    metrics_accumulator = compute_intermediate_metrics(ground_truth, predictions, config)

    metrics = metrics_accumulator.accumulate_metrics()

    assert metrics_accumulator.total_gt_property_value_counts_per_doc["type"] == [2]
    assert metrics["property_precision"]["type"] == 1.0
    assert metrics["property_recall"]["type"] == 0.5


def test_evaluate_set_properties_with_more_predictions(property_schema: PropertySchema):
    ground_truth = to_extraction_output(to_entities([{"type": ["project", "team"]}]))
    predictions = to_extraction_output(
        to_entities([{"type": ["project", "team", "organization", "company", "location"]}])
    )

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)

    metrics_accumulator = compute_intermediate_metrics(ground_truth, predictions, config)

    metrics = metrics_accumulator.accumulate_metrics()

    assert metrics_accumulator.total_gt_property_value_counts_per_doc["type"] == [2]
    assert metrics["property_precision"]["type"] == 0.4
    assert metrics["property_recall"]["type"] == 1.0
    assert metrics_accumulator.total_unmatched_counts_per_doc["type"] == 3


def test_evaluate_set_properties_with_more_ground_truth(property_schema: PropertySchema):
    ground_truth = to_extraction_output(to_entities([{"type": ["project", "team", "organization"]}]))
    predictions = to_extraction_output(to_entities([{"type": ["project", "team"]}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)

    metrics_accumulator = compute_intermediate_metrics(ground_truth, predictions, config)

    metrics = metrics_accumulator.accumulate_metrics()

    assert metrics_accumulator.total_gt_property_value_counts_per_doc["type"] == [3]
    assert metrics["property_precision"]["type"] == 1.0
    np.testing.assert_allclose(metrics["property_recall"]["type"], 0.66666666)
    assert metrics_accumulator.total_unmatched_counts_per_doc["type"] == 0


def test_evaluate_set_properties_in_different_order_and_normalized(property_schema: PropertySchema):
    ground_truth = to_extraction_output(
        to_entities(
            [{"alternative names": ["Microsoft Research", "MSRC", "Microsoft Research Cambridge", "MSR Cambridge"]}]
        )
    )
    predictions = to_extraction_output(to_entities([{"alternative names": ["microsoft research cambridge"]}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)

    metrics_accumulator = compute_intermediate_metrics(ground_truth, predictions, config)

    metrics = metrics_accumulator.accumulate_metrics()

    assert metrics_accumulator.total_gt_property_value_counts_per_doc["alternative names"] == [4]
    assert metrics["property_precision"]["alternative names"] == 1.0
    assert metrics["property_recall"]["alternative names"] == 0.25


def test_evaluate_missing_properties_in_prediction(property_schema: PropertySchema):
    ground_truth = to_extraction_output(to_entities([{"name": "Steven", "type": "person"}]))
    predictions = to_extraction_output(to_entities([{"name": "Steven"}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    assert metrics["property_precision"]["name"] == 1.0
    assert metrics["property_recall"]["name"] == 1.0
    assert metrics["property_precision"]["type"] is None
    assert metrics["property_recall"]["type"] == 0


def test_evaluate_more_properties_in_prediction(property_schema: PropertySchema):
    ground_truth = to_extraction_output(to_entities([{"name": "Steven"}]))
    predictions = to_extraction_output(to_entities([{"name": "Steven", "type": "person"}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    assert metrics["property_precision"]["name"] == 1.0
    assert metrics["property_recall"]["name"] == 1.0
    assert metrics["property_precision"]["type"] == 0
    assert metrics["property_recall"]["type"] is None


def test_edit_score(property_schema: PropertySchema):
    ground_truth = to_entities([{"team": "Marketing"}])
    predictions = to_entities([{"team": "Strategic Marketing"}])
    value_matching_record = edit_score(ground_truth[0], predictions[0], property_schema.properties["team"])
    assert value_matching_record.matched_scores[0] > 0


def test_aggregate_across_documents_simple(property_schema: PropertySchema):
    ground_truth1 = to_extraction_output(to_entities([{"name": "John", "type": "person"}]))
    predictions1 = to_extraction_output(to_entities([{"name": "John"}]))

    ground_truth2 = to_extraction_output(to_entities([{"name": "London"}]))
    predictions2 = to_extraction_output(to_entities([{"name": "London"}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)

    metrics_accumulator = compute_intermediate_metrics(ground_truth1, predictions1, config)

    metrics_accumulator2 = compute_intermediate_metrics(ground_truth2, predictions2, config)

    metrics_accumulator.update(metrics_accumulator2)

    metrics = metrics_accumulator.accumulate_metrics()

    assert len(metrics_accumulator.total_scores_per_doc["name"]) == 2
    assert len(metrics_accumulator.total_gt_property_value_counts_per_doc["name"]) == 2
    assert metrics_accumulator.total_unmatched_counts_per_doc["name"] == 0

    assert len(metrics_accumulator.total_scores_per_doc["type"]) == 1
    assert len(metrics_accumulator.total_gt_property_value_counts_per_doc["type"]) == 1
    assert metrics_accumulator.total_unmatched_counts_per_doc["type"] == 0

    assert not metrics["property_precision"]["type"]
    assert metrics["property_recall"]["type"] == 0.0
    assert metrics["property_precision"]["name"] == 1.0
    assert metrics["property_recall"]["name"] == 1.0


def test_evaluate_entities_with_different_properties(property_schema: PropertySchema):
    ground_truth = to_extraction_output(
        to_entities(
            [
                {
                    "name": "Cliff Meltzer",
                    "type": "person",
                    "job title": "CEO and President",
                    "team": "Executive Team",
                    "employer": "Digital Fountain",
                },
                {
                    "name": "Michael Luby",
                    "type": "person",
                    "job title": "CTO and Co-Founder",
                    "team": "Executive Team",
                    "employer": "Digital Fountain",
                },
                {
                    "name": "Amin Shokrollahi",
                    "type": "person",
                    "job title": "Chief Scientist",
                    "team": "Executive Team",
                    "employer": "Digital Fountain",
                },
            ]
        )
    )
    predictions = to_extraction_output(
        to_entities(
            [
                {
                    "name": "Cliff Meltzer",
                    "type": "person",
                    "job title": "CEO and President",
                    "team": "Executive Team",
                    "company": "Digital Fountain",
                },
                {
                    "name": "Michael Luby",
                    "type": "person",
                    "job title": "CTO and Co-Founder",
                    "team": "Executive Team",
                    "company": "Digital Fountain",
                },
                {
                    "name": "Amin Shokrollahi",
                    "type": "person",
                    "job title": "Chief Scientist",
                    "team": "Executive Team",
                    "company": "Digital Fountain",
                },
            ]
        )
    )

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    assert not metrics["property_precision"]["employer"]
    assert metrics["property_recall"]["employer"] == 0.0

    assert metrics["property_precision"]["company"] == 0.0
    assert not metrics["property_recall"]["company"]

    assert metrics["property_precision"]["type"] == 1.0
    assert metrics["property_recall"]["type"] == 1.0
    assert metrics["property_precision"]["name"] == 1.0
    assert metrics["property_recall"]["name"] == 1.0
    assert metrics["property_precision"]["job title"] == 1.0
    assert metrics["property_recall"]["job title"] == 1.0
    assert metrics["property_precision"]["team"] == 1.0
    assert metrics["property_recall"]["team"] == 1.0


def test_aggregate_across_documents_with_multiple_property_values(property_schema: PropertySchema):
    ground_truth1 = to_extraction_output(
        to_entities([{"name": "Alexandria", "type": ["project"], "definitions": ["amazing team"]}])
    )
    predictions1 = to_extraction_output(
        to_entities(
            [{"name": "Alexandria", "type": ["project", "team"], "definitions": ["great project", "amazing team"]}]
        )
    )

    ground_truth2 = to_extraction_output(
        to_entities([{"name": "Infer.Net", "type": ["software"], "definitions": ["software definition"]}])
    )
    predictions2 = to_extraction_output(
        to_entities(
            [
                {
                    "name": "Infer.Net",
                    "type": ["software", "team", "organization", "company", "location"],
                    "definitions": ["software definition", "software definition1"],
                }
            ]
        )
    )

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    metrics_accumulator = compute_intermediate_metrics(ground_truth1, predictions1, config)
    metrics_accumulator2 = compute_intermediate_metrics(ground_truth2, predictions2, config)
    metrics_accumulator.update(metrics_accumulator2)

    metrics = metrics_accumulator.accumulate_metrics()

    # Found all property values in the ground truth
    np.testing.assert_allclose(metrics["property_recall"]["name"], 1.0)
    np.testing.assert_allclose(metrics["property_recall"]["type"], 1.0)
    np.testing.assert_allclose(metrics["property_recall"]["definitions"], 1.0)

    # Each extra property value in the prediction is counted as a false positive
    np.testing.assert_allclose(metrics["property_precision"]["name"], 1.0)
    np.testing.assert_allclose(metrics["property_precision"]["type"], 2 / 7)
    np.testing.assert_allclose(metrics["property_precision"]["definitions"], 0.5)


def test_aggregate_across_documents_with_multiple_entities(property_schema: PropertySchema):
    ground_truth1 = to_extraction_output(
        to_entities([{"name": "Alexandria", "type": ["project"]}, {"name": "FNKE", "type": ["project"]}])
    )
    predictions1 = to_extraction_output(
        to_entities([{"name": "Alexandria", "type": ["project", "team"]}, {"name": "FNKE", "type": ["project"]}])
    )

    ground_truth2 = to_extraction_output(to_entities([{"type": ["software"], "definitions": ["software definition"]}]))
    predictions2 = to_extraction_output(to_entities([{"type": ["software"], "definitions": ["software definition"]}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    metrics_accumulator = compute_intermediate_metrics(ground_truth1, predictions1, config)
    metrics_accumulator2 = compute_intermediate_metrics(ground_truth2, predictions2, config)
    metrics_accumulator.update(metrics_accumulator2)
    metrics = metrics_accumulator.accumulate_metrics()

    assert len(metrics_accumulator.total_scores_per_doc["name"]) == 2
    assert metrics_accumulator.total_gt_property_value_counts_per_doc["name"] == [1, 1]
    assert metrics_accumulator.total_unmatched_counts_per_doc["name"] == 0

    assert len(metrics_accumulator.total_scores_per_doc["type"]) == 3
    assert metrics_accumulator.total_gt_property_value_counts_per_doc["type"] == [1, 1, 1]
    assert metrics_accumulator.total_unmatched_counts_per_doc["type"] == 1

    assert len(metrics_accumulator.total_scores_per_doc["definitions"]) == 1
    assert metrics_accumulator.total_gt_property_value_counts_per_doc["definitions"] == [1]
    assert metrics_accumulator.total_unmatched_counts_per_doc["definitions"] == 0

    np.testing.assert_allclose(metrics["property_recall"]["name"], 1.0)
    np.testing.assert_allclose(metrics["property_recall"]["type"], 1.0)
    np.testing.assert_allclose(metrics["property_recall"]["definitions"], 1.0)

    np.testing.assert_allclose(metrics["property_precision"]["name"], 1.0)
    np.testing.assert_allclose(metrics["property_precision"]["type"], 0.75)
    np.testing.assert_allclose(metrics["property_precision"]["definitions"], 1.0)


def test_aggregate_across_documents_with_multiple_entities_and_missing_properties_in_predictions(
    property_schema: PropertySchema,
):
    ground_truth1 = to_extraction_output(
        to_entities([{"name": "Alexandria", "type": ["project"]}, {"name": "FNKE", "type": ["project"]}])
    )
    predictions1 = to_extraction_output(to_entities([{"name": "Alexandria"}, {"name": "FNKE"}]))

    ground_truth2 = to_extraction_output(to_entities([{"name": "Phi2", "definitions": ["small language model"]}]))
    predictions2 = to_extraction_output(to_entities([{"name": "Phi2"}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    metrics_accumulator = compute_intermediate_metrics(ground_truth1, predictions1, config)
    metrics_accumulator2 = compute_intermediate_metrics(ground_truth2, predictions2, config)
    metrics_accumulator.update(metrics_accumulator2)
    metrics = metrics_accumulator.accumulate_metrics()

    assert len(metrics_accumulator.total_scores_per_doc["name"]) == 3
    assert metrics_accumulator.total_gt_property_value_counts_per_doc["name"] == [1, 1, 1]
    assert metrics_accumulator.total_unmatched_counts_per_doc["name"] == 0

    assert len(metrics_accumulator.total_scores_per_doc["type"]) == 2
    assert metrics_accumulator.total_gt_property_value_counts_per_doc["type"] == [1, 1]
    assert metrics_accumulator.total_unmatched_counts_per_doc["type"] == 0

    assert len(metrics_accumulator.total_scores_per_doc["definitions"]) == 1
    assert metrics_accumulator.total_gt_property_value_counts_per_doc["definitions"] == [1]
    assert metrics_accumulator.total_unmatched_counts_per_doc["definitions"] == 0

    assert metrics["property_precision"]["name"] == 1.0

    # no type or definitions in the predictions so precision can't be computed
    assert not metrics["property_precision"]["type"]
    assert not metrics["property_precision"]["definitions"]

    assert metrics["property_recall"]["name"] == 1.0
    assert metrics["property_recall"]["type"] == 0.0
    assert metrics["property_recall"]["definitions"] == 0.0


def test_aggregate_across_documents_with_multiple_entities_and_missing_properties_in_ground_truth(
    property_schema: PropertySchema,
):
    ground_truth1 = to_extraction_output(to_entities([{"name": "Alexandria"}, {"name": "FNKE"}]))
    predictions1 = to_extraction_output(
        to_entities([{"name": "Alexandria", "type": ["project"]}, {"name": "FNKE", "type": ["project"]}])
    )

    ground_truth2 = to_extraction_output(to_entities([{"name": "Phi2"}]))
    predictions2 = to_extraction_output(to_entities([{"name": "Phi2", "definitions": ["small language model"]}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    metrics_accumulator = compute_intermediate_metrics(ground_truth1, predictions1, config)
    metrics_accumulator2 = compute_intermediate_metrics(ground_truth2, predictions2, config)
    metrics_accumulator.update(metrics_accumulator2)
    metrics = metrics_accumulator.accumulate_metrics()

    assert len(metrics_accumulator.total_scores_per_doc["name"]) == 3
    assert metrics_accumulator.total_gt_property_value_counts_per_doc["name"] == [1, 1, 1]
    assert metrics_accumulator.total_unmatched_counts_per_doc["name"] == 0

    assert len(metrics_accumulator.total_scores_per_doc["type"]) == 1
    assert metrics_accumulator.total_gt_property_value_counts_per_doc["type"] == [0, 0]
    assert metrics_accumulator.total_unmatched_counts_per_doc["type"] == 0

    assert len(metrics_accumulator.total_scores_per_doc["definitions"]) == 1
    assert metrics_accumulator.total_gt_property_value_counts_per_doc["definitions"] == [0]
    assert metrics_accumulator.total_unmatched_counts_per_doc["definitions"] == 0

    assert metrics["property_precision"]["name"] == 1.0
    assert metrics["property_precision"]["type"] == 0.0
    assert metrics["property_precision"]["definitions"] == 0.0

    assert metrics["property_recall"]["name"] == 1.0

    # no types or definitions in the ground truth so recall can't be computed
    assert not metrics["property_recall"]["type"]
    assert not metrics["property_recall"]["definitions"]


def test_evaluate_same_property_across_documents(property_schema: PropertySchema):
    ground_truth1 = to_extraction_output(to_entities([{"type": ["person", "lead"]}]))
    predictions1 = to_extraction_output(to_entities([{"type": ["person", "lead"]}]))

    ground_truth2 = to_extraction_output(to_entities([{"type": ["organization"]}]))
    predictions2 = to_extraction_output(to_entities([{"type": ["company"]}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    metrics_accumulator = compute_intermediate_metrics(ground_truth1, predictions1, config)
    metrics_accumulator2 = compute_intermediate_metrics(ground_truth2, predictions2, config)
    metrics_accumulator.update(metrics_accumulator2)
    metrics = metrics_accumulator.accumulate_metrics()

    np.testing.assert_allclose(metrics["property_precision"]["type"], 0.66666666)
    np.testing.assert_allclose(metrics["property_recall"]["type"], 0.66666666)


def test_unmatched_entities_in_predictions_should_not_be_evaluated(property_schema: PropertySchema):
    ground_truth = to_extraction_output(to_entities([{"name": "Microsoft Research"}, {"name": "Alexandria"}]))
    predictions = to_extraction_output(
        to_entities([{"name": "Microsoft Research"}, {"name": "Alexandria"}, {"name": "FNKE", "type": "project"}])
    )

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    assert metrics["matched_count"] == 2
    assert metrics["unmatched_counts"]["extra_gt_entities"] == 0
    assert metrics["unmatched_counts"]["extra_pred_entities"] == 1
    assert metrics["unmatched_counts"]["pairs"] == 0

    assert metrics["property_precision"]["name"] == 1
    assert metrics["property_recall"]["name"] == 1
    assert "type" not in metrics["property_precision"]
    assert "type" not in metrics["property_recall"]


def test_unmatched_entities_in_gt_should_not_be_evaluated(property_schema: PropertySchema):
    ground_truth = to_extraction_output(
        to_entities([{"name": "Microsoft Research"}, {"name": "Alexandria"}, {"name": "FNKE", "type": "project"}])
    )
    predictions = to_extraction_output(to_entities([{"name": "Microsoft Research"}, {"name": "Alexandria"}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    assert metrics["matched_count"] == 2
    assert metrics["unmatched_counts"]["extra_gt_entities"] == 1
    assert metrics["unmatched_counts"]["extra_pred_entities"] == 0
    assert metrics["unmatched_counts"]["pairs"] == 0

    assert metrics["property_precision"]["name"] == 1
    assert metrics["property_recall"]["name"] == 1
    assert "type" not in metrics["property_precision"]
    assert "type" not in metrics["property_recall"]


def test_json_values_are_not_evaluated(property_schema: PropertySchema):
    ground_truth = to_extraction_output(
        to_entities(
            [
                {
                    "name": "Consolidated statement of income",
                    "type": "document",
                    "key points": [
                        {
                            "name": "Increase in operating revenues",
                            "Q2 1999": "$2,597 million",
                            "Q2 2000": "$4,227 million",
                            "six months 1999": "$4,875 million",
                            "six months 2000": "$7,333 million",
                        },
                        {
                            "name": "Improvement in operating income",
                            "first six months 1999": "loss of $36 million",
                            "first six months 2000": "gain of $652 million",
                        },
                    ],
                }
            ]
        )
    )
    predictions = ground_truth.copy()

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model)
    aesop_metric_calculator = ValueAveragedAesopMetricCalculator(config)
    metrics = aesop_metric_calculator.run(predictions, ground_truth)["dataset_metrics"]

    assert metrics["matched_count"] == 1
    assert metrics["unmatched_counts"]["extra_gt_entities"] == 0
    assert metrics["unmatched_counts"]["extra_pred_entities"] == 0
    assert metrics["unmatched_counts"]["pairs"] == 0

    assert metrics["property_precision"]["name"] == 1.0
    assert metrics["property_recall"]["name"] == 1.0
    assert metrics["property_precision"]["type"] == 1.0
    assert metrics["property_recall"]["type"] == 1.0


def test_final_statistics(property_schema: PropertySchema):
    ground_truth1 = to_extraction_output(to_entities([{"name": "John"}, {"name": "Alexandria"}, {"name": "FNKE"}]))
    predictions1 = to_extraction_output(to_entities([{"name": "James"}]))
    ground_truth2 = to_extraction_output(
        to_entities([{"name": "Alexandria"}, {"name": "FNKE"}, {"name": "Infer.Net"}, {"name": "Phi3"}])
    )
    predictions2 = to_extraction_output(to_entities([{"name": "Alexandria"}]))
    ground_truth3 = to_extraction_output(
        to_entities([{"name": "Alexandria"}, {"name": "FNKE"}, {"name": "Infer.Net"}, {"name": "Phi3"}])
    )
    predictions3 = to_extraction_output(to_entities([{"name": "Team"}, {"name": "Model"}]))
    ground_truth4 = to_extraction_output(to_entities([{"name": "Alexandria"}]))
    predictions4 = to_extraction_output(to_entities([{"name": "Alexandria"}, {"name": "New Alexandria"}]))

    config = make_default_value_averaged_aesop_config(property_schema, embed_model=embed_model, matching_threshold=0.5)
    metrics_accumulator = compute_intermediate_metrics(ground_truth1, predictions1, config)
    metrics_accumulator2 = compute_intermediate_metrics(ground_truth2, predictions2, config)
    metrics_accumulator.update(metrics_accumulator2)
    metrics_accumulator3 = compute_intermediate_metrics(ground_truth3, predictions3, config)
    metrics_accumulator.update(metrics_accumulator3)
    metrics_accumulator4 = compute_intermediate_metrics(ground_truth4, predictions4, config)
    metrics_accumulator.update(metrics_accumulator4)

    metrics = metrics_accumulator.accumulate_metrics()

    assert metrics["unmatched_fractions"]["pairs"] == 3 / 5
    assert metrics["unmatched_fractions"]["extra_pred_entities"] == 1 / 6
    assert metrics["unmatched_fractions"]["extra_gt_entities"] == 7 / 12
    assert metrics["matched_count"] == 2
    assert metrics["unmatched_counts"]["extra_gt_entities"] == 7
    assert metrics["unmatched_counts"]["extra_pred_entities"] == 1
    assert metrics["unmatched_counts"]["pairs"] == 3


def test_str_match_score(property_schema: PropertySchema):
    ground_truth = to_entities([{"email": "test@microsoft.com"}])
    predictions = ground_truth.copy()
    value_matching_record = str_score(ground_truth[0], predictions[0], property_schema.properties["email"])
    assert value_matching_record.matched_scores[0] == 1.0


def test_embedding_based_score(property_schema: PropertySchema):
    ground_truth = to_entities(
        [
            {
                "definitions": [
                    "A tool used for extracting topics. It is used in a branch and has been tested for extracting personal topics and for the LLM stuff."
                ]
            }
        ]
    )
    predictions = to_entities(
        [
            {
                "definitions": [
                    "A software used for extracting entities from documents in the incremental KB construction procedure. It also has a type hierarchy and can model a lot of types"
                ]
            }
        ]
    )
    value_matching_record = embedding_score(ground_truth[0], predictions[0], property_schema.properties["definitions"])
    assert value_matching_record.matched_scores[0] > 0.5
