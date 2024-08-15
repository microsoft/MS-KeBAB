# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false
# ruff: noqa: E731, PLR2004, E711
"""Tests for the metrics module."""

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from kebab.extraction.metrics.bipartite_matching import (
    compute_bipartite_metrics,
    compute_scores_across_documents,
    compute_umatched_statistics,
    edit_distance,
    embedding_distance,
    normalize_property_names,
    str_match_distance,
    token_distance,
)
from kebab.extraction.metrics.metric_helpers import (
    EntityMatcher,
    MatchedEntitiesScorer,
    MetricsAccumulator,
    compute_properties_union,
)
from kebab.extraction.metrics.property_statistics import normalize_value


embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
encoder = tiktoken.get_encoding("cl100k_base")
token_score = lambda e1, e2, key: 1 - token_distance(e1, e2, key, encoder=encoder)
embedding_score = lambda e1, e2, key: 1 - embedding_distance(e1, e2, embed_model, key)
str_score = lambda e1, e2, key: 1 - str_match_distance(e1, e2, key)
edit_score = lambda e1, e2, key: 1 - edit_distance(e1, e2, key)


def test_normalize_value():
    assert normalize_value("ProJect") == "project"
    assert normalize_value("  Project  ") == "project"
    assert normalize_value("fnke project") == "fnke project"
    assert normalize_value("FNKE\tProject") == "fnke project"
    assert normalize_value("FNKE\nProject") == "fnke project"


def test_match_entities_in_different_order():
    ground_truth = [{"name": "Microsoft Research"}, {"name": "Alexandria"}, {"name": "FNKE"}]

    predictions = [
        {
            "name": "FNKE",
        },
        {"name": "Microsoft Research"},
        {"name": "Alexandria"},
    ]

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(lambda e1, e2: token_distance(e1, e2, key="name"))

    np.testing.assert_allclose(matched_pair.left_ind, [0, 1, 2])
    np.testing.assert_allclose(matched_pair.right_ind, [1, 2, 0])


def test_match_entities_with_larger_ground_truth():
    ground_truth = [
        {"name": "Microsoft Research"},
        {"name": "Alexandria"},
        {"name": "FNKE"},
        {"name": "Extraction Workstream"},
    ]

    predictions = [{"name": "Alexandria"}]

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(lambda e1, e2: token_distance(e1, e2, key="name"))

    np.testing.assert_allclose(matched_pair.left_ind, [1])
    np.testing.assert_allclose(matched_pair.right_ind, [0])
    np.testing.assert_allclose(matched_pair.distances, [[1], [0], [1], [1]])


def test_match_entities_with_larger_predictions():
    ground_truth = [{"name": "Alexandria"}]

    predictions = [
        {"name": "Microsoft Research"},
        {"name": "Alexandria"},
        {"name": "FNKE"},
        {"name": "Extraction Workstream"},
    ]

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(lambda e1, e2: token_distance(e1, e2, key="name"))

    np.testing.assert_allclose(matched_pair.left_ind, [0])
    np.testing.assert_allclose(matched_pair.right_ind, [1])
    np.testing.assert_allclose(matched_pair.distances, [[1, 0, 1, 1]])


def test_match_multiple_different_entities():
    ground_truth = [{"name": "Microsoft Research"}, {"name": "Alexandria"}, {"name": "FNKE"}, {"name": "MSRC"}]
    predictions = [{"name": "Cambridge"}, {"name": "FNKE"}, {"name": "Extraction Workstream"}, {"name": "Alexandria"}]

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(lambda e1, e2: token_distance(e1, e2, key="name"), threshold=0.5)

    np.testing.assert_allclose(matched_pair.left_ind, [1, 2])
    np.testing.assert_allclose(matched_pair.right_ind, [3, 1])
    np.testing.assert_allclose(matched_pair.left_unmatched, [0, 3])
    np.testing.assert_allclose(matched_pair.right_unmatched, [2, 0])


def test_match_entities_with_no_overlap():
    ground_truth = [{"name": "Microsoft Research"}, {"name": "Alexandria"}]
    predictions = [{"name": "Cambridge"}, {"name": "London"}]

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(lambda e1, e2: token_distance(e1, e2, key="name"), threshold=0.5)

    # matching score is lower than the threshold so no entities are matched
    np.testing.assert_allclose(matched_pair.left_ind, [])
    np.testing.assert_allclose(matched_pair.right_ind, [])
    np.testing.assert_allclose(matched_pair.left_unmatched, [0, 1])
    np.testing.assert_allclose(matched_pair.right_unmatched, [1, 0])

    properties_union = compute_properties_union(ground_truth, predictions, matched_pair)

    assert not properties_union

    entities_scorer = MatchedEntitiesScorer(ground_truth, predictions, matched_pair.left_ind, matched_pair.right_ind)

    evaluated_property_metrics = entities_scorer.score(token_score, "name")

    # no entities are matched so no scores are computed
    assert not evaluated_property_metrics.score_lists
    assert not evaluated_property_metrics.unmatched_count
    assert not evaluated_property_metrics.property_value_count
    assert not evaluated_property_metrics.zero_out_recall
    assert not evaluated_property_metrics.zero_out_prec


def test_entity_matches_with_low_matching_scores_should_not_be_evaluated():
    ground_truth = [{"name": "Microsoft Research"}, {"name": "Alexandria"}, {"name": "FNKE"}]
    predictions = [{"name": "Cambridge"}, {"name": "London"}]

    metrics_accumulator = compute_bipartite_metrics(
        ground_truth, predictions, embed_model=embed_model, matching_threshold=0.5
    )
    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    unmatched_pair_fraction, extra_pred_entities_fraction, extra_gt_entities_fraction = compute_umatched_statistics(
        metrics_accumulator
    )

    # no entities are matched
    assert metrics_accumulator.matched_count == 0
    assert metrics_accumulator.unmatched_pair_count == 2
    assert unmatched_pair_fraction == 1.0

    # extra entity in the ground truth
    assert metrics_accumulator.unmatched_extra_gt_count == 1
    assert metrics_accumulator.unmatched_extra_pred_count == 0
    assert extra_gt_entities_fraction == 1 / 3
    assert extra_pred_entities_fraction == 0.0

    # no property metrics are computed
    assert not property_precision
    assert not property_recall


def test_match_entities_with_different_thresholds():
    ground_truth = [{"name": "Microsoft Research", "type": "company"}, {"name": "Alexandria"}, {"name": "FNKE"}]
    predictions = [
        {"name": "Microsoft Corporation", "type": "organization"},
        {"name": "Alexandria"},
        {"name": "Extraction Project"},
    ]

    # threshold 1.0
    metrics_accumulator = compute_bipartite_metrics(
        ground_truth, predictions, embed_model=embed_model, matching_threshold=1.0
    )

    unmatched_pair_fraction, extra_pred_entities_fraction, extra_gt_entities_fraction = compute_umatched_statistics(
        metrics_accumulator
    )

    # no extra entities in the ground truth or predictions
    assert metrics_accumulator.unmatched_extra_gt_count == 0
    assert metrics_accumulator.unmatched_extra_pred_count == 0
    assert extra_pred_entities_fraction == 0.0
    assert extra_gt_entities_fraction == 0.0

    # with a threshold of 1, all entities are matched
    assert metrics_accumulator.matched_count == 3
    assert metrics_accumulator.unmatched_pair_count == 0
    assert unmatched_pair_fraction == 0

    # threshold 0.5
    metrics_accumulator = compute_bipartite_metrics(
        ground_truth, predictions, embed_model=embed_model, matching_threshold=0.5
    )
    unmatched_pair_fraction, extra_pred_entities_fraction, extra_gt_entities_fraction = compute_umatched_statistics(
        metrics_accumulator
    )

    # with a threshold of 0.5, the first two entities are matched
    assert metrics_accumulator.matched_count == 2
    assert metrics_accumulator.unmatched_pair_count == 1
    assert unmatched_pair_fraction == 1 / 3
    assert extra_pred_entities_fraction == 0.0
    assert extra_gt_entities_fraction == 0.0

    # threshold 0.0
    metrics_accumulator = compute_bipartite_metrics(
        ground_truth, predictions, embed_model=embed_model, matching_threshold=0.0
    )
    unmatched_pair_fraction, extra_pred_entities_fraction, extra_gt_entities_fraction = compute_umatched_statistics(
        metrics_accumulator
    )

    # with a threshold of 0, one entity is matched
    assert metrics_accumulator.matched_count == 1
    assert metrics_accumulator.unmatched_pair_count == 2
    assert unmatched_pair_fraction == 2 / 3
    assert extra_pred_entities_fraction == 0.0
    assert extra_gt_entities_fraction == 0.0


def test_compute_scores_of_target_entities_with_themselves():
    ground_truth = [{"name": "FNKE", "type": "project", "description": "fnke project description"}]
    predictions = ground_truth.copy()

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(lambda e1, e2: token_distance(e1, e2, key="name"))

    properties_union = compute_properties_union(ground_truth, predictions, matched_pair)

    assert properties_union == {"name", "type", "description"}

    for key in properties_union:
        entities_scorer = MatchedEntitiesScorer(
            ground_truth, predictions, matched_pair.left_ind, matched_pair.right_ind
        )
        evaluated_property_metrics = entities_scorer.score(token_score, key)

        np.testing.assert_allclose(evaluated_property_metrics.score_lists, [1.0])
        np.testing.assert_allclose(evaluated_property_metrics.unmatched_count, 0)
        np.testing.assert_allclose(evaluated_property_metrics.property_value_count, [1])
        np.testing.assert_allclose(evaluated_property_metrics.zero_out_recall, [False])
        np.testing.assert_allclose(evaluated_property_metrics.zero_out_prec, [False])


def test_evaluate_target_entities_with_themselves():
    ground_truth = [{"name": "FNKE", "type": "project", "description": "fnke project description"}]
    predictions = ground_truth.copy()

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert property_precision["name"] == property_precision["type"] == 1.0
    assert property_recall["name"] == property_recall["type"] == 1.0


def test_evaluate_target_entities_with_more_predictions():
    ground_truth = [{"name": "Alexandria"}, {"name": "FNKE"}]

    predictions = [
        {"name": "Microsoft Research"},
        {"name": "Alexandria"},
        {"name": "FNKE"},
        {"name": "Extraction Workstream"},
    ]

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    unmatched_pair_fraction, extra_pred_entities_fraction, extra_gt_entities_fraction = compute_umatched_statistics(
        metrics_accumulator
    )

    assert metrics_accumulator.matched_count == 2
    assert metrics_accumulator.unmatched_extra_pred_count == 2
    assert metrics_accumulator.unmatched_extra_gt_count == 0
    assert unmatched_pair_fraction == 0
    assert extra_gt_entities_fraction == 0
    assert extra_pred_entities_fraction == 0.5

    assert property_precision["name"] == 1.0
    assert property_recall["name"] == 1.0


def test_evaluate_target_entities_with_more_ground_truth():
    ground_truth = [
        {"name": "Microsoft Research"},
        {"name": "Alexandria"},
        {"name": "FNKE"},
        {"name": "Extraction Workstream"},
    ]

    predictions = [{"name": "Alexandria"}, {"name": "FNKE"}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    unmatched_pair_fraction, extra_pred_entities_fraction, extra_gt_entities_fraction = compute_umatched_statistics(
        metrics_accumulator
    )

    assert metrics_accumulator.matched_count == 2
    assert metrics_accumulator.unmatched_extra_gt_count == 2
    assert metrics_accumulator.unmatched_extra_pred_count == 0
    assert unmatched_pair_fraction == 0
    assert extra_gt_entities_fraction == 0.5
    assert extra_pred_entities_fraction == 0

    assert property_precision["name"] == 1.0
    assert property_recall["name"] == 1.0


def test_evaluate_set_properties_with_same_values():
    ground_truth = [{"name": "Alexandria", "types": ["project"]}]
    predictions = [{"name": "Alexandria", "types": ["project"]}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert property_precision["name"] == property_precision["types"] == 1.0
    assert property_recall["name"] == property_recall["types"] == 1.0


def test_evaluate_set_properties_with_different_values():
    ground_truth = [{"types": ["project", "team"]}]
    predictions = [{"types": ["project"]}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert metrics_accumulator.total_property_value_counts_per_doc["types"] == [2]
    assert property_precision["types"] == 1.0
    assert property_recall["types"] == 0.5


def test_evaluate_set_properties_with_more_predictions():
    ground_truth = [{"types": ["project", "team"]}]
    predictions = [{"types": ["project", "team", "organization", "company", "location"]}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert metrics_accumulator.total_property_value_counts_per_doc["types"] == [2]
    assert property_precision["types"] == 0.4
    assert property_recall["types"] == 1.0
    assert metrics_accumulator.total_unmatched_counts_per_doc["types"] == 3


def test_evaluate_set_properties_with_more_ground_truth():
    ground_truth = [{"types": ["project", "team", "organization"]}]
    predictions = [{"types": ["project", "team"]}]
    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert metrics_accumulator.total_property_value_counts_per_doc["types"] == [3]
    assert property_precision["types"] == 1.0
    np.testing.assert_allclose(property_recall["types"], 0.66666666)
    assert metrics_accumulator.total_unmatched_counts_per_doc["types"] == 0


def test_evaluate_set_properties_in_different_order_and_normalized():
    ground_truth = [
        {"alternative names": ["Microsoft Research", "MSRC", "Microsoft Research Cambridge", "MSR Cambridge"]}
    ]
    predictions = [{"alternative names": ["microsoft research cambridge"]}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert metrics_accumulator.total_property_value_counts_per_doc["alternative names"] == [4]
    assert property_precision["alternative names"] == 1.0
    assert property_recall["alternative names"] == 0.25


def test_evaluate_missing_properties_in_prediction():
    ground_truth = [{"name": "Steven", "type": "person"}]
    predictions = [{"name": "Steven"}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert property_precision["name"] == 1.0
    assert property_recall["name"] == 1.0
    assert property_precision["type"] == None
    assert property_recall["type"] == 0


def test_evaluate_more_properties_in_prediction():
    ground_truth = [{"name": "Steven"}]
    predictions = [{"name": "Steven", "type": "person"}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert property_precision["name"] == 1.0
    assert property_recall["name"] == 1.0
    assert property_precision["type"] == 0
    assert property_recall["type"] == None


def test_edit_score():
    ground_truth = [{"team": "Marketing"}]
    predictions = [{"team": "Strategic Marketing"}]
    score = edit_score(ground_truth[0], predictions[0], "team")
    assert score > 0


def test_combine_normalized_property_names():
    ground_truth = [{"name": "Alexandria", "type": "project", "types": ["team"], "issues": ["issue1", "issue2"]}]
    ground_truth = [normalize_property_names(entity) for entity in ground_truth]

    assert "type" not in ground_truth[0]
    assert set(ground_truth[0]["types"]) == {"project", "team"}
    assert "issue" not in ground_truth[0]
    assert set(ground_truth[0]["issues"]) == {"issue1", "issue2"}


def test_normalized_property_names():
    ground_truth = [{"name": "Alexandria", "type": ["project"], "types": ["team"]}]
    ground_truth = [normalize_property_names(entity) for entity in ground_truth]

    assert "type" not in ground_truth[0]
    assert set(ground_truth[0]["types"]) == {"project", "team"}


def test_evaluate_normalized_property_names():
    ground_truth = [{"name": "Alexandria", "descriptions": ["great project", "amazing team"]}]
    predictions = [{"name": "Alexandria", "description": "great project"}]

    ground_truth = [normalize_property_names(entity) for entity in ground_truth]
    predictions = [normalize_property_names(entity) for entity in predictions]

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(lambda e1, e2: token_distance(e1, e2, key="name"))

    properties_union = compute_properties_union(ground_truth, predictions, matched_pair)

    assert properties_union == {"name", "descriptions"}

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    np.testing.assert_allclose(property_precision["name"], 1.0)
    np.testing.assert_allclose(property_recall["name"], 1.0)
    np.testing.assert_allclose(property_precision["descriptions"], 1.0)
    np.testing.assert_allclose(property_recall["descriptions"], 0.5)


def test_evaluate_normalized_property_names_with_more_predictions():
    ground_truth = [{"name": "Alexandria", "description": "great project"}]
    predictions = [{"name": "Alexandria", "descriptions": ["great project", "amazing team"]}]

    ground_truth = [normalize_property_names(entity) for entity in ground_truth]
    predictions = [normalize_property_names(entity) for entity in predictions]

    entity_matcher = EntityMatcher(ground_truth, predictions)
    matched_pair = entity_matcher.match(lambda e1, e2: token_distance(e1, e2, key="name"))

    properties_union = compute_properties_union(ground_truth, predictions, matched_pair)

    assert properties_union == {"name", "descriptions"}

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    np.testing.assert_allclose(property_precision["name"], 1.0)
    np.testing.assert_allclose(property_recall["name"], 1.0)
    np.testing.assert_allclose(property_precision["descriptions"], 0.5)
    np.testing.assert_allclose(property_recall["descriptions"], 1.0)


def test_aggregate_across_documents_simple():
    ground_truth1 = [{"name": "John", "type": "person"}]
    predictions1 = [{"name": "John"}]

    ground_truth2 = [{"name": "Alexandria"}]
    predictions2 = [{"name": "Alexandria"}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth1, predictions1, embed_model=embed_model)
    metrics_accumulator2 = compute_bipartite_metrics(ground_truth2, predictions2, embed_model=embed_model)
    metrics_accumulator.update(metrics_accumulator2)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert len(metrics_accumulator.total_scores_per_doc["name"]) == 2
    assert len(metrics_accumulator.total_property_value_counts_per_doc["name"]) == 2
    assert len(metrics_accumulator.total_zero_out_recall["name"]) == 2
    assert len(metrics_accumulator.total_zero_out_precision["name"]) == 2

    assert len(metrics_accumulator.total_scores_per_doc["type"]) == 1
    assert len(metrics_accumulator.total_property_value_counts_per_doc["type"]) == 1
    assert len(metrics_accumulator.total_zero_out_recall["type"]) == 1
    assert len(metrics_accumulator.total_zero_out_precision["type"]) == 1

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert not property_precision["type"]
    assert property_recall["type"] == 0.0
    assert property_precision["name"] == 1.0
    assert property_recall["name"] == 1.0


def test_evaluate_entities_with_different_properties():
    # This will be updated when we update normalization by property name,
    # e.g. company and employer are the same in this case
    ground_truth1 = [
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
    predictions1 = [
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

    metrics_accumulator = compute_bipartite_metrics(ground_truth1, predictions1, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert not property_precision["employer"]
    assert property_recall["employer"] == 0.0

    assert property_precision["company"] == 0.0
    assert not property_recall["company"]

    assert property_precision["type"] == 1.0
    assert property_recall["type"] == 1.0
    assert property_precision["name"] == 1.0
    assert property_recall["name"] == 1.0
    assert property_precision["job title"] == 1.0
    assert property_recall["job title"] == 1.0
    assert property_precision["team"] == 1.0
    assert property_recall["team"] == 1.0


def test_aggregate_across_documents_with_multiple_property_values():
    ground_truth1 = [{"name": "Alexandria", "types": ["project"], "descriptions": ["amazing team"]}]
    predictions1 = [
        {"name": "Alexandria", "types": ["project", "team"], "descriptions": ["great project", "amazing team"]}
    ]

    ground_truth2 = [{"name": "Infer.Net", "types": ["software"], "descriptions": ["software description"]}]
    predictions2 = [
        {
            "name": "Infer.Net",
            "types": ["software", "team", "organization", "company", "location"],
            "descriptions": ["software description", "software description1"],
        }
    ]

    metrics_accumulator = compute_bipartite_metrics(ground_truth1, predictions1, embed_model=embed_model)
    metrics_accumulator2 = compute_bipartite_metrics(ground_truth2, predictions2, embed_model=embed_model)
    metrics_accumulator.update(metrics_accumulator2)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    # Found all property values in the ground truth
    np.testing.assert_allclose(property_recall["name"], 1.0)
    np.testing.assert_allclose(property_recall["types"], 1.0)
    np.testing.assert_allclose(property_recall["descriptions"], 1.0)

    # Each extra property value in the prediction is counted as a false positive
    np.testing.assert_allclose(property_precision["name"], 1.0)
    np.testing.assert_allclose(property_precision["types"], 2 / 7)
    np.testing.assert_allclose(property_precision["descriptions"], 0.5)


def test_aggregate_across_documents_with_multiple_entities():
    ground_truth1 = [{"name": "Alexandria", "types": ["project"]}, {"name": "FNKE", "types": ["project"]}]
    predictions1 = [{"name": "Alexandria", "types": ["project", "team"]}, {"name": "FNKE", "types": ["project"]}]

    ground_truth2 = [{"types": ["software"], "descriptions": ["software description"]}]
    predictions2 = [{"types": ["software"], "descriptions": ["software description"]}]

    metrics_accumulator = MetricsAccumulator()

    metrics_accumulator = compute_bipartite_metrics(ground_truth1, predictions1, embed_model=embed_model)
    metrics_accumulator2 = compute_bipartite_metrics(ground_truth2, predictions2, embed_model=embed_model)
    metrics_accumulator.update(metrics_accumulator2)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert (
        len(metrics_accumulator.total_scores_per_doc["name"])
        == len(metrics_accumulator.total_zero_out_recall["name"])
        == len(metrics_accumulator.total_zero_out_precision["name"])
        == 2
    )
    assert (
        len(metrics_accumulator.total_scores_per_doc["types"])
        == len(metrics_accumulator.total_zero_out_recall["types"])
        == len(metrics_accumulator.total_zero_out_precision["types"])
        == 3
    )
    assert (
        len(metrics_accumulator.total_scores_per_doc["descriptions"])
        == len(metrics_accumulator.total_zero_out_recall["descriptions"])
        == len(metrics_accumulator.total_zero_out_precision["descriptions"])
        == 1
    )

    np.testing.assert_allclose(property_recall["name"], 1.0)
    np.testing.assert_allclose(property_recall["types"], 1.0)
    np.testing.assert_allclose(property_recall["descriptions"], 1.0)

    np.testing.assert_allclose(property_precision["name"], 1.0)
    np.testing.assert_allclose(property_precision["types"], 0.75)
    np.testing.assert_allclose(property_precision["descriptions"], 1.0)


def test_aggregate_across_documents_with_multiple_entities_and_missing_properties_in_predictions():
    ground_truth1 = [{"name": "Alexandria", "types": ["project"]}, {"name": "FNKE", "types": ["project"]}]
    predictions1 = [{"name": "Alexandria"}, {"name": "FNKE"}]

    ground_truth2 = [{"name": "Phi2", "descriptions": ["small language model"]}]
    predictions2 = [{"name": "Phi2"}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth1, predictions1, embed_model=embed_model)
    metrics_accumulator2 = compute_bipartite_metrics(ground_truth2, predictions2, embed_model=embed_model)
    metrics_accumulator.update(metrics_accumulator2)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert (
        len(metrics_accumulator.total_scores_per_doc["name"])
        == len(metrics_accumulator.total_zero_out_recall["name"])
        == len(metrics_accumulator.total_zero_out_precision["name"])
        == 3
    )
    assert (
        len(metrics_accumulator.total_scores_per_doc["types"])
        == len(metrics_accumulator.total_zero_out_recall["types"])
        == len(metrics_accumulator.total_zero_out_precision["types"])
        == 2
    )
    assert (
        len(metrics_accumulator.total_scores_per_doc["descriptions"])
        == len(metrics_accumulator.total_zero_out_recall["descriptions"])
        == len(metrics_accumulator.total_zero_out_precision["descriptions"])
        == 1
    )

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert property_precision["name"] == 1.0

    # no types or descriptions in the predictions so precision can't be computed
    assert not property_precision["types"]
    assert not property_precision["descriptions"]

    assert property_recall["name"] == 1.0
    assert property_recall["types"] == 0.0
    assert property_recall["descriptions"] == 0.0


def test_aggregate_across_documents_with_multiple_entities_and_missing_properties_in_ground_truth():
    ground_truth1 = [{"name": "Alexandria"}, {"name": "FNKE"}]
    predictions1 = [{"name": "Alexandria", "types": ["project"]}, {"name": "FNKE", "types": ["project"]}]

    ground_truth2 = [{"name": "Phi2"}]
    predictions2 = [{"name": "Phi2", "descriptions": ["small language model"]}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth1, predictions1, embed_model=embed_model)
    metrics_accumulator2 = compute_bipartite_metrics(ground_truth2, predictions2, embed_model=embed_model)
    metrics_accumulator.update(metrics_accumulator2)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert (
        len(metrics_accumulator.total_scores_per_doc["name"])
        == len(metrics_accumulator.total_zero_out_recall["name"])
        == len(metrics_accumulator.total_zero_out_precision["name"])
        == 3
    )

    assert len(metrics_accumulator.total_scores_per_doc["types"]) == 1
    assert (
        len(metrics_accumulator.total_zero_out_recall["types"])
        == len(metrics_accumulator.total_zero_out_precision["types"])
        == 2
    )
    assert (
        len(metrics_accumulator.total_scores_per_doc["descriptions"])
        == len(metrics_accumulator.total_zero_out_recall["descriptions"])
        == len(metrics_accumulator.total_zero_out_precision["descriptions"])
        == 1
    )

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert property_precision["name"] == 1.0
    assert property_precision["types"] == 0.0
    assert property_precision["descriptions"] == 0.0

    assert property_recall["name"] == 1.0

    # no types or descriptions in the ground truth so recall can't be computed
    assert not property_recall["types"]
    assert not property_recall["descriptions"]


def test_evaluate_same_property_across_documents():
    ground_truth = [{"types": ["person", "lead"]}]
    predictions = [{"types": ["person", "lead"]}]

    ground_truth1 = [{"types": ["organization"]}]
    predictions1 = [{"types": ["company"]}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)
    metrics_accumulator2 = compute_bipartite_metrics(ground_truth1, predictions1, embed_model=embed_model)
    metrics_accumulator.update(metrics_accumulator2)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    np.testing.assert_allclose(property_precision["types"], 0.66666666)
    np.testing.assert_allclose(property_recall["types"], 0.66666666)


def test_unmatched_entities_in_predictions_should_not_be_evaluated():
    ground_truth = [{"name": "Microsoft Research"}, {"name": "Alexandria"}]
    predictions = [{"name": "Microsoft Research"}, {"name": "Alexandria"}, {"name": "FNKE", "type": "project"}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert metrics_accumulator.matched_count == 2
    assert metrics_accumulator.unmatched_extra_gt_count == 0
    assert metrics_accumulator.unmatched_extra_pred_count == 1
    assert metrics_accumulator.unmatched_pair_count == 0

    assert property_precision["name"] == 1
    assert property_recall["name"] == 1
    assert "type" not in property_precision
    assert "type" not in property_recall


def test_unmatched_entities_in_gt_should_not_be_evaluated():
    ground_truth = [{"name": "Microsoft Research"}, {"name": "Alexandria"}, {"name": "FNKE", "type": "project"}]
    predictions = [{"name": "Microsoft Research"}, {"name": "Alexandria"}]

    metrics_accumulator = compute_bipartite_metrics(ground_truth, predictions, embed_model=embed_model)

    property_precision, property_recall = compute_scores_across_documents(metrics_accumulator)

    assert metrics_accumulator.matched_count == 2
    assert metrics_accumulator.unmatched_extra_gt_count == 1
    assert metrics_accumulator.unmatched_extra_pred_count == 0
    assert metrics_accumulator.unmatched_pair_count == 0

    assert property_precision["name"] == 1
    assert property_recall["name"] == 1
    assert "type" not in property_precision
    assert "type" not in property_recall


def test_final_statistics():
    ground_truth1 = [{"name": "John"}, {"name": "Alexandria"}, {"name": "FNKE"}]
    predictions1 = [{"name": "James"}]
    ground_truth2 = [{"name": "Alexandria"}, {"name": "FNKE"}, {"name": "Infer.Net"}, {"name": "Phi3"}]
    predictions2 = [{"name": "Alexandria"}]
    ground_truth3 = [{"name": "Alexandria"}, {"name": "FNKE"}, {"name": "Infer.Net"}, {"name": "Phi3"}]
    predictions3 = [{"name": "Team"}, {"name": "Model"}]
    ground_truth4 = [{"name": "Alexandria"}]
    predictions4 = [{"name": "Alexandria"}, {"name": "New Alexandria"}]
    metrics_accumulator = compute_bipartite_metrics(
        ground_truth1, predictions1, embed_model=embed_model, matching_threshold=0.5
    )
    metrics_accumulator2 = compute_bipartite_metrics(
        ground_truth2, predictions2, embed_model=embed_model, matching_threshold=0.5
    )
    metrics_accumulator.update(metrics_accumulator2)
    metrics_accumulator3 = compute_bipartite_metrics(
        ground_truth3, predictions3, embed_model=embed_model, matching_threshold=0.5
    )
    metrics_accumulator.update(metrics_accumulator3)
    metrics_accumulator4 = compute_bipartite_metrics(
        ground_truth4, predictions4, embed_model=embed_model, matching_threshold=0.5
    )
    metrics_accumulator.update(metrics_accumulator4)

    unmatched_pair_fraction, extra_pred_entities_fraction, extra_gt_entities_fraction = compute_umatched_statistics(
        metrics_accumulator
    )

    assert unmatched_pair_fraction == 3 / 5
    assert extra_pred_entities_fraction == 1 / 6
    assert extra_gt_entities_fraction == 7 / 12
    assert metrics_accumulator.matched_count == 2
    assert metrics_accumulator.unmatched_extra_gt_count == 7
    assert metrics_accumulator.unmatched_extra_pred_count == 1
    assert metrics_accumulator.unmatched_pair_count == 3


def test_str_match_score():
    ground_truth = [{"email": "alexandria@microsoft.com"}]
    predictions = ground_truth.copy()
    score = str_score(ground_truth[0], predictions[0], "email")
    assert score == 1.0


def test_embedding_based_score():
    ground_truth = [
        {
            "description": "A tool used for extracting topics. It is used in a branch and has been tested for extracting personal topics and for the LLM stuff."
        }
    ]
    predictions = [
        {
            "description": "A software used for extracting entities from documents in the incremental KB construction procedure. It also has a type hierarchy and can model a lot of types"
        }
    ]
    score = embedding_score(ground_truth[0], predictions[0], "description")
    assert score > 0.5


def test_token_distance_with_non_string_values():
    # integers and strings are comparable
    assert token_distance(1, "1") == 0
    assert token_distance("1", 1) == 0
    assert token_distance(1, 1) == 0
    assert token_distance(1, "3") == 1
    assert token_distance("1", 3) == 1
    assert token_distance(1, 3) == 1

    # floats and strings are comparable
    assert token_distance(1.23, "1.23") == 0
    assert token_distance("1.23", 1.23) == 0
    assert token_distance(1.23, 1.23) == 0
    assert token_distance(2, "1.23") == 1
    assert token_distance("2", 1.23) == 1
    assert token_distance(2, 1.23) == 1
