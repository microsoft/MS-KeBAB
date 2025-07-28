# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import math
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from kebab.contracts.entity import Entity
from kebab.tasks.linking import LinkingTask
from kebab.utils.io_helpers import compare_files_ignore_linebreaks


@pytest.fixture
def _setup_and_teardown() -> Generator[None, Any, None]:
    output_dir = Path(__file__).parents[1] / "output" / "linking"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_linking_read_write_items_roundtrip() -> None:
    """Test reading and writing items in the LinkingTask."""
    entity_pairs_file_path = Path(__file__).parents[1] / "data" / "linking" / "entity_pairs.jsonl"
    boolean_labels_file_path = Path(__file__).parents[1] / "data" / "linking" / "boolean_labels.jsonl"
    schema_file_path = Path(__file__).parents[1] / "data" / "linking" / "property_schema.json"
    task_instance = LinkingTask(
        "Linking-Alexandria-Train",
        str(entity_pairs_file_path),
        str(schema_file_path),
        str(boolean_labels_file_path),
    )

    # Prepare data
    items = list(task_instance.read_items())
    boolean_labels = [predicted_boolean for _, predicted_boolean in items if predicted_boolean is not None]
    boolean_labels_output_file_path = Path(__file__).parents[1] / "output" / "linking" / "boolean_labels.jsonl"
    task_instance.write_items(
        boolean_labels_output_file_path,
        boolean_labels,
    )

    # Assert the output file is created and contains the expected data
    assert len(items) == 2
    first_entity_pair = items[0][0]
    assert isinstance(first_entity_pair, tuple)
    assert len(first_entity_pair) == 2
    assert all(isinstance(entity, Entity) for entity in first_entity_pair)
    first_entity = first_entity_pair[0]
    assert first_entity.entity_id == "doc_0_entity_0"
    first_boolean_label = items[0][1]
    assert isinstance(first_boolean_label, bool)
    assert first_boolean_label
    assert compare_files_ignore_linebreaks(
        boolean_labels_file_path,
        boolean_labels_output_file_path,
    )


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_linking_read_int_labels() -> None:
    """Test reading integer labels in the LinkingTask."""
    # Prepare the data
    entity_pairs_file_path = Path(__file__).parents[1] / "data" / "linking" / "entity_pairs.jsonl"
    int_labels_file_path = Path(__file__).parents[1] / "data" / "linking" / "int_labels.jsonl"
    schema_file_path = Path(__file__).parents[1] / "data" / "linking" / "property_schema.json"
    task_instance = LinkingTask(
        "Linking-Alexandria-UnitTest",
        str(entity_pairs_file_path),
        str(schema_file_path),
        str(int_labels_file_path),
    )

    # Assert the task instance is created and can read items
    items = list(task_instance.read_items())
    assert len(items) == 2
    assert isinstance(items[0][1], bool)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_linking_metrics(tmp_path: Path) -> None:
    """Test the linking metrics calculation."""
    # Prepare the data
    labels = np.asarray([True, True, False, False, False])
    pairs = [(Entity(f"frag_{i}_left"), Entity(f"frag_{i}_right")) for i in range(len(labels))]
    predictions = np.asarray([True, False, False, False, True])
    prob_predictions = np.asarray([0.91, 0.3, 0.45, 0.45, 0.89])
    odds = prob_predictions / (1 - prob_predictions)
    log_odds = np.log(odds)
    shift = 5
    log_odds += shift
    shifted_predictions = np.exp(log_odds) / (1 + np.exp(log_odds))
    log_probs = np.where(labels, np.log(shifted_predictions), np.log(1 - shifted_predictions))
    log_prob_total = log_probs.sum()

    labels_path = tmp_path / "labels.jsonl"
    with open(labels_path, "w") as f:
        for label in labels:
            f.write(json.dumps(bool(label)) + "\n")

    binary_predictions_path = tmp_path / "predictions.jsonl"
    with open(binary_predictions_path, "w") as f:
        for prediction in predictions:
            f.write(json.dumps(float(prediction)) + "\n")

    prob_predictions_path = tmp_path / "prob_predictions.jsonl"
    with open(prob_predictions_path, "w") as f:
        for prediction in log_odds:
            f.write(json.dumps(float(prediction)) + "\n")

    entity_pairs_path = tmp_path / "entity_pairs.jsonl"
    with open(entity_pairs_path, "w") as f:
        for left_entity, right_entity in pairs:
            pair = (left_entity.to_dict(), right_entity.to_dict())
            f.write(json.dumps(pair) + "\n")

    schema_file_path = tmp_path / "schema.json"
    schema_file_path.write_text("")

    # Evaluate
    task_instance = LinkingTask(
        "Linking-Metrics-Train",
        str(entity_pairs_path),
        str(schema_file_path),
        str(labels_path),
    )

    # binary predictions
    metrics = task_instance.evaluate(binary_predictions_path, adjust_to_test_prior=False)
    assert metrics["total"] == 5
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["ppv"] == 0.5
    assert np.isclose(metrics["npv"], 2 / 3)
    assert math.isnan(metrics["log_prob"])
    assert math.isnan(metrics["log_odds_adjustment"])
    assert np.isclose(metrics["optimistic_log_prob"], -3.2958, atol=1e-4, rtol=1e-4)

    # probabilistic predictions
    metrics = task_instance.evaluate(prob_predictions_path, adjust_to_test_prior=False)
    assert metrics["total"] == 5
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 3
    assert metrics["false_negative"] == 0
    assert metrics["true_negative"] == 0
    assert metrics["precision"] == 0.4
    assert metrics["recall"] == 1.0
    assert metrics["ppv"] == 0.4
    assert math.isnan(metrics["npv"])
    assert np.isclose(metrics["log_prob"], log_prob_total)
    assert math.isnan(metrics["log_odds_adjustment"])
    assert math.isnan(metrics["optimistic_log_prob"])

    # log odds adjustment
    metrics = task_instance.evaluate(prob_predictions_path, adjust_to_test_prior=True)

    def adj_log_prob(adj: float) -> float:
        adj_log_odds = log_odds + adj
        adj_prob_predictions = np.exp(adj_log_odds) / (1 + np.exp(adj_log_odds))
        log_probs = np.where(labels, np.log(adj_prob_predictions), np.log(1 - adj_prob_predictions))
        log_prob_total = log_probs.sum()
        return log_prob_total

    adjustments = np.linspace(-10, 10, 20001)
    best_adj = None
    best_log_prob = None

    for adj in adjustments:
        log_prob = adj_log_prob(adj)

        if best_log_prob is None or log_prob > best_log_prob:
            best_log_prob = log_prob
            best_adj = adj

    assert metrics["total"] == 5
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["ppv"] == 0.5
    assert np.isclose(metrics["npv"], 2 / 3)
    assert np.isclose(metrics["log_prob"], cast(float, best_log_prob))
    assert np.isclose(metrics["log_odds_adjustment"], cast(float, best_adj), atol=1e-3, rtol=1e-3)
    assert math.isnan(metrics["optimistic_log_prob"])
