# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import filecmp
import json
import math
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.tasks.linking import LinkingTask


@pytest.fixture
def _setup_and_teardown() -> Generator[None, Any, None]:
    output_dir = Path(__file__).parents[1] / "output" / "linking"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_linking_read_write_items_roundtrip() -> None:
    # Arrange
    entity_pairs_file_path = Path(__file__).parents[1] / "data" / "linking" / "entity_pairs.jsonl"
    boolean_labels_file_path = Path(__file__).parents[1] / "data" / "linking" / "boolean_labels.jsonl"
    schema_file_path = Path(__file__).parents[1] / "data" / "linking" / "property_schema.json"
    task_instance = LinkingTask(
        "Linking-Alexandria-Train",
        str(entity_pairs_file_path),
        str(schema_file_path),
        str(boolean_labels_file_path),
    )

    # Act
    items = list(task_instance.read_items())
    boolean_labels = [predicted_boolean for _, predicted_boolean in items if predicted_boolean is not None]
    boolean_labels_output_file_path = Path(__file__).parents[1] / "output" / "linking" / "boolean_labels.jsonl"
    task_instance.write_items(
        boolean_labels_output_file_path,
        boolean_labels,
    )

    # Assert
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
    assert filecmp.cmp(
        str(boolean_labels_file_path),
        str(boolean_labels_output_file_path),
    )


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_linking_read_int_labels() -> None:
    # Arrange
    entity_pairs_file_path = Path(__file__).parents[1] / "data" / "linking" / "entity_pairs.jsonl"
    int_labels_file_path = Path(__file__).parents[1] / "data" / "linking" / "int_labels.jsonl"
    schema_file_path = Path(__file__).parents[1] / "data" / "linking" / "property_schema.json"
    task_instance = LinkingTask(
        "Linking-Alexandria-UnitTest",
        str(entity_pairs_file_path),
        str(schema_file_path),
        str(int_labels_file_path),
    )

    # Act
    items = list(task_instance.read_items())
    assert len(items) == 2
    assert isinstance(items[0][1], bool)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_linking_metrics(tmp_path: Path) -> None:
    """Test the linking metrics calculation."""
    # Prepare the data
    labels = [True, True, False, False, False]
    predictions = [True, False, False, False, True]
    prob_predictions = [0.9, 0.1, 0.1, 0.1, 0.9]
    odds = [p / (1 - p) for p in prob_predictions]
    log_odds = [math.log(o) for o in odds]
    log_probs = [
        math.log(pred) if label else math.log(1 - pred) for pred, label in zip(prob_predictions, labels, strict=False)
    ]
    log_prob_total = sum(log_probs)

    labels_path = tmp_path / "labels.jsonl"
    with open(labels_path, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")

    predictions_path = tmp_path / "predictions.jsonl"
    with open(predictions_path, "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")

    prob_predictions_path = tmp_path / "prob_predictions.jsonl"
    with open(prob_predictions_path, "w") as f:
        for prediction in log_odds:
            f.write(json.dumps(prediction) + "\n")

    entity_pairs_path = tmp_path / "entity_pairs.jsonl"
    entity_pairs_path.write_text("")

    schema_file_path = tmp_path / "schema.json"
    schema_file_path.write_text("")

    # Evaluate
    task_instance = LinkingTask(
        "Linking-Metrics-Train",
        str(entity_pairs_path),
        str(schema_file_path),
        str(labels_path),
    )

    metrics = task_instance.evaluate(predictions_path)
    assert metrics["total"] == 5
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["tpr"] == 0.5
    assert abs(metrics["tnr"] - 2 / 3) < 1e-6
    assert abs(metrics["empirical_log_prob"] - -3.2966) < 1e-3
    assert "log_prob" not in metrics

    metrics = task_instance.evaluate(prob_predictions_path)
    assert metrics["total"] == 5
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["tpr"] == 0.5
    assert abs(metrics["tnr"] - 2 / 3) < 1e-6
    assert abs(metrics["empirical_log_prob"] - -3.296) < 1e-3
    assert abs(metrics["log_prob"] - log_prob_total) < 1e-3
