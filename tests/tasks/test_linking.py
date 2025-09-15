# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import math
import shutil
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from kebab.contracts.entity import Entity
from kebab.tasks.linking import LinkingTask
from kebab.utils.io_helpers import compare_files_ignore_linebreaks


@pytest.fixture
def _setup_and_teardown(tmp_path: Path) -> Generator[None, Any, None]:
    output_dir = tmp_path / "linking"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_linking_read_write_items_roundtrip(tmp_path: Path) -> None:
    """Test reading and writing items in the LinkingTask."""
    entity_pairs_path = Path(__file__).parents[1] / "data" / "linking" / "entity_pairs.jsonl"
    boolean_labels_path = Path(__file__).parents[1] / "data" / "linking" / "boolean_labels.jsonl"
    schema_path = Path(__file__).parents[1] / "data" / "linking" / "property_schema.json"
    task_instance = LinkingTask(
        "Linking-Alexandria-Train",
        entity_pairs_path,
        schema_path,
        boolean_labels_path,
    )

    # Prepare data
    items = list(task_instance.read_items())
    boolean_labels = [predicted_boolean for _, predicted_boolean in items if predicted_boolean is not None]
    boolean_labels_output_path = tmp_path / "boolean_labels.jsonl"
    task_instance.write_items(
        boolean_labels_output_path,
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
        boolean_labels_path,
        boolean_labels_output_path,
    )


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_linking_read_int_labels() -> None:
    """Test reading integer labels in the LinkingTask."""
    # Prepare the data
    entity_pairs_path = Path(__file__).parents[1] / "data" / "linking" / "entity_pairs.jsonl"
    int_labels_path = Path(__file__).parents[1] / "data" / "linking" / "int_labels.jsonl"
    schema_path = Path(__file__).parents[1] / "data" / "linking" / "property_schema.json"
    task_instance = LinkingTask(
        "Linking-Alexandria-UnitTest",
        entity_pairs_path,
        schema_path,
        int_labels_path,
    )

    # Assert the task instance is created and can read items
    items = list(task_instance.read_items())
    assert len(items) == 2
    assert isinstance(items[0][1], bool)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_linking_metrics(tmp_path: Path) -> None:
    """Test the linking metrics calculation."""
    # labels
    labels = np.asarray([True, True, False, False, False])
    total = len(labels)
    pairs = [(Entity(f"frag_{i}_left"), Entity(f"frag_{i}_right")) for i in range(len(labels))]

    # binary predictions
    predictions = np.asarray([True, False, False, False, True])

    # probabilistic predictions
    prob_predictions = np.asarray([0.91, 0.3, 0.45, 0.45, 0.89])
    odds = prob_predictions / (1 - prob_predictions)
    log_odds = np.log(odds)

    log_probs = np.where(labels, np.log(prob_predictions), np.log(1 - prob_predictions))
    log_prob_avg = log_probs.mean()

    def write_jsonl(path: Path, data: Iterable[Any]) -> None:
        """Helper function to write a list of data to a JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(labels_path, (bool(label) for label in labels))

    binary_predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(binary_predictions_path, (float(pred) for pred in predictions))

    prob_predictions_path = tmp_path / "prob_predictions.jsonl"
    write_jsonl(prob_predictions_path, (float(label) for label in log_odds))

    entity_pairs_path = tmp_path / "entity_pairs.jsonl"
    write_jsonl(entity_pairs_path, ((left.to_dict(), right.to_dict()) for left, right in pairs))

    schema_path = tmp_path / "schema.json"
    schema_path.write_text("")

    # Evaluate
    task_instance = LinkingTask("Linking-Metrics-Train", entity_pairs_path, schema_path, labels_path)

    # binary predictions
    binary_metrics_path = tmp_path / "binary_metrics.json"
    metrics = task_instance.evaluate(
        binary_predictions_path, result_output_path=binary_metrics_path, adjust_to_test_prior=False, output_dir=tmp_path
    )

    for value in metrics.values():
        assert isinstance(value, (int | float)) or value is None

    # validate metrics were written correctly
    with open(binary_metrics_path, encoding="utf-8") as f:
        loaded_metrics = json.load(f)

    def _equal(a: Any, b: Any) -> bool:
        """Check equality of two values, treating NaNs as equal."""
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return True
            return a == b
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            return all(_equal(a[k], b[k]) for k in a)
        return a == b

    assert _equal(loaded_metrics, metrics)

    # validate metrics
    assert metrics["total"] == 5
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 2
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["ppv"] == 0.5
    assert np.isclose(metrics["npv"], 2 / 3)
    assert math.isnan(metrics["log_prob"])
    assert math.isnan(metrics["log_odds_adjustment"])
    assert np.isclose(metrics["optimistic_log_prob"], -3.2958 / total, atol=1e-4, rtol=1e-4)

    # probabilistic predictions
    metrics = task_instance.evaluate(prob_predictions_path, adjust_to_test_prior=False, output_dir=tmp_path)
    for value in metrics.values():
        assert isinstance(value, (int | float)) or value is None

    assert metrics["total"] == 5
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 2
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["ppv"] == 0.5
    assert np.isclose(metrics["npv"], 2 / 3)
    assert np.isclose(metrics["log_prob"], log_prob_avg)
    assert math.isnan(metrics["log_odds_adjustment"])
    assert math.isnan(metrics["optimistic_log_prob"])

    # log odds adjustment
    metrics = task_instance.evaluate(prob_predictions_path, adjust_to_test_prior=True, output_dir=tmp_path)

    def adj_log_prob(adj: float) -> float:
        adj_log_odds = log_odds + adj
        adj_prob_predictions = np.exp(adj_log_odds) / (1 + np.exp(adj_log_odds))
        log_probs = np.where(labels, np.log(adj_prob_predictions), np.log(1 - adj_prob_predictions))
        log_prob_avg = log_probs.mean()
        return log_prob_avg

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
    assert metrics["true_negative"] == 2
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["ppv"] == 0.5
    assert np.isclose(metrics["npv"], 2 / 3)
    assert np.isclose(metrics["log_prob"], cast(float, best_log_prob))
    assert np.isclose(metrics["log_odds_adjustment"], cast(float, best_adj), atol=1e-3, rtol=1e-3)
    assert math.isnan(metrics["optimistic_log_prob"])

    # test corner cases
    labels = np.asarray([True, False, True, False])
    pairs = [(Entity(f"frag_{i}_left"), Entity(f"frag_{i}_right")) for i in range(len(labels))]
    write_jsonl(labels_path, (bool(label) for label in labels))
    write_jsonl(entity_pairs_path, ((left.to_dict(), right.to_dict()) for left, right in pairs))

    # test corner cases
    def test_case_1_zero(
        predictions: np.ndarray,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
        precision: float,
        recall: float,
        ppv: float,
        npv: float,
        opt_log_prob: float,
    ) -> None:
        """Helper function to test case 1 with zero FN."""
        write_jsonl(binary_predictions_path, (float(pred) for pred in predictions))
        task_instance = LinkingTask("Linking-Metrics-Train", entity_pairs_path, schema_path, labels_path)
        metrics = task_instance.evaluate(binary_predictions_path, adjust_to_test_prior=False, output_dir=tmp_path)
        assert metrics["total"] == 4
        assert metrics["true_positive"] == tp
        assert metrics["false_positive"] == fp
        assert metrics["true_negative"] == tn
        assert metrics["false_negative"] == fn

        assert np.isclose(metrics["precision"], precision)
        assert np.isclose(metrics["recall"], recall)
        assert np.isclose(metrics["ppv"], ppv)
        assert np.isclose(metrics["npv"], npv)

        assert np.isclose(metrics["optimistic_log_prob"], opt_log_prob / 4)

    # FN = 0
    test_case_1_zero(
        np.asarray([True, False, True, True]),
        tp=2,
        fp=1,
        tn=1,
        fn=0,
        precision=2 / 3,
        recall=1.0,
        ppv=2 / 3,
        npv=1.0,
        opt_log_prob=2 * np.log(2 / 3) + 1 * np.log(1 / 3),
    )

    # TN = 0
    test_case_1_zero(
        np.asarray([True, True, False, True]),
        tp=1,
        fp=2,
        tn=0,
        fn=1,
        precision=1 / 3,
        recall=1 / 2,
        ppv=1 / 3,
        npv=0,
        opt_log_prob=1 * np.log(1 / 3) + 2 * np.log(2 / 3),
    )

    # TP = 0
    test_case_1_zero(
        np.asarray([False, False, False, True]),
        tp=0,
        fp=1,
        tn=1,
        fn=2,
        precision=0,
        recall=0,
        ppv=0,
        npv=1 / 3,
        opt_log_prob=1 * np.log(1 / 3) + 2 * np.log(2 / 3),
    )

    # FP = 0
    test_case_1_zero(
        np.asarray([True, False, False, False]),
        tp=1,
        fp=0,
        tn=2,
        fn=1,
        precision=1,
        recall=1 / 2,
        ppv=1,
        npv=2 / 3,
        opt_log_prob=2 * np.log(2 / 3) + 1 * np.log(1 / 3),
    )
