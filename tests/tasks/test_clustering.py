# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import filecmp
import json
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.tasks.clustering import ClusteringTaskInstance


@pytest.fixture
def _setup_and_teardown() -> Generator[None, Any, None]:
    output_dir = Path(__file__).parents[1] / "output" / "clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    Task.clear_created_task_types()
    yield
    Task.clear_created_task_types()
    shutil.rmtree(output_dir)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_clustering_read_write_items_roundtrip() -> None:
    # Arrange
    entity_pairs_file_path = Path(__file__).parents[1] / "data" / "clustering" / "entity_fragments.jsonl"
    labels_file_path = Path(__file__).parents[1] / "data" / "clustering" / "labels.jsonl"
    schema_file_path = Path(__file__).parents[1] / "data" / "clustering" / "property_schema.json"
    task_instance = ClusteringTaskInstance(
        "Clustering-Roundtrip-Train",
        Task(TaskType.Clustering),
        str(entity_pairs_file_path),
        str(schema_file_path),
        str(labels_file_path),
    )

    # Act
    items = list(task_instance.read_items())
    labels = [pred_label for _, pred_label in items if pred_label is not None]
    pred_labels_output_file_path = Path(__file__).parents[1] / "output" / "clustering" / "labels.jsonl"
    task_instance.write_items(
        pred_labels_output_file_path,
        labels,
    )

    # Assert
    assert len(items) == 2
    entity = items[0][0]
    assert isinstance(entity, Entity)
    assert entity.entity_id == "doc_0_entity_0"
    first_label = items[0][1]
    assert isinstance(first_label, str)
    assert first_label == "0"
    assert filecmp.cmp(
        str(labels_file_path),
        str(pred_labels_output_file_path),
    )


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_clustering_metrics(tmp_path: Path) -> None:
    """Test the clustering metrics calculation."""
    # Prepare the data
    labels = ["0", "1", "0", "1", "2"]
    predictions = ["a", "b", "a", "a", "a"]

    labels_path = tmp_path / "labels.jsonl"
    with open(labels_path, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")

    predictions_path = tmp_path / "predictions.jsonl"
    with open(predictions_path, "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")

    entity_fragments_path = tmp_path / "entity_fragments.jsonl"
    entity_fragments_path.write_text("")

    schema_file_path = tmp_path / "schema.json"
    schema_file_path.write_text("")

    # Evaluate
    task_instance = ClusteringTaskInstance(
        "Clustering-Metrics-Train",
        Task(TaskType.Clustering),
        str(entity_fragments_path),
        str(schema_file_path),
        str(labels_path),
    )

    metrics = task_instance.evaluate(predictions_path)
    assert metrics["fragments"] == 5
    assert metrics["predicted_clusters"] == 2
    assert metrics["ground_truth_clusters"] == 3
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.8
