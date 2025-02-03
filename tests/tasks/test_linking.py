# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import filecmp
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.tasks.linking import LinkingTaskInstance


@pytest.fixture
def _setup_and_teardown() -> Generator[None, Any, None]:
    output_dir = Path(__file__).parents[1] / "output" / "linking"
    output_dir.mkdir(parents=True, exist_ok=True)
    Task.clear_created_task_types()
    yield
    Task.clear_created_task_types()
    shutil.rmtree(output_dir)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_linking_read_write_items_roundtrip() -> None:
    # Arrange
    entity_pairs_file_path = Path(__file__).parents[1] / "data" / "linking" / "entity_pairs.jsonl"
    boolean_labels_file_path = Path(__file__).parents[1] / "data" / "linking" / "boolean_labels.jsonl"
    schema_file_path = Path(__file__).parents[1] / "data" / "linking" / "propert_schema.json"
    task_instance = LinkingTaskInstance(
        "Linking-Alexandria-Train",
        Task(TaskType.Linking),
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
