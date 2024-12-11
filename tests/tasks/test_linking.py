# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import filecmp
import shutil
from pathlib import Path
from typing import Any, Generator

import pytest
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
def test_linking_read_items() -> None:
    # Arrange
    entity_pairs_file_path = Path(__file__).parents[1] / "data" / "linking" / "entity_pairs.jsonl"
    boolean_labels_file_path = (
        Path(__file__).parents[1] / "data" / "linking" / "boolean_labels.jsonl"
    )
    task_instance = LinkingTaskInstance(
        "Linking-Alexandria-Train",
        Task(TaskType.Extraction),
        str(entity_pairs_file_path),
        str(boolean_labels_file_path),
    )

    # Act
    items = list(task_instance.read_items())
    boolean_labels = [predicted_boolean for _, predicted_boolean in items if predicted_boolean is not None]
    boolean_labels_output_file_path = (
        Path(__file__).parents[1] / "output" / "linking" / "boolean_labels.jsonl"
    )
    task_instance.write_items(
        boolean_labels_output_file_path,
        boolean_labels,
    )

    # Assert
    assert filecmp.cmp(
        str(boolean_labels_file_path),
        str(boolean_labels_output_file_path),
    )
