# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Any, Generator

import pytest
from kebab.contracts.task import Task, TaskType
from kebab.tasks.extraction import ExtractionTaskInstance


@pytest.fixture
def _setup_and_teardown() -> Generator[None, Any, None]:
    Task.clear_created_task_types()
    yield
    Task.clear_created_task_types()


@pytest.mark.usefixtures("_setup_and_teardown")
def test_extraction_read_write_items() -> None:
    task = Task(TaskType.Extraction)
    task_instance = ExtractionTaskInstance(
        "Extraction-ReDocRED-Train",
        task,
        str(Path(__file__).parents[1] / "data" / "extraction" / "plain_text_items.jsonl"),
        str(Path(__file__).parents[1] / "data" / "extraction" / "plain_text_extracted_entities.jsonl"),
    )
    items = task_instance.read_items()
    assert items is not None
