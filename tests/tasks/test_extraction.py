# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import filecmp
import shutil
from pathlib import Path
from typing import Any, Generator

import pytest
from kebab.contracts.document import Document
from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.tasks.extraction import ExtractionTaskInstance


@pytest.fixture
def _setup_and_teardown() -> Generator[None, Any, None]:
    output_dir = Path(__file__).parents[1] / "output" / "extraction"
    output_dir.mkdir(parents=True, exist_ok=True)
    Task.clear_created_task_types()
    yield
    Task.clear_created_task_types()
    shutil.rmtree(output_dir)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_extraction_read_write_items_roundtrip() -> None:
    # Arrange
    items_file_path = Path(__file__).parents[1] / "data" / "extraction" / "plain_text_items.jsonl"
    extracted_entities_file_path = (
        Path(__file__).parents[1] / "data" / "extraction" / "plain_text_extracted_entities.jsonl"
    )
    task_instance = ExtractionTaskInstance(
        "Extraction-Alexandria-Train",
        Task(TaskType.Extraction),
        str(items_file_path),
        str(extracted_entities_file_path),
    )

    # Act
    items = list(task_instance.read_items())
    extracted_entity_lists = [entity_list for _, entity_list in items if entity_list is not None]
    extracted_entities_output_file_path = (
        Path(__file__).parents[1] / "output" / "extraction" / "plain_text_extracted_entities.jsonl"
    )
    task_instance.write_items(
        extracted_entities_output_file_path,
        extracted_entity_lists,
    )

    # Assert
    assert len(items) == 2
    first_doc = items[0][0]
    assert isinstance(first_doc, Document)
    assert first_doc.document_id == "doc_0"
    first_doc_extracted_entities = items[0][1]
    assert first_doc_extracted_entities is not None
    assert len(first_doc_extracted_entities) == 2
    first_extracted_entity = first_doc_extracted_entities[0]
    assert isinstance(first_extracted_entity, Entity)
    assert first_extracted_entity.entity_id == "doc_0_entity_0"
    assert filecmp.cmp(
        str(extracted_entities_file_path),
        str(extracted_entities_output_file_path),
    )
