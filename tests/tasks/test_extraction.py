# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from kebab.contracts.document import Document
from kebab.contracts.entity import Entity
from kebab.tasks.extraction import ExtractionTask
from kebab.utils.io_helpers import compare_files_ignore_linebreaks


@pytest.fixture
def _setup_and_teardown() -> Generator[None, Any, None]:
    output_dir = Path(__file__).parents[1] / "output" / "extraction"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_extraction_read_write_items_roundtrip() -> None:
    # Arrange
    items_file_path = Path(__file__).parents[1] / "data" / "extraction" / "plain_text_items.jsonl"
    extracted_entities_file_path = (
        Path(__file__).parents[1] / "data" / "extraction" / "plain_text_extracted_entities.jsonl"
    )
    metrics_config_file_path = Path(__file__).parents[1] / "data" / "extraction" / "metrics_config.json"
    property_schema_file_path = Path(__file__).parents[1] / "data" / "extraction" / "property_schema.json"
    task_instance = ExtractionTask(
        "Extraction-Alexandria-Train",
        extracts=items_file_path,
        schema=property_schema_file_path,
        ground_truth=extracted_entities_file_path,
        metrics_config=metrics_config_file_path,
    )

    # Act
    items = list(task_instance.read_items())
    extracted_entity_lists = [item.entities for item in items if item.entities is not None]
    extracted_entities_output_file_path = (
        Path(__file__).parents[1] / "output" / "extraction" / "plain_text_extracted_entities.jsonl"
    )
    task_instance.write_items(
        extracted_entities_output_file_path,
        extracted_entity_lists,
    )

    # Assert
    assert len(items) == 2
    first_doc = items[0].document
    assert isinstance(first_doc, Document)
    assert first_doc.document_id == "doc_0"
    first_doc_extracted_entities = items[0].entities
    assert first_doc_extracted_entities is not None
    assert len(first_doc_extracted_entities) == 2
    first_extracted_entity = first_doc_extracted_entities[0]
    assert isinstance(first_extracted_entity, Entity)
    assert first_extracted_entity.entity_id == "doc_0_entity_0"
    assert compare_files_ignore_linebreaks(
        extracted_entities_file_path,
        extracted_entities_output_file_path,
    )
