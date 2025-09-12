# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from kebab.contracts.entity import Entity
from kebab.tasks.entity_generation import EntityGenerationTask


@pytest.fixture
def _setup_and_teardown() -> Generator[None, Any, None]:
    output_dir = Path(__file__).parents[1] / "output" / "entity_generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_entity_generation_read_write_items_roundtrip() -> None:
    # Arrange
    entities_file_path = Path(__file__).parents[1] / "data" / "entity_generation" / "entities.jsonl"
    schema_file_path = Path(__file__).parents[1] / "data" / "entity_generation" / "propert_schema.json"
    task_instance = EntityGenerationTask(
        "EntityGeneration-Alexandria-Train",
        entities_file_path,
        schema_file_path,
    )

    # Act
    items = list(task_instance.read_items())

    # Assert
    assert len(items) == 2
    first_entity = items[0]
    assert isinstance(first_entity, Entity)
    assert first_entity.entity_id == "doc_0_entity_0"
