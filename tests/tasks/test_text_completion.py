# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import filecmp
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from kebab.contracts.entity import Entity
from kebab.tasks.text_completion import TextCompletionTask


@pytest.fixture
def _setup_and_teardown() -> Generator[None, Any, None]:
    output_dir = Path(__file__).parents[1] / "output" / "text_completion"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(output_dir)

@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_text_completion_read_items() -> None:
    # Arrange
    documents_file_path = Path(__file__).parents[1] / "data" / "text_completion" / "plain_text_items.jsonl"
    task_instance = TextCompletionTask(
        "Linking-Alexandria-Train",
        str(documents_file_path),
        "",
    )
    items = list(task_instance.read_items())
    items[0].document_id = "doc_0"
    items[0].data["title"] = "Paris"
    items[0].data["text"] = "The capital of France is Paris."
    items[1].document_id = "doc_1"
    items[1].data["title"] = "Beijing"
    items[1].data["text"] = "The capital of China is Beijing."
