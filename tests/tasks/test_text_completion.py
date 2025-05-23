# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import filecmp
import math
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from kebab.tasks.text_completion import TextCompletionUsingDocumentsTask


@pytest.fixture
def _setup_and_teardown() -> Generator[None, Any, None]:
    output_dir = Path(__file__).parents[1] / "output" / "text_completion"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


@pytest.mark.usefixtures(_setup_and_teardown.__name__)
def test_text_completion_read_items_and_generate_queries() -> None:
    # Arrange
    documents_file_path = Path(__file__).parents[1] / "data" / "text_completion" / "documents.jsonl"
    predictions_file_path = Path(__file__).parents[1] / "data" / "text_completion" / "predicted_contents.jsonl"
    predictions_output_file_path = Path(__file__).parents[1] / "output" / "text_completion" / "predicted_contents.jsonl"
    task_instance = TextCompletionUsingDocumentsTask(
        "TextCompletion",
        str(documents_file_path),
    )

    # Act
    items = list(task_instance.read_items())
    partial_queries = list(task_instance.generate_partial_queries())
    task_instance.write_items(
        predictions_output_file_path,
        [(query["target_content"], -0.1) for query in partial_queries],
    )
    metrics = task_instance.evaluate(predictions_output_file_path)

    # Assert
    assert items[0].document_id == "doc_0"
    assert items[0].data["title"] == "Paris"
    assert items[0].data["text"] == "The capital of France is Paris."
    assert items[1].document_id == "doc_1"
    assert items[1].data["title"] == "Beijing"
    assert items[1].data["text"] == "The capital of China is Beijing."
    assert partial_queries == [
        {"text_with_mask": "The <mask>", "target_content": "capital", "document_id": "doc_0"},
        {"text_with_mask": "The capital <mask>", "target_content": "of", "document_id": "doc_0"},
        {"text_with_mask": "The capital of <mask>", "target_content": "France", "document_id": "doc_0"},
        {"text_with_mask": "The capital of France <mask>", "target_content": "is", "document_id": "doc_0"},
        {"text_with_mask": "The capital of France is <mask>", "target_content": "Paris", "document_id": "doc_0"},
        {"text_with_mask": "The <mask>", "target_content": "capital", "document_id": "doc_1"},
        {"text_with_mask": "The capital <mask>", "target_content": "of", "document_id": "doc_1"},
        {"text_with_mask": "The capital of <mask>", "target_content": "China", "document_id": "doc_1"},
        {"text_with_mask": "The capital of China <mask>", "target_content": "is", "document_id": "doc_1"},
        {"text_with_mask": "The capital of China is <mask>", "target_content": "Beijing", "document_id": "doc_1"},
    ]
    assert filecmp.cmp(
        str(predictions_file_path),
        str(predictions_output_file_path),
    )
    assert metrics["mean_log_prob"] == -0.1
    assert metrics["variance_log_prob"] == 0.0
    assert math.isclose(metrics["perplexity"], 1.105, abs_tol=1e-3)
