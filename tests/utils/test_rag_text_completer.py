# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Any, override

import pytest
from kebab.tasks.text_completion import TextCompletionUsingDocumentsTask
from kebab.utils.io_helpers import ItemJsonlReader
from kebab.utils.rag_text_completer import BaseRAGTextCompleter

class MockRAGTextCompleter(BaseRAGTextCompleter):
    @override
    def get_augmented_context(self, query: dict[str, Any]) -> str:
        return ""

    @override
    def complete_single_partial_query(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str = "",
        seed: int = 42,
    ) -> dict[str, Any]:
        # Dummy logic for testing.
        return {
            "predicted_content": target_content,
            "target_content_logprob": -0.1,
        }

@pytest.fixture
def text_completion_task() -> TextCompletionUsingDocumentsTask:
    documents_file_path = Path(__file__).parents[1] / "data" / "text_completion" / "documents.jsonl"
    return TextCompletionUsingDocumentsTask(
        "TextCompletion",
        documents_file_path,
    )

def _extract_predictions(indexed_results: list[tuple[int, dict[str, Any]]]) -> list[dict[str, str | float]]:
    """Sort results by index, filter to attempted completions, and extract prediction fields."""
    indexed_results.sort(key=lambda x: x[0])
    return [
        {k: v for k, v in result.items() if k in ("predicted_content", "target_content_logprob")}
        for _, result in indexed_results
        if result.get("completion_attempted", False)
    ]

def test_rag_text_completer(text_completion_task) -> None:
    # Arrange
    predictions_file_path = Path(__file__).parents[1] / "data" / "text_completion" / "document_predictions.jsonl"
    expected_predictions = list(ItemJsonlReader[dict[str, str | float]](predictions_file_path).read_items())
    text_completer = MockRAGTextCompleter()

    # Act
    partial_queries = text_completion_task.generate_partial_queries()
    indexed_results = list(text_completer.complete_partial_queries(partial_queries))
    predictions = _extract_predictions(indexed_results)

    # Assert
    assert predictions == expected_predictions

    # Act
    verbose_partial_queries = text_completion_task.generate_partial_queries(verbose=True)
    indexed_results = list(text_completer.complete_partial_queries(verbose_partial_queries))
    predictions = _extract_predictions(indexed_results)

    # Assert
    assert predictions == expected_predictions