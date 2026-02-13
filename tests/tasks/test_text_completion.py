# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from pathlib import Path

import pytest
from kebab.tasks.text_completion import TextCompletionUsingDocumentsTask
from kebab.utils.io_helpers import compare_files_ignore_linebreaks


@pytest.fixture
def text_completion_task() -> TextCompletionUsingDocumentsTask:
    documents_file_path = Path(__file__).parents[1] / "data" / "text_completion" / "documents.jsonl"
    return TextCompletionUsingDocumentsTask(
        "TextCompletion",
        documents_file_path,
    )


@pytest.fixture
def email_text_completion_task() -> TextCompletionUsingDocumentsTask:
    emails_file_path = Path(__file__).parents[1] / "data" / "text_completion" / "emails.jsonl"
    return TextCompletionUsingDocumentsTask(
        "TextCompletion",
        emails_file_path,
    )


def test_text_completion_read_items_and_generate_queries(text_completion_task, tmp_path: Path) -> None:
    # Arrange
    predictions_file_path = Path(__file__).parents[1] / "data" / "text_completion" / "document_predictions.jsonl"
    predictions_output_file_path = tmp_path / "document_predictions.jsonl"

    # Act
    items = list(text_completion_task.read_items())
    partial_queries = list(text_completion_task.generate_partial_queries())
    text_completion_task.write_items(
        predictions_output_file_path,
        [
            {
                "predicted_content": query["target_content"],
                "target_content_logprob": -0.1,
                "predicted_content_top_logprobs": [[query["target_content"], -0.1]],
            }
            for query in partial_queries
        ],
    )

    metrics = text_completion_task.evaluate(predictions_output_file_path)

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
    assert compare_files_ignore_linebreaks(
        predictions_file_path,
        predictions_output_file_path,
    )
    assert metrics["mean_log_prob"] == -0.1
    assert metrics["variance_log_prob"] == 0.0
    assert math.isclose(metrics["perplexity"], 1.105, abs_tol=1e-3)


def test_text_completion_generate_verbose_partial_queries(text_completion_task, tmp_path: Path) -> None:
    # Arrange
    predictions_file_path = Path(__file__).parents[1] / "data" / "text_completion" / "document_predictions_verbose.json"
    predictions_output_file_path = tmp_path / "document_predictions_verbose_test.json"

    # Act
    verbose_partial_queries = list(text_completion_task.generate_partial_queries(verbose=True))
    text_completion_task.write_items(
        predictions_output_file_path,
        [
            query
            if query["text_with_mask"] == ""
            else query
            | {
                "predicted_content": query["target_content"],
                "target_content_logprob": -0.1,
                "predicted_content_top_logprobs": [[query["target_content"], -0.1]],
            }
            for query in verbose_partial_queries
        ],
        verbose=True,
    )

    # Assert
    # The verbose partial queries should include skipped words/spaces/punctuations with empty
    # text_with_mask.
    assert verbose_partial_queries[:12] == [
        {"text_with_mask": "", "target_content": "The", "document_id": "doc_0"},
        {"text_with_mask": "", "target_content": " ", "document_id": "doc_0"},
        {"text_with_mask": "The <mask>", "target_content": "capital", "document_id": "doc_0"},
        {"text_with_mask": "", "target_content": " ", "document_id": "doc_0"},
        {"text_with_mask": "The capital <mask>", "target_content": "of", "document_id": "doc_0"},
        {"text_with_mask": "", "target_content": " ", "document_id": "doc_0"},
        {"text_with_mask": "The capital of <mask>", "target_content": "France", "document_id": "doc_0"},
        {"text_with_mask": "", "target_content": " ", "document_id": "doc_0"},
        {"text_with_mask": "The capital of France <mask>", "target_content": "is", "document_id": "doc_0"},
        {"text_with_mask": "", "target_content": " ", "document_id": "doc_0"},
        {"text_with_mask": "The capital of France is <mask>", "target_content": "Paris", "document_id": "doc_0"},
        {"text_with_mask": "", "target_content": ".", "document_id": "doc_0"},
    ]
    assert compare_files_ignore_linebreaks(
        predictions_file_path,
        predictions_output_file_path,
    )


def test_text_completion_generate_partial_queries_with_words_after_mask(text_completion_task) -> None:
    # Act
    partial_queries = list(text_completion_task.generate_partial_queries(words_after_mask=1))

    # Assert
    # The partial queries should include non-whitespace words/punctuations after the mask.
    assert partial_queries == [
        {"text_with_mask": "The <mask> of", "target_content": "capital", "document_id": "doc_0"},
        {"text_with_mask": "The capital <mask> France", "target_content": "of", "document_id": "doc_0"},
        {"text_with_mask": "The capital of <mask> is", "target_content": "France", "document_id": "doc_0"},
        {"text_with_mask": "The capital of France <mask> Paris", "target_content": "is", "document_id": "doc_0"},
        {"text_with_mask": "The capital of France is <mask>.", "target_content": "Paris", "document_id": "doc_0"},
        {"text_with_mask": "The <mask> of", "target_content": "capital", "document_id": "doc_1"},
        {"text_with_mask": "The capital <mask> China", "target_content": "of", "document_id": "doc_1"},
        {"text_with_mask": "The capital of <mask> is", "target_content": "China", "document_id": "doc_1"},
        {"text_with_mask": "The capital of China <mask> Beijing", "target_content": "is", "document_id": "doc_1"},
        {"text_with_mask": "The capital of China is <mask>.", "target_content": "Beijing", "document_id": "doc_1"},
    ]


def test_email_text_completion_read_items_and_generate_queries(email_text_completion_task) -> None:
    # Act
    items = list(email_text_completion_task.read_items())
    partial_queries = list(email_text_completion_task.generate_partial_queries())

    # Assert
    assert items[0].document_id == "email_0"
    assert items[0].data["subject"] == "Hello"
    assert items[0].data["body"] == "Is China the most populous country?"
    assert items[0].data["date"] == "2024-04-25T00:00:00Z"
    assert items[1].document_id == "email_1"
    assert items[1].data["subject"] == "Re: Hello"
    assert items[1].data["body"] == "No, India has the largest population."
    assert items[1].data["date"] == "2024-04-25T12:00:00Z"
    assert partial_queries[-1] == {
        "text_with_mask": "No, India has the largest <mask>",
        "target_content": "population",
        "document_id": "email_1",
    }


def test_calculate_thresholds(text_completion_task) -> None:
    # Act
    predictions_file_path = Path(__file__).parents[1] / "data" / "text_completion" / "document_predictions.jsonl"
    queries_with_to_eval_flags = list(text_completion_task.generate_partial_queries_to_eval(predictions_file_path))

    # Assert
    assert all(not query["to_eval"] for query in queries_with_to_eval_flags)
