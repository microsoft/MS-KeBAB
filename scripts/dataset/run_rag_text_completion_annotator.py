# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script demonstrates the usage of a RAG text completer, specifically the Phi model, for a text
completion task. It generates partial queries, completes them using the Phi model, evaluates the
results, and outputs evaluation metrics. Additionally, it annotates the documents by highlighting
words that are challenging to predict based on their log probabilities.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, override

from kebab.tasks.metrics.text_completion.fair_keyword_scorer import FairKeywordScorer
from kebab.tasks.text_completion import TextCompletionUsingDocumentsTask
from kebab.utils.rag_text_completer import (
    BaseGptOssRAGTextCompleter,
    BasePhiRAGTextCompleter,
    BaseQwenRAGTextCompleter,
    BaseRAGTextCompleter,
)


if __name__ == "__main__":
    # Create a text completion task instance.
    documents_file_path = Path(__file__).parents[2] / "tests" / "data" / "text_completion" / "documents.jsonl"
    task_instance = TextCompletionUsingDocumentsTask(
        "TextCompletion",
        documents_file_path,
    )

    # Prepare partial queries for each word.
    queries = list(task_instance.generate_partial_queries(verbose=True))

    class DummyPhiRAGTextCompleter(BasePhiRAGTextCompleter):
        """A local class with a dummy RAG function."""

        @override
        def get_augmented_context(self, query: dict[str, Any]) -> str:
            return ""

    class DummyGptOssRAGTextCompleter(BaseGptOssRAGTextCompleter):
        """A local class with a dummy RAG function."""

        @override
        def get_augmented_context(self, query: dict[str, Any]) -> str:
            return ""

    class DummyQwenRAGTextCompleter(BaseQwenRAGTextCompleter):
        """A local class with a dummy RAG function."""

        @override
        def get_augmented_context(self, query: dict[str, Any]) -> str:
            return ""

    # Run the text completer.
    base_model = DummyPhiRAGTextCompleter()
    # Examples of using other models:
    # Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B, Qwen/Qwen3-8B, Qwen/Qwen3-14B, Qwen/Qwen3-32B
    # base_model = DummyQwenRAGTextCompleter(
    #     model_id="Qwen/Qwen3-14B", prediction_method=PredictionMethod.LOGPROBS_FROM_TEXT_RESPONSE
    # )
    # openai/gpt-oss-20b, openai/gpt-oss-120b
    # base_model = DummyGptOssRAGTextCompleter(
    #     model_id="openai/gpt-oss-20b", gpu_id=3, prediction_method=PredictionMethod.LOGPROBS_FROM_TEXT_RESPONSE
    # )

    # Run the completion task with the base model on each word.
    base_results = [
        result
        for _, result in sorted(
            base_model.complete_partial_queries(
                partial_queries=queries,
                verbose=True,
            ),
            key=lambda x: int(x[0]),
        )
    ]

    # Write the base results to a file.
    base_results_for_threshold_calculation_path = Path(
        os.path.splitext(documents_file_path)[0] + "_tc_base_results_for_t_calc.jsonl"
    )
    task_instance.write_items(base_results_for_threshold_calculation_path, base_results, strict=False)

    # Initialize the scorer with the base predictions to evaluate candidate models.
    task_instance.scorer = FairKeywordScorer(base_predictions=base_results_for_threshold_calculation_path)

    # Prepare partial queries with to_eval flags based on threshold calculation from base results.
    queries_with_to_eval_flags = task_instance.generate_partial_queries_to_eval(
        verbose=True,
    )

    # For demonstration, we use the same base model for evaluation. In practice, you can use a different model for evaluation if desired.
    model_to_eval = base_model

    # Run the completion task with the model to evaluate.
    results = [
        result
        for _, result in sorted(
            model_to_eval.complete_partial_queries(
                partial_queries=queries_with_to_eval_flags,
                verbose=True,
            ),
            key=lambda x: int(x[0]),
        )
    ]

    # Write the results to a file for evaluation.
    results_to_evaulate_path = Path(os.path.splitext(documents_file_path)[0] + "_tc_results_to_eval.jsonl")
    task_instance.write_items(results_to_evaulate_path, results, strict=False)

    # Evaluate the results.
    task_instance.fair_keyword_evaluate(
        to_eval_predictions=results_to_evaulate_path,
        metrics_output_path=Path(os.path.splitext(documents_file_path)[0] + "_tc_metrics.json"),
    )

    # Write the full results for debugging.
    verbose_base_results_output_path = Path(os.path.splitext(documents_file_path)[0] + "_tc_verbose_base_results.json")
    task_instance.write_items(
        path=verbose_base_results_output_path,
        items=base_results,
        verbose=True,
    )
    verbose_results_output_path = Path(os.path.splitext(documents_file_path)[0] + "_tc_verbose_results.json")
    task_instance.write_items(
        path=verbose_results_output_path,
        items=results,
        verbose=True,
    )

    # Annotate each document with log probabilities.
    results_grouped_by_doc = defaultdict(list)
    for result in results:
        results_grouped_by_doc[result["document_id"]].append(result)
    for doc_id, results_per_doc in results_grouped_by_doc.items():
        BaseRAGTextCompleter.generate_annotated_doc_html(
            text_completion_results=results_per_doc,
            output_path=os.path.splitext(documents_file_path)[0] + f"_base_annotated_{doc_id}.html",
        )
