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

from kebab.tasks.text_completion import TextCompletionUsingDocumentsTask
from kebab.utils.rag_text_completer import (
    BaseGptOssRAGTextCompleter,
    BasePhiRAGTextCompleter,
    BaseQwenRAGTextCompleter,
    BaseRAGTextCompleter,
)


def process_results(results: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, float, list[list[str | float]]]]:
    """Process completion results and extract prediction data.

    Args:
        results: List of completion result dictionaries.

    Returns:
        List of tuples containing (predicted_content, target_content_logprob, predicted_content_top_logprobs).
    """
    results_to_eval = []
    for _, result in results:
        if result["text_with_mask"] != "" and result.get("to_eval", True):
            predicted_content = result["predicted_content"]
            target_content_logprob = BaseRAGTextCompleter.get_target_content_logprob_with_fallback(result)
            predicted_content_top_logprobs = [
                [item["token"], item["logprob"]] for item in result["predicted_content_top_logprobs"][0]
            ]
            results_to_eval.append((predicted_content, target_content_logprob, predicted_content_top_logprobs))
        if "to_eval" in result:
            result["to_eval"] = "True" if result["to_eval"] else "False"  # Convert to str.
    return results_to_eval


if __name__ == "__main__":
    # Create a text completion task instance.
    documents_file_path = Path(__file__).parents[2] / "tests" / "data" / "text_completion" / "documents.jsonl"
    task_instance = TextCompletionUsingDocumentsTask(
        "TextCompletion",
        documents_file_path,
    )

    # Prepare partial queries for each word.
    queries = task_instance.generate_partial_queries(verbose=True)

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
    # Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B, Qwen/Qwen3-8B, Qwen/Qwen3-14B, Qwen/Qwen3-32B
    # base_model = DummyQwenRAGTextCompleter(
    #     model_id="Qwen/Qwen3-14B", prediction_method=PredictionMethod.LOGPROBS_FROM_TEXT_RESPONSE
    # )
    # Examples of using other models:
    base_model = DummyPhiRAGTextCompleter()
    # base_model = DummyGptOssRAGTextCompleter(
    #     model_id="openai/gpt-oss-20b", gpu_id=3, prediction_method=PredictionMethod.LOGPROBS_FROM_TEXT_RESPONSE
    # )

    # Run the completion task with the base model on each word.
    base_results = process_results(list(
        base_model.complete_partial_queries(
            partial_queries=queries,
            verbose=True,
        )
    ))

    # Write the base results to a file.
    base_results_for_threshold_calculation_path = Path(
        os.path.splitext(documents_file_path)[0] + "_tc_base_results_for_t_calc.jsonl"
    )
    task_instance.write_items(base_results_for_threshold_calculation_path, base_results, strict=False)

    # Prepare partial queries with to_eval flags based on threshold calculation from base results.
    queries_with_to_eval_flags = task_instance.generate_partial_queries_to_eval(
        base_predictions=base_results_for_threshold_calculation_path,
        verbose=True,
    )

    # For demonstration, we use the same base model for evaluation. In practice, you can use a different model for evaluation if desired.
    model_to_eval = base_model

    # Run the completion task with the model to evaluate.
    results = process_results(list(
        model_to_eval.complete_partial_queries(
            partial_queries=queries_with_to_eval_flags,
            verbose=True,
        )
    ))

    # Write the results to a file for evaluation.
    results_to_evaulate_path = Path(os.path.splitext(documents_file_path)[0] + "_tc_results_to_eval.jsonl")
    task_instance.write_items(results_to_evaulate_path, results, strict=False)

    # Evaluate the results.
    task_instance.fair_keyword_evaluate(
        to_eval_predictions=results_to_evaulate_path,
        base_predictions=base_results_for_threshold_calculation_path,
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

    for _, result in results:
        results_grouped_by_doc[result["document_id"]].append(result)
    for doc_id, results_per_doc in results_grouped_by_doc.items():
        BaseRAGTextCompleter.generate_annotated_doc_html(
            text_completion_results=results_per_doc,
            output_path=os.path.splitext(documents_file_path)[0] + f"_base_annotated_{doc_id}.html",
        )
