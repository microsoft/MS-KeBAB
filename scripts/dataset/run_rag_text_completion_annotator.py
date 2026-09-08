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
from collections.abc import Iterable
from pathlib import Path
from typing import Any, override

from kebab.tasks.metrics.text_completion.alternating_estimator import ModelInfo
from kebab.tasks.metrics.text_completion.clipped_perplexity_scorer import ClippedPerplexityScorer
from kebab.tasks.text_completion import EvaluationMethod, TextCompletionUsingDocumentsTask
from kebab.utils.rag_text_completer import (
    BaseGptOssRAGTextCompleter,
    BasePhiRAGTextCompleter,
    BaseQwenRAGTextCompleter,
    BaseRAGTextCompleter,
)


def run_perplexity_evaluation_example(
    documents_file_path: Path,
    task_instance: TextCompletionUsingDocumentsTask,
    base_results: Iterable[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Run an example evaluation using the perplexity evaluation method."""
    # Use the base results as an example for perplexity evaluation. In practice, you can run a different model for evaluation if desired.
    results = base_results

    # Write the results to a file for evaluation.
    results_to_evaluate_path = Path(os.path.splitext(documents_file_path)[0] + "_tc_results_for_perplexity_eval.jsonl")
    task_instance.write_items(results_to_evaluate_path, results, strict=False)

    # Evaluate the results.
    metrics = task_instance.evaluate(
        predictions=results_to_evaluate_path,
        result_output_path=Path(os.path.splitext(documents_file_path)[0] + "_tc_metrics_perplexity.json"),
        method=EvaluationMethod.PERPLEXITY,
    )

    return metrics


def run_clipped_perplexity_evaluation_example(
    documents_file_path: Path,
    task_instance: TextCompletionUsingDocumentsTask,
    base_model: BaseRAGTextCompleter,
    base_results: Iterable[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Run an example evaluation using the clipped perplexity evaluation method."""
    # Write the base results to a file.
    base_results_for_threshold_calculation_path = Path(
        os.path.splitext(documents_file_path)[0] + "_tc_base_results_for_clipped_perplexity_t_calc.jsonl"
    )
    task_instance.write_items(base_results_for_threshold_calculation_path, base_results, strict=False)

    # Initialize the scorer with the base predictions to evaluate candidate models.
    task_instance.scorer = ClippedPerplexityScorer(base_predictions=base_results_for_threshold_calculation_path)

    # Prepare partial queries with to_eval flags based on threshold calculation from base results.
    queries_with_to_eval_flags = task_instance.generate_partial_queries_to_eval(
        verbose=True,
    )

    # For demonstration, we use the same base model for evaluation. In practice, you can use a different model for evaluation if desired.
    model_to_eval = base_model

    # Run the completion task with the model to evaluate.
    results = list(
        model_to_eval.complete_partial_queries(
            partial_queries=queries_with_to_eval_flags,
            verbose=True,
        )
    )

    # Write the results to a file for evaluation.
    results_to_evaulate_path = Path(
        os.path.splitext(documents_file_path)[0] + "_tc_results_for_clipped_perplexity_eval.jsonl"
    )
    task_instance.write_items(results_to_evaulate_path, results, strict=False)

    # Evaluate the results.
    metrics = task_instance.evaluate(
        predictions=results_to_evaulate_path,
        result_output_path=Path(os.path.splitext(documents_file_path)[0] + "_tc_metrics_clipped_perplexity.json"),
        method=EvaluationMethod.CLIPPED_PERPLEXITY,
    )

    # Write the full results for debugging.
    verbose_base_results_output_path = Path(
        os.path.splitext(documents_file_path)[0] + "_tc_verbose_base_results_for_clipped_perplexity.json"
    )
    task_instance.write_items(
        path=verbose_base_results_output_path,
        items=base_results,
        verbose=True,
    )
    verbose_results_output_path = Path(
        os.path.splitext(documents_file_path)[0] + "_tc_verbose_results_for_clipped_perplexity.json"
    )
    task_instance.write_items(
        path=verbose_results_output_path,
        items=results,
        verbose=True,
    )

    # Evaluate the verbose results.
    metrics_from_verbose = task_instance.evaluate(
        predictions=verbose_results_output_path,
        result_output_path=Path(
            os.path.splitext(documents_file_path)[0] + "_tc_metrics_from_verbose_for_clipped_perplexity.json"
        ),
        method=EvaluationMethod.CLIPPED_PERPLEXITY,
    )
    assert metrics == metrics_from_verbose, (
        "Metrics from verbose results should match metrics from non-verbose results."
    )

    return metrics


def run_alternating_evaluation_example(
    documents_file_path: Path,
    task_instance: TextCompletionUsingDocumentsTask,
    base_results: Iterable[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Run an example evaluation using the alternating estimation method."""
    # Use the base results as an example for alternating estimation evaluation. In practice, you can run a different model for evaluation if desired.
    results = base_results

    # Write the results to a file for evaluation.
    results_to_evaluate_path = Path(os.path.splitext(documents_file_path)[0] + "_tc_results_for_alternating_eval.jsonl")
    task_instance.write_items(results_to_evaluate_path, results, verbose=True)

    good_model_infos = {}
    good_model_infos["phi_RAG_0_1"] = ModelInfo(predictions_path=results_to_evaluate_path)
    bad_model_infos = {}
    bad_model_infos["phi_RAG_0_-1"] = ModelInfo(predictions_path=results_to_evaluate_path)
    dont_know_model_infos = {}
    dont_know_model_infos["phi_RAG_0_0"] = ModelInfo(predictions_path=results_to_evaluate_path)

    # Evaluate the results.
    metrics = task_instance.alternating_evaluate(
        good_model_infos=good_model_infos,
        bad_model_infos=bad_model_infos,
        dont_know_model_infos=dont_know_model_infos,
    )

    return metrics


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
    base_results = list(
        base_model.complete_partial_queries(
            partial_queries=queries,
            verbose=True,
        )
    )

    clipped_perplexity_metrics = run_clipped_perplexity_evaluation_example(
        documents_file_path=documents_file_path,
        task_instance=task_instance,
        base_model=base_model,
        base_results=base_results,
    )
    print("Clipped Perplexity Evaluation Metrics:", clipped_perplexity_metrics)

    perplexity_metrics = run_perplexity_evaluation_example(
        documents_file_path=documents_file_path,
        task_instance=task_instance,
        base_results=base_results,
    )
    print("Perplexity Evaluation Metrics:", perplexity_metrics)

    alternating_estimation_metrics = run_alternating_evaluation_example(
        documents_file_path=documents_file_path,
        task_instance=task_instance,
        base_results=base_results,
    )
    print("Alternating Evaluation Metrics:", alternating_estimation_metrics)

    # Annotate each document with log probabilities.
    results_grouped_by_doc = defaultdict(list)
    for _, result in base_results:
        results_grouped_by_doc[result["document_id"]].append(result)
    for doc_id, results_per_doc in results_grouped_by_doc.items():
        BaseRAGTextCompleter.generate_annotated_doc_html(
            text_completion_results=results_per_doc,
            output_path=os.path.splitext(documents_file_path)[0] + f"_base_annotated_{doc_id}.html",
        )
