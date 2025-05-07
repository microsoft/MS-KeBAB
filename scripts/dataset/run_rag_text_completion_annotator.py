# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script demonstrates the usage of a RAG text completer, specifically the Phi model, for a text
completion task. It generates partial queries, completes them using the Phi model, evaluates the
results, and outputs evaluation metrics. Additionally, it annotates the documents by highlighting
words that are challenging to predict based on their log probabilities.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from kebab.tasks.text_completion import TextCompletionTask
from kebab.utils.rag_text_completer import BaseRAGTextCompleter, PhiRAGTextCompleter


if __name__ == "__main__":
    # Create a text completion task instance.
    documents_file_path = Path(__file__).parents[2] / "tests" / "data" / "text_completion" / "documents.jsonl"
    task_instance = TextCompletionTask(
        "TextCompletion",
        str(documents_file_path),
    )

    # Prepare partial queries.
    queries = task_instance.generate_partial_queries(verbose=True)

    # Run the text completer.
    phi_annotator = PhiRAGTextCompleter()
    results = list(
        phi_annotator.complete_partial_queries(
            partial_queries=queries,
            get_augmented_context=lambda _: "",  # A dummy RAG that always returns empty context.
            verbose=True,
        )
    )

    results_to_evaluate = []
    for result in results:
        if result["text_with_mask"] != "":
            predicted_content = result["predicted_content"]
            if result["target_content_logprob"] != float("-inf"):
                results_to_evaluate.append((predicted_content, result["target_content_logprob"]))
            else:
                # Fallback to the smallest log probability among the top k predicted tokens when the
                # target content log probability is -inf.
                results_to_evaluate.append(
                    (predicted_content, result["predicted_content_top_logprobs"][0][-1]["logprob"])
                )

    # Write the results to a file.
    output_to_evaulate_path = Path(os.path.splitext(documents_file_path)[0] + "_tc_results_to_eval.jsonl")
    task_instance.write_items(output_to_evaulate_path, results_to_evaluate)

    # Evluate the results.
    task_instance.evaluate(
        output_to_evaluate=output_to_evaulate_path,
        eval_result_path=Path(os.path.splitext(documents_file_path)[0] + "_tc_metrics.json"),
    )

    # Write the full output for debugging.
    output_path = os.path.splitext(documents_file_path)[0] + "_tc_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Annotate each document with log probabilities.
    results_grouped_by_doc = defaultdict(list)
    for result in results:
        results_grouped_by_doc[result["document_id"]].append(result)
    for doc_id, results_per_doc in results_grouped_by_doc.items():
        BaseRAGTextCompleter.generate_annotated_doc_html(
            text_completion_results=results_per_doc,
            output_path=os.path.splitext(documents_file_path)[0] + f"_annotated_{doc_id}.html",
        )
