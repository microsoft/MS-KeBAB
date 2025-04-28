# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""TODO"""

from __future__ import annotations

import json
import os
from pathlib import Path

from kebab.tasks.text_completion import TextCompletionTask
from kebab.utils.rag_text_completer import BaseRAGTextCompleter, PhiRAGTextCompleter


if __name__ == "__main__":
    # Create a text completion task instance
    documents_file_path = Path(__file__).parents[3] / "tests" / "data" / "text_completion" / "documents.jsonl"
    predictions_file_path = Path(__file__).parents[3] / "tests" / "data" / "text_completion" / "predicted_contents.jsonl"
    predictions_output_file_path = Path(__file__).parents[3] / "tests" / "output" / "text_completion" / "predicted_contents.jsonl"
    task_instance = TextCompletionTask(
        "TextCompletion",
        str(documents_file_path),
    )

    # Prepare partial queries
    queries = task_instance.generate_partial_queries(verbose=True)

    # Run RAG to augment partial queries with context
    queries = BaseRAGTextCompleter.augment_partial_queries_with_contexts(
        queries=queries,
        get_augmented_context=lambda _: "",
    )

    # Run the text completer
    phi_annotator = PhiRAGTextCompleter()
    results = list(phi_annotator.complete_partial_queries(queries, verbose=True))

    output_path = os.path.splitext(documents_file_path)[0] + "_tc_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    doc_id = "Invalid"
    results_per_doc = []
    for result in results:
        if result["document_id"] != doc_id and doc_id != "Invalid":
            BaseRAGTextCompleter.generate_annotated_doc_html(results_per_doc, os.path.splitext(documents_file_path)[0] + f"_annotated_{doc_id}.html")
            results_per_doc = [result]
        else:
            results_per_doc.append(result)
        doc_id = result["document_id"]
    if results_per_doc:
        BaseRAGTextCompleter.generate_annotated_doc_html(results_per_doc, os.path.splitext(documents_file_path)[0] + f"_annotated_{doc_id}.html")
        results_per_doc = []

    results_to_evaluate = []
    for result in results:
        if result["text_with_mask"] != "":
            predicted_content = result["target_content_top_logprobs"][0][0]["token"]
            if result["target_content_logprob"] != float("-inf"):
                results_to_evaluate.append((predicted_content, result["target_content_logprob"]))
            else:
                results_to_evaluate.append((predicted_content, result["target_content_top_logprobs"][0][-1]["logprob"]))

    # Write the results to a file
    output_to_evaulate_path = Path(os.path.splitext(documents_file_path)[0] + "_tc_results_to_eval.jsonl")
    task_instance.write_items(output_to_evaulate_path, results_to_evaluate)

    # Evluate the results
    task_instance.evaluate(
        output_to_evaluate=output_to_evaulate_path,
        eval_result_path=Path(os.path.splitext(documents_file_path)[0] + "_tc_metrics.json"),
    )
