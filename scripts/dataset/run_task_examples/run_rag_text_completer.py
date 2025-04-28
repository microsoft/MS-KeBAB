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
    documents_file_path = Path(__file__).parents[3] / "tests" / "data" / "text_completion" / "documents.jsonl"
    predictions_file_path = Path(__file__).parents[3] / "tests" / "data" / "text_completion" / "predicted_contents.jsonl"
    predictions_output_file_path = Path(__file__).parents[3] / "tests" / "output" / "text_completion" / "predicted_contents.jsonl"
    task_instance = TextCompletionTask(
        "TextCompletion",
        str(documents_file_path),
    )

    queries = task_instance.generate_partial_queries(verbose=True)
    queries = BaseRAGTextCompleter.augment_partial_queries_with_contexts(queries)
    phi_annotator = PhiRAGTextCompleter()
    results = phi_annotator.complete_text_and_evaluate(queries, verbose=True)

    output_path = os.path.splitext(documents_file_path)[0] + "_tc_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    doc_id = "Invalid"
    results_per_doc = []
    for result in results["all_results"]:
        if result["document_id"] != doc_id and doc_id != "Invalid":
            BaseRAGTextCompleter.generate_annotated_doc_html(results_per_doc, os.path.splitext(documents_file_path)[0] + f"_annotated_{doc_id}.html")
            results_per_doc = [result]
        else:
            results_per_doc.append(result)
        doc_id = result["document_id"]
    if results_per_doc:
        BaseRAGTextCompleter.generate_annotated_doc_html(results_per_doc, os.path.splitext(documents_file_path)[0] + f"_annotated_{doc_id}.html")
        results_per_doc = []
