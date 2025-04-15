# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Iterable
from pathlib import Path

from kebab.contracts.document import Document
from kebab.contracts.task import Task, TaskType
from kebab.utils.io_helpers import DocumentJsonlReader, ItemJsonlReader, save_dict_to_json


class TextCompletionTask(Task):
    """Represents a text completion benchmark task with its data files."""

    __data_documents: Path

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.TextCompletion

    @property
    def data_documents(self) -> Path:
        """Return path to documents."""
        return self.__data_documents

    def __init__(
        self,
        name: str,
        documents: str,
    ):
        """Initialize a linking task."""
        super().__init__(name, schema="")
        self.__data_documents = Path(documents)

    def read_items(self) -> Iterable[Document]:
        return DocumentJsonlReader(self.__data_documents).read_items()

    def generate_partial_queries(self) -> Iterable[dict[str, str]]:
        docs = self.read_items()
        for doc in docs:
            text = doc.data["text"]
            words = re.findall(r"\w+|\s+|[^\w\s]", text)
            for i, word in enumerate(words):
                if i == 0 or not word.strip().isalnum():
                    continue
                text_with_mask = "".join(words[ : i] + ["<mask>"])
                yield {
                    "text_with_mask": text_with_mask,
                    "target_content": word,
                    "document_id": doc.document_id,
                }

    def write_items(self, path: Path, items: Iterable[tuple[str, float]]) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            for predicted_content, target_content_logprob in items:
                json_line = json.dumps(
                    {
                        "predicted_content": predicted_content,
                        "target_content_logprob": target_content_logprob,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                file.write(json_line + "\n")

    def evaluate(
        self,
        output_to_evaluate: Path,
        eval_result_path: Path | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the text completion task."""
        predictions = ItemJsonlReader[dict[str, str | float]](output_to_evaluate).read_items()
        log_probs = [float(pred["target_content_logprob"]) for pred in predictions]

        metrics = {}
        if log_probs:
            metrics["mean_log_prob"] = statistics.mean(log_probs)
            metrics["variance_log_prob"] = statistics.variance(log_probs, metrics["mean_log_prob"])
            metrics["perplexity"] = math.exp(-metrics["mean_log_prob"])
        else:
            raise ValueError("No log probabilities found in predictions.")

        if eval_result_path:
            save_dict_to_json(metrics, eval_result_path)

        return metrics
