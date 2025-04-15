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
        """Initialize a text completion task."""
        super().__init__(name, schema="")  # kb schema is not used in this task
        self.__data_documents = Path(documents)

    def read_items(self) -> Iterable[Document]:
        """
        Read data items from a JSONL file containing documents.

        Returns:
            Iterable[Document]: An iterable of `Document` objects.
        """
        return DocumentJsonlReader(self.__data_documents).read_items()

    def generate_partial_queries(self) -> Iterable[dict[str, str]]:
        """
        Generate partial queries for participants to fill in.

        Returns:
            Iterable[dict[str, str]]: An iterable of dicts, where each dictionary contains:
                - "text_with_mask": The text with a chunk replaced by "<mask>".
                - "target_content": The content of the chunk that was replaced by "<mask>".
                - "document_id": The ID of the document being processed.
        """
        docs = self.read_items()
        for doc in docs:
            text = doc.data["text"]
            # Split text into alphanumeric words, consecutive spaces/tabs, and punctuations.
            # TODO (allenwang-ms): Use a more sophisticated tokenizer for better handling of
            # punctuations, special characters, non-Latin scripts, etc.
            words = re.findall(r"\w+|\s+|[^\w\s]", text)
            for i, word in enumerate(words):
                # Skip the first word and non-alphanumeric words.
                if i == 0 or not word.strip().isalnum():
                    continue
                text_with_mask = "".join(words[:i] + ["<mask>"])
                yield {
                    "text_with_mask": text_with_mask,
                    "target_content": word,
                    "document_id": doc.document_id,
                }

    def write_items(self, path: Path, items: Iterable[tuple[str, float]]) -> None:
        """
        Write prediction items to the specified path.

        Args:
            path: The file path where the prediction items should be written.
            items: An iterable of tuples, where each tuple contains:
                - predicted_content: The predicted content.
                - target_content_logprob: The log probability of the target content.
        """
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

        # TODO (allenwang-ms): Consider adding more evaluation metrics, such as BLEU, to assess the
        # quality of predicted contents.

        if eval_result_path:
            save_dict_to_json(metrics, eval_result_path)

        return metrics
