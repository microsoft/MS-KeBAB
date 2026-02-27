# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
import math
import re
import statistics
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from logging import Logger
from pathlib import Path
from typing import Any, ClassVar

from kebab.contracts.document import Document
from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.utils.io_helpers import (
    DocumentJsonlReader,
    EntityJsonlReader,
    ItemJsonlReader,
    resolve_path,
    save_dict_to_json,
)


class TextCompletionTaskBase(Task):
    """Base class for text completion benchmark task."""

    MASK: str = "<mask>"
    """The token used to mask the text in text completion tasks."""

    SCHEMA_ID_TO_TEXT_FIELD_MAP: ClassVar[dict[str, str]] = {
        "plain_text": "text",
        "email": "body",
    }
    """Mapping from schemas to the corresponding text field names."""

    __documents: Path

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Return task type."""

    @property
    def documents(self) -> Path:
        """Return path to documents."""
        return self.__documents

    def __init__(
        self,
        name: str,
        documents: Path,
        schema: Path | None = None,
        root_for_relative_paths: Path | None = None,
    ):
        """Initialize a text completion task."""
        super().__init__(name, schema=schema, root_for_relative_paths=root_for_relative_paths)
        self.__documents = resolve_path(documents, root_for_relative_paths)

    def read_items(self) -> Iterable[Document]:
        """
        Read data items from a JSONL file containing documents.

        Returns:
            Iterable[Document]: An iterable of `Document` objects.
        """
        return DocumentJsonlReader(self.__documents).read_items()

    def generate_partial_queries(self, verbose: bool = False, words_after_mask: int = 0) -> Iterable[dict[str, str]]:
        """
        Generate partial queries for participants to fill in.

        Args:
            verbose: Defaults to False. If True, includes skipped words with an empty
            "text_with_mask" field for debugging.
            words_after_mask: Defaults to 0. Number of non-whitespace words/tokens to include
            after the mask position.

        Returns:
            Iterable[dict[str, str]]: An iterable of dicts, where each dictionary contains:
                - "text_with_mask": The text with a chunk replaced by "<mask>".
                - "target_content": The content of the chunk that was replaced by "<mask>".
                - "document_id": The ID of the document being processed.
        """
        max_word_length = 30
        docs = self.read_items()
        for doc in docs:
            text = doc.data[TextCompletionTaskBase.SCHEMA_ID_TO_TEXT_FIELD_MAP[doc.schema.schema_id]]
            # Split text into alphanumeric words, consecutive spaces/tabs, and punctuations.
            # TODO (allenwang-ms): Use a more sophisticated tokenizer for better handling of
            # punctuations, special characters, non-Latin scripts, etc.
            words = re.findall(r"\w+|\s+|[^\w\s]", text)
            for i, word in enumerate(words):
                # Skip the first word, non-alphanumeric words, and long words.
                if i == 0 or not word.isalnum() or len(word) > max_word_length:
                    if verbose:
                        yield {
                            "text_with_mask": "",
                            "target_content": word,
                            "document_id": doc.document_id,
                        }
                    continue
                # Find the end index by counting non-whitespace words/tokens after the mask.
                end_index = i + 1
                word_count = 0
                while end_index < len(words) and word_count < words_after_mask:
                    if words[end_index].strip():
                        word_count += 1
                    end_index += 1
                text_with_mask = "".join([*words[:i], TextCompletionTaskBase.MASK, *words[i + 1 : end_index]])
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
            for item in items:
                if item is None:
                    json_line = json.dumps(None)
                else:
                    predicted_content, target_content_logprob = item
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
        predictions: Path,
        result_output_path: Path | None = None,
        logger: Logger | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the text completion task."""
        if logger:
            logger.info("Starting evaluation for the text completion task.")

        predicted_vals = list(ItemJsonlReader[dict[str, str | float]](predictions).read_items())
        queries = list(self.generate_partial_queries())
        print(len(predicted_vals), len(queries))
        predictions_and_queries = zip(predicted_vals, queries, strict=True)

        log_probs = []
        log_probs_by_doc = defaultdict(list)
        for prediction, query in predictions_and_queries:
            log_prob = None if prediction is None else float(prediction["target_content_logprob"])
            log_probs.append(log_prob)
            log_probs_by_doc[query["document_id"]].append(log_prob)

        metrics = {}
        if log_probs:
            TextCompletionTaskBase.__calculate_metrics(metrics, log_probs)
        else:
            raise ValueError("No log probabilities found in predictions.")

        metrics_by_doc = []
        for doc_id, log_probs_per_doc in log_probs_by_doc.items():
            metrics_per_doc = {}
            if log_probs_per_doc:
                metrics_per_doc["document_id"] = doc_id
                TextCompletionTaskBase.__calculate_metrics(metrics_per_doc, log_probs_per_doc)
            else:
                raise ValueError(f"No log probabilities found for document {doc_id}.")
            metrics_by_doc.append(metrics_per_doc)
        metrics["metrics_by_doc"] = metrics_by_doc

        # TODO (allenwang-ms): Consider adding more evaluation metrics, such as BLEU, to assess the
        # quality of predicted contents.

        if logger:
            logger.info("Evaluation metrics calculated successfully.")
            logger.info(f"Metrics: {metrics}")
        if result_output_path:
            save_dict_to_json(metrics, result_output_path)

        return metrics

    @staticmethod
    def __calculate_metrics(metrics: dict[str, Any], log_probs: list[float]) -> None:
        """
        Fill the metrics dictionary based on the provided log probabilities.

        Args:
            metrics: The dictionary to fill with metrics.
            log_probs: A list of log probability values.
        """
        metrics["mean_log_prob"] = statistics.mean(log_probs)
        metrics["variance_log_prob"] = statistics.variance(log_probs, metrics["mean_log_prob"])
        metrics["perplexity"] = math.exp(-metrics["mean_log_prob"])
        metrics["num_predictions"] = len(log_probs)


class TextCompletionUsingDocumentsTask(TextCompletionTaskBase):
    """Represents an end-to-end text completion benchmark task with its data files."""

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.TextCompletionUsingDocuments

    def __init__(
        self,
        name: str,
        documents: Path,
        root_for_relative_paths: Path | None = None,
    ):
        """Initialize an end-to-end text completion task."""
        super().__init__(
            name, documents=documents, schema=None, root_for_relative_paths=root_for_relative_paths
        )  # kb schema is not used in this task


class TextCompletionUsingKBTask(TextCompletionTaskBase):
    """Represents a kb-augmented text completion benchmark task with its data files."""

    __kb: Path

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.TextCompletionUsingKB

    @property
    def kb(self) -> Path:
        """Return path to knowledge base."""
        return self.__kb

    def __init__(
        self,
        name: str,
        documents: Path,
        kb: Path,
        schema: Path | None = None,
        root_for_relative_paths: Path | None = None,
    ):
        """Initialize a kb-augmented text completion task."""
        super().__init__(name, documents=documents, schema=schema, root_for_relative_paths=root_for_relative_paths)
        self.__kb = resolve_path(kb, root_for_relative_paths)

    def read_kb(self) -> Iterable[Entity]:
        """
        Read entities from knowledge base.

        Returns:
            Iterable[Entity]: An iterable of `Entity` objects.
        """
        return EntityJsonlReader(self.__kb).read_items()

    def remove_sources_from_kb(self, kb: list[Entity], remove_sources_list: list[str]) -> list[Entity]:
        """Return a new KB after removing property values corresponding to specified sources from all entities.

        Args:
            kb: A list of entities
            remove_sources_list: A list of sources IDs to remove.

        Returns:
            A new list of entities with filtered values.
        """
        updated_kb = [entity.remove_sources(remove_sources_list) for entity in kb]
        updated_kb = [entity for entity in updated_kb if entity is not None]  # remove any empty entities
        return updated_kb
