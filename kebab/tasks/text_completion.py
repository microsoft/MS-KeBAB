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

import numpy as np
from scipy.special import logsumexp

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

    def __generate_partial_queries_with_thresholds(
        self,
        base_predictions: Path,
        verbose: bool = False,
        words_after_mask: int = 0,
        include_to_eval_flags_only: bool = False,
    ) -> Iterable[dict[str, Any]]:
        """
        Generate partial queries with thresholds based on the model's predictions.

        Args:
            base_predictions: The path to the base model's predictions, which will be used to
            calculate thresholds for evaluation.
            verbose: Defaults to False, only includes queries marked for evaluation. If True,
            includes all queries, including skipped words with empty "text_with_mask" for debugging.
            words_after_mask: Defaults to 0. Number of non-whitespace words/tokens to include
            after the mask position.
            include_to_eval_flags_only: Defaults to False, includes both `to_eval` flags and
            threshold values; if True, only includes `to_eval` flags without the threshold values.

        Returns:
            Iterable[dict[str, Any]]: An iterable of queries, where each query is a dict containing:
                - Fields from `generate_partial_queries` output.
                - "to_eval": A boolean flag indicating whether this query should be evaluated based
                on the calculated threshold.
                - "t": The calculated threshold for evaluation (included only if
                `include_to_eval_flags_only` is False).
        """
        base_predictions_iter = iter(ItemJsonlReader[dict[str, str | float]](base_predictions).read_items())
        queries = self.generate_partial_queries(verbose=verbose, words_after_mask=words_after_mask)

        for query in queries:
            query: dict[str, Any]
            if query["text_with_mask"] == "":
                query["to_eval"] = False
            else:
                try:
                    prediction = next(base_predictions_iter)
                except StopIteration as e:
                    raise ValueError(
                        "Number of base predictions is less than the number of queries being evaluated."
                    ) from e
                t = TextCompletionTaskBase.__calculate_t(prediction, query)
                query["to_eval"] = t < 0  # Evaluate only when the threshold is less than 0.
                if not include_to_eval_flags_only:
                    query["t"] = t
            # If `verbose` is True, include all queries, otherwise include only those with `to_eval` True.
            if verbose or query["to_eval"]:
                yield query

    def generate_partial_queries_to_eval(
        self,
        base_predictions: Path,
        verbose: bool = False,
        words_after_mask: int = 0,
    ) -> Iterable[dict[str, Any]]:
        """
        Generate partial queries to complete for evaluation, including only `to_eval` flags.

        Args:
            base_predictions: The path to the base model's predictions, which will be used to
            calculate thresholds for evaluation.
            verbose: Defaults to False, only includes queries marked for evaluation; if True,
            includes all queries, including skipped words with empty "text_with_mask" for debugging.
            words_after_mask: Defaults to 0. Number of non-whitespace words/tokens to include
            after the mask position.

        Returns:
            Iterable[dict[str, Any]]: An iterable of queries, where each query is a dict containing:
                - "text_with_mask": The text with a chunk replaced by "<mask>".
                - "target_content": The content of the chunk that was replaced by "<mask>".
                - "document_id": The ID of the document being processed.
                - "to_eval": A boolean flag indicating whether this query should be evaluated based
                on the calculated threshold.
        """
        yield from self.__generate_partial_queries_with_thresholds(
            base_predictions=base_predictions,
            verbose=verbose,
            words_after_mask=words_after_mask,
            include_to_eval_flags_only=True,
        )

    def write_items(
        self, path: Path, items: Iterable[dict[str, Any]], verbose: bool = False, strict: bool = True
    ) -> None:
        """
        Write prediction items to the specified path.

        Args:
            path: The file path where the prediction items should be written.
            items: An iterable of dictionaries, where if verbose is False, each dictionary must
            contain:
                - "predicted_content": The predicted content.
                - "target_content_logprob": The log probability of the target content.
                - "predicted_content_top_logprobs": The top log probabilities for the predicted content.
            verbose: Defaults to False, writes one JSONL line per item with only the required
            fields. If True, writes all items as a JSON array with all fields.
            strict: Only applies when `verbose` is False. Defaults to True, raises an error when
            required fields are missing; if False, skips items with missing required fields.
        """
        required_fields = ("predicted_content", "target_content_logprob", "predicted_content_top_logprobs")

        if verbose:
            output_items = list(items)
            with open(path, "w", encoding="utf-8", newline="\n") as file:
                json.dump(output_items, file, ensure_ascii=False, indent=2, default=str)
                file.write("\n")
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as file:
                for item in items:
                    output_item = {key: item[key] for key in required_fields if key in item}
                    # Ensure all required fields are present when not in verbose mode.
                    if len(output_item) < len(required_fields):
                        if strict:
                            missing_fields = set(required_fields) - output_item.keys()
                            raise ValueError(f"Missing required fields: {missing_fields}, in item: {item}")
                        else:
                            continue

                    json_line = json.dumps(
                        output_item,
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

        predicted_vals = ItemJsonlReader[dict[str, str | float]](predictions).read_items()
        queries = self.generate_partial_queries()
        predictions_and_queries = zip(predicted_vals, queries, strict=True)

        log_probs = []
        log_probs_by_doc = defaultdict(list)
        for prediction, query in predictions_and_queries:
            log_prob = float(prediction["target_content_logprob"])
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

    def fair_keyword_evaluate(
        self,
        to_eval_predictions: Path,
        base_predictions: Path,
        metrics_output_path: Path | None = None,
        logger: Logger | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the text completion task."""
        if logger:
            logger.info("Starting evaluation for the text completion task.")

        predicted_vals = ItemJsonlReader[dict[str, Any]](to_eval_predictions).read_items()
        predicted_vals_iter = iter(predicted_vals)
        queries = self.__generate_partial_queries_with_thresholds(base_predictions=base_predictions)

        scores = []
        for query in queries:
            query: dict[str, Any]
            if query["text_with_mask"] != "" and query["to_eval"]:
                prediction = next(predicted_vals_iter)
                t = query["t"]
                top_logprobs = prediction["predicted_content_top_logprobs"]
                top_logprobs.sort(key=lambda x: x[1], reverse=True)

                # Calculate cumulative probabilities and find largest n such that:
                # log(p_n/c_n) - c_{n-1}/p_n * log(1 + p_n/c_{n-1}) - t > 0
                k = 0
                cumulative_probs = []

                for i, (_, logprob) in enumerate(top_logprobs):
                    prob = np.exp(logprob)
                    if i == 0:
                        cumulative_probs.append(prob)
                        c_prev = 0
                    else:
                        cumulative_probs.append(cumulative_probs[-1] + prob)
                        c_prev = cumulative_probs[i - 1]

                    c_current = cumulative_probs[i]

                    # Calculate the condition: log(p_n/c_n) - c_{n-1}/p_n * log(1 + p_n/c_{n-1}) - t > 0
                    term1 = np.log(prob / c_current)

                    # For i=0, c_prev = 0, and the limit of term2 is 0.
                    term2 = (c_prev / prob) * np.log(1 + prob / c_prev) if c_prev > 0 else 0

                    condition = term1 - term2 - t

                    if condition > 0:
                        k = i + 1

                # If no k found, keep at least one element.
                if k == 0:
                    k = 1

                # Truncate and normalize: (p_1/c_k, ..., p_k/c_k)
                c_k = cumulative_probs[k - 1]
                truncated_logprobs = []
                for i in range(k):
                    token, logprob = top_logprobs[i]
                    normalized_prob = np.exp(logprob) / c_k
                    normalized_logprob = np.log(normalized_prob)
                    truncated_logprobs.append(
                        {
                            "token": token,
                            "logprob": normalized_logprob,
                        }
                    )

                # Calculate score using the truncated distribution.
                target_content = query["target_content"]
                truncated_result = TextCompletionTaskBase.prepare_results_from_top_logprobs(
                    target_content=target_content,
                    top_logprobs=[truncated_logprobs],
                )

                target_logprob_truncated = truncated_result["target_content_logprob"]
                score = max(0, target_logprob_truncated - t)
                scores.append(score)

        metrics = {}
        metrics["total_score"] = sum(scores)
        metrics["mean_score"] = statistics.mean(scores)
        metrics["variance_score"] = statistics.variance(scores, metrics["mean_score"]) if len(scores) > 1 else 0.0
        metrics["num_eval_words"] = len(scores)
        metrics["num_positive_scores"] = sum(1 for score in scores if score > 0)

        if logger:
            logger.info("Evaluation metrics calculated successfully.")
            logger.info(f"Metrics: {metrics}")
        if metrics_output_path:
            save_dict_to_json(metrics, metrics_output_path)

        return metrics

    @staticmethod
    def prepare_results_from_top_logprobs(
        target_content: str,
        top_logprobs: list[list[dict[str, Any]]] | None,
        additional_info: dict[str, Any] | None = None,
        allow_target_content_as_prefix: bool = False,
    ) -> dict[str, Any]:
        """
        Processes the top log probabilities and prepares the results.

        Args:
            target_content: The expected content to fill in the mask.
            top_logprobs: A list of lists containing the top log probabilities for each token position.
            additional_info: Additional information to include in the results.
            allow_target_content_as_prefix: Whether to allow the target content to be a prefix of the predicted token.

        Returns:
            dict[str, Any]: A dictionary including the following:
                - "predicted_content" (str): The predicted content for the masked position.
                - "target_content_logprob" (float): The log probability of the target content.
                - "predicted_content_top_logprobs" (list[list[dict[str, Any]]]): Each outer list
                  contains the top log probabilities for the corresponding token position.
        """
        if top_logprobs is None or len(top_logprobs) == 0 or len(top_logprobs[0]) == 0:
            results = {
                "predicted_content": "<Not Finished Correctly>",
                "target_content_logprob": float("-inf"),
                "predicted_content_top_logprobs": [[{"token": target_content, "logprob": float("-inf")}]],
            }
            if additional_info:
                results |= additional_info
            return results

        target_content_logprob = float("-inf")
        target_content_lower = target_content.strip().lower()
        # When the target content contains multiple tokens, LLM can return the target content
        # through multiple tokenization paths. For example, LLM can return "Beijing" as two tokens
        # ["Be", "ijing"] or just a single token ["Beijing"]; `self.tokenizer.encode` only returns
        # the former. We will approximately calculate the combined log prob by summing the
        # probabilities of all prefixes of the target content in `top_tokens`.
        # TODO (allenwang-ms): account for all possible tokenization paths; calculate the accurate
        # log prob of a sequence of tokens by multiple forward passes.
        prefix_logprobs = []
        for logprob in top_logprobs[0]:
            token = logprob["token"].strip().lower()
            if (
                # Check if the predicted token matches the target content or a prefix of it.
                target_content_lower.startswith(token)
                # When `allow_target_content_as_prefix` is True, also allow the target content to be a prefix of the predicted token.
                or (allow_target_content_as_prefix and token.startswith(target_content_lower))
            ) and token:  # Ensure non-empty token.
                prefix_logprobs.append(logprob["logprob"])
        if prefix_logprobs:
            # Convert to numpy array for logsumexp operations.
            prefix_array = np.array(prefix_logprobs, dtype=float)
            target_content_logprob = logsumexp(prefix_array)

        results = {
            "predicted_content": top_logprobs[0][0]["token"],
            "target_content_logprob": target_content_logprob,
            "predicted_content_top_logprobs": [[logprob["token"], logprob["logprob"]] for logprob in top_logprobs[0]],
        }
        if additional_info:
            results |= additional_info

        return results

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

    @staticmethod
    def __calculate_t(prediction: dict[str, Any], query: dict[str, Any], a: float = -4.0, b: float = 1.0) -> float:
        """
        Calculate threshold ensuring baseline has zero score.
        t = min(0, max(a, base_logprob + b, c))
        where c is calculated from the position of target content in the distribution.
        """
        base_logprob = prediction["target_content_logprob"]

        # Get the target content and distribution.
        target_content = query["target_content"].strip().lower()
        top_logprobs = prediction["predicted_content_top_logprobs"]

        # Sort by logprob descending.
        top_logprobs.sort(key=lambda x: x[1], reverse=True)

        # Find position of target content (match first token that equals or is prefix).
        target_position = -1
        for i, (token, _) in enumerate(top_logprobs):
            token_lower = token.strip().lower()
            # Check if token is a prefix of target content.
            if target_content.startswith(token_lower):
                target_position = i
                break

        # Calculate cumulative probabilities.
        cumulative_probs = []
        for i, (_, logprob) in enumerate(top_logprobs):
            prob = np.exp(logprob)
            if i == 0:
                cumulative_probs.append(prob)
            else:
                cumulative_probs.append(cumulative_probs[-1] + prob)

        # Calculate c based on whether target was found.
        if target_position != -1:
            # Target found at position n.
            n = target_position
            p_n = np.exp(top_logprobs[n][1])
            c_n = cumulative_probs[n]
            c_prev = cumulative_probs[n - 1] if n > 0 else 0

            # c = log(p_n/c_n) - c_{n-1}/p_n * log(1 + p_n/c_{n-1})
            term1 = np.log(p_n / c_n)
            term2 = (c_prev / p_n) * np.log(1 + p_n / c_prev) if c_prev > 0 else 0

            c = term1 - term2
        else:
            # Target not found, use k = length of distribution.
            k = len(top_logprobs) - 1  # Last position (0-indexed)
            p_k = np.exp(top_logprobs[k][1])
            c_k = cumulative_probs[k]

            # c = log(p_k/(c_k+p_k)) - c_k/p_k * log(1 + p_k/c_k)
            term1 = np.log(p_k / (c_k + p_k))
            term2 = (c_k / p_k) * np.log(1 + p_k / c_k)

            c = term1 - term2

        # Final threshold: t = min(0, max(a, baseline_logprob + b, c))
        t = min(0, max(a, base_logprob + b, c))
        return t


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
