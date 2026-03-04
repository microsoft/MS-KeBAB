# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, ClassVar, cast, override

import numpy as np
import torch
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.generation.utils import GenerateOutput

from kebab.tasks.text_completion import TextCompletionTaskBase


class RateLimiter:
    """
    A token bucket rate limiter for controlling API request rates.

    Uses a ``threading.Condition`` so waiting threads sleep until a token is
    expected to be available, eliminating spinlock CPU overhead and lock
    contention.

    The burst capacity is capped at 1 second's worth of tokens so the limiter
    enforces a smooth rate from the very first request rather than allowing a
    large initial burst that exhausts the minute's allowance instantly.

    Attributes:
        rate: Maximum number of requests allowed per minute.
        tokens: Current number of available tokens.
        max_tokens: Maximum burst capacity (1 second's worth of tokens, minimum 1).
        last_update: Monotonic timestamp of the last token refill.
    """

    def __init__(self, requests_per_minute: int):
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum number of requests allowed per minute.
        """
        self._rate_per_sec: float = requests_per_minute / 60.0
        # Burst cap: at most 1 second's worth of tokens (minimum 1).
        self._max_tokens: float = max(1.0, self._rate_per_sec)
        self._tokens: float = self._max_tokens
        self._last_update: float = time.monotonic()
        self._condition = threading.Condition(threading.Lock())

    def _refill(self) -> None:
        """Refill tokens proportional to elapsed time. Must be called under the lock."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate_per_sec)
        self._last_update = now

    def acquire(self) -> None:
        """
        Block until a token is available, then consume it.

        Uses ``Condition.wait(timeout=...)`` so the calling thread sleeps
        efficiently rather than spinning.
        """
        with self._condition:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Sleep exactly as long as it takes for one token to appear.
                wait_secs = (1.0 - self._tokens) / self._rate_per_sec
                self._condition.wait(timeout=wait_secs)


class BaseRAGTextCompleter(ABC):
    """
    A base class for RAG text completers, which provides a common interface for different RAG text
    completion models. The derived classes should implement the `get_augmented_context` and
    `complete_single_partial_query` methods.
    """

    @abstractmethod
    def get_augmented_context(self, query: dict[str, Any]) -> str:
        """
        Retrieves augmented context for a given partial query, i.e., a RAG (Retrieval-Augmented
        Generation) function.

        Args:
            query: A dictionary containing the partial query, typically including fields like
            "text_with_mask" and "target_content".

        Returns:
            str: The augmented context as a string that provides relevant background information
            to assist with text completion.
        """
        raise NotImplementedError

    @abstractmethod
    def complete_single_partial_query(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str = "",
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Processes a single partial query and returns the predicted content and log probabilities.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            target_content: The expected content to fill in the mask.
            augmented_context: Additional context to help with the prediction.
            seed: seed for the random number generator.

        Returns:
            dict[str, Any]: A dictionary requiring the following:
                - "predicted_content" (str): The predicted content for the masked position.
                - "target_content_logprob" (float): The log probability of the target content.
        """
        raise NotImplementedError

    def complete_partial_queries(
        self,
        partial_queries: Iterable[dict[str, Any]],
        seed: int = 42,
        verbose: bool = False,
        max_workers: int | None = None,
        requests_per_minute: int = 750,
        batch_size: int = 1_000,
        query_timeout_secs: float = 1_800.0,
        max_retries: int = 3,
    ) -> Iterable[dict[str, Any]]:
        """
        Processes a collection of partial queries in sequential batches, with
        per-query timeouts and automatic retries for stuck requests.

        Each batch of ``batch_size`` queries is submitted to a thread pool.
        If any query does not complete within ``query_timeout_secs``, it is
        cancelled and retried in a subsequent pass (up to ``max_retries``
        total attempts).  The next batch is not started until the current
        batch is fully resolved, preventing thousands of in-flight requests
        from piling up.

        Args:
            partial_queries: An iterable of dictionaries, where each dictionary represents a partial
                query to complete.
            seed: seed for the random number generator.
            verbose: Defaults to False. If True, includes additional information such as the
                original partial query and augmented context in the results for debugging.
            max_workers: Maximum number of parallel workers. Defaults to
                ``requests_per_minute // 12``.
            requests_per_minute: Rate limit for API requests per minute. Default is 750.
            batch_size: Number of queries to process per batch before moving on. Default is 1000.
            query_timeout_secs: Maximum seconds to wait for a single query before considering it
                stuck and retrying. Default is 120.
            max_retries: Maximum number of attempts per query (including the first). Default is 3.

        Returns:
            Iterable[dict[str, Any]]: An iterable of dictionaries in the same order as the input.
        """
        from concurrent.futures import Future

        queries_list = list(partial_queries)
        total_queries = len(queries_list)

        if total_queries == 0:
            return []

        effective_workers = max_workers if max_workers is not None else max(1, requests_per_minute // 12)
        rate_limiter = RateLimiter(requests_per_minute)

        # Pre-allocate results; None means "not yet completed".
        all_results: list[dict[str, Any] | None] = [None] * total_queries

        def run_single_query(index: int, query: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
            """Execute one query, returning (index, result) or (index, None) on skip."""
            result: dict[str, Any] = {}
            if verbose:
                result = copy.deepcopy(query)

            if query["text_with_mask"] == "":
                if verbose:
                    return (index, result)
                return (index, None)

            rate_limiter.acquire()

            augmented_context = self.get_augmented_context(query)
            if verbose:
                result["augmented_context"] = augmented_context

            result_single_query = self.complete_single_partial_query(
                text_with_mask=query["text_with_mask"],
                target_content=query["target_content"],
                augmented_context=augmented_context,
                seed=seed,
            )

            result["predicted_content"] = result_single_query["predicted_content"]
            result["target_content_logprob"] = result_single_query["target_content_logprob"]
            if verbose:
                result |= result_single_query

            return (index, result)

        def process_batch(indices: list[int]) -> list[int]:
            """
            Submit queries for the given indices, wait with timeout, and return
            the list of indices that did NOT complete in time.
            """
            pending_futures: dict[Future[tuple[int, dict[str, Any] | None]], int] = {}
            timed_out_indices: list[int] = []

            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                for idx in indices:
                    future = executor.submit(run_single_query, idx, queries_list[idx])
                    pending_futures[future] = idx

                completed_count = 0
                for future in as_completed(pending_futures, timeout=query_timeout_secs):
                    idx = pending_futures.pop(future)
                    try:
                        result_idx, result_data = future.result(timeout=0)
                        if result_data is not None:
                            all_results[result_idx] = result_data
                        completed_count += 1
                        if completed_count % 100 == 0 or completed_count == len(indices):
                            print(f"  Batch progress: {completed_count}/{len(indices)} queries completed.")
                    except Exception as e:  # noqa: BLE001 - intentionally catching all worker exceptions for retry
                        print(f"  Query {idx} failed with error: {e}")
                        timed_out_indices.append(idx)

            # Any futures still in pending_futures after as_completed's timeout are stuck.
            for future, idx in pending_futures.items():
                future.cancel()
                timed_out_indices.append(idx)
                print(f"  Query {idx} timed out after {query_timeout_secs}s, will retry.")

            return timed_out_indices

        # --- Main loop: process in batches ---
        global_completed = 0
        for batch_start in range(0, total_queries, batch_size):
            batch_end = min(batch_start + batch_size, total_queries)
            batch_indices = list(range(batch_start, batch_end))
            print(
                f"\n=== Processing batch [{batch_start}\u2013{batch_end - 1}] "
                f"({len(batch_indices)} queries, total {total_queries}) ==="
            )

            pending = batch_indices
            for attempt in range(1, max_retries + 1):
                if not pending:
                    break
                if attempt > 1:
                    print(
                        f"  Retry attempt {attempt}/{max_retries} for {len(pending)} stuck queries "
                        f"in batch [{batch_start}\u2013{batch_end - 1}]."
                    )
                failed = process_batch(pending)
                pending = failed

            batch_completed = sum(1 for i in batch_indices if all_results[i] is not None)
            global_completed += batch_completed
            print(
                f"=== Batch [{batch_start}\u2013{batch_end - 1}] done: "
                f"{batch_completed}/{len(batch_indices)} succeeded "
                f"({global_completed}/{total_queries} overall) ==="
            )

            if pending:
                max_display = 20
                print(
                    f"  WARNING: {len(pending)} queries failed after {max_retries} attempts: "
                    f"{pending[:max_display]}{'...' if len(pending) > max_display else ''}"
                )

        # Yield results in original order
        for i in range(total_queries):
            result = all_results[i]
            if result is not None:
                yield result

    @staticmethod
    def prepare_results_from_top_logprobs(
        target_content: str,
        top_logprobs: list[list[dict[str, Any]]] | None,
        additional_info: dict[str, Any] | None = None,
        allow_be_prefix_of_target_content: bool = False,
        allow_target_content_as_prefix: bool = False,
    ) -> dict[str, Any]:
        """
        Processes the top log probabilities and prepares the results.

        Args:
            target_content: The expected content to fill in the mask.
            top_logprobs: A list of lists containing the top log probabilities for each token position.
            additional_info: Additional information to include in the results.
            allow_be_prefix_of_target_content: Whether to allow the predicted token to be a prefix of the target content.
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

        # Combine probabilities for duplicate tokens and sort by probability descending.
        for position_logprobs in top_logprobs:
            token_logprob_map: dict[str, float] = {}
            for entry in position_logprobs:
                token = entry["token"]
                lp = entry["logprob"]
                if token in token_logprob_map:
                    # Sum probabilities in log-space: log(exp(a) + exp(b))
                    token_logprob_map[token] = cast(float, logsumexp(np.array([token_logprob_map[token], lp])))
                else:
                    token_logprob_map[token] = lp
            # Normalize if the sum of probabilities exceeds 1.
            total_logprob = cast(float, logsumexp(np.array(list(token_logprob_map.values()))))
            if total_logprob > 0.0:  # sum of probs > 1
                for token in token_logprob_map:
                    token_logprob_map[token] -= total_logprob
            # Rebuild the list sorted by logprob descending.
            position_logprobs.clear()
            position_logprobs.extend(
                sorted(
                    [{"token": t, "logprob": lp} for t, lp in token_logprob_map.items()],
                    key=lambda x: x["logprob"],
                    reverse=True,
                )
            )

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
            if token == target_content_lower or (
                (
                    # Check if the predicted token matches the target content or a prefix of it.
                    (allow_be_prefix_of_target_content and target_content_lower.startswith(token))
                    # When `allow_target_content_as_prefix` is True, also allow the target content to be a prefix of the predicted token.
                    or (allow_target_content_as_prefix and token.startswith(target_content_lower))
                )
                and token
            ):
                prefix_logprobs.append(logprob["logprob"])
        if prefix_logprobs:
            # Convert to numpy array for logsumexp operations.
            prefix_array = np.array(prefix_logprobs, dtype=float)
            target_content_logprob = logsumexp(prefix_array)

        results = {
            "predicted_content": top_logprobs[0][0]["token"],
            "target_content_logprob": target_content_logprob,
            "predicted_content_top_logprobs": top_logprobs,
        }
        if additional_info:
            results |= additional_info

        return results

    @staticmethod
    def get_target_content_logprob_with_fallback(result: dict[str, Any], top_k: int = 20) -> float:
        """
        Gets the target content log probability with fallback.

        Args:
            result: A dictionary containing the result of text completion.
            top_k: The number of top predictions to consider.
        """
        if result["target_content_logprob"] != float("-inf"):
            return result["target_content_logprob"]
        # Fallback to the smallest log probability among the top k predicted tokens when the
        # target content log probability is -inf.
        target_content_logprob = result["predicted_content_top_logprobs"][0][-1]["logprob"]
        # Fallback to a small value to avoid -inf log probability or when the prob is larger than 1 / top_k
        # which usually means a shorter list of predictions being returned.
        if target_content_logprob > math.log(1 / top_k) or target_content_logprob == float("-inf"):
            target_content_logprob = math.log(1 / 100_000)
        return target_content_logprob

    @staticmethod
    def prepare_top_logprobs_from_json_response(
        json_response: str,
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """
        Prepares the top log probabilities from the JSON response.

        Args:
            json_response: The JSON response string from the model.

        Returns:
            tuple[list[dict[str, Any]] | None, str | None]: A tuple containing the list of top log
            probabilities and an error message if any.
        """
        match = re.search(r"(\[.*?\])", json_response, re.DOTALL)
        error = None
        if not match:
            error = "Could not find JSON list in model response."
            return None, error
        json_str = match.group(1)
        top_logprobs = None
        try:
            top_probs = json.loads(json_str)
        except Exception as e:  # noqa: BLE001 - intentionally catching all exceptions to log JSON parsing failures
            error = f"Error parsing JSON: {e}"
            return None, error
        try:
            # Clean up top_probs: combine probabilities for duplicate tokens and normalize.
            word_prob_map = {}
            for item in top_probs:
                word = item.get("word", "")
                prob = item.get("prob", 0.0)
                if word in word_prob_map:
                    word_prob_map[word] += prob
                else:
                    word_prob_map[word] = prob
            # Normalize probabilities so they sum to 1.
            total_prob = sum(word_prob_map.values())
            if total_prob > 0:
                for word in word_prob_map:
                    word_prob_map[word] /= total_prob
            # Convert back to list format: use "token" as the key so that the format is consistent with the result from logits.
            top_probs = [{"token": token, "prob": prob} for token, prob in word_prob_map.items() if prob > 0.0]

            # Sort top_probs by "prob" descending.
            top_probs_sorted = sorted(top_probs, key=lambda x: x.get("prob", 1 / 100_000), reverse=True)
            # Convert to top_logprobs format (replace 'prob' with 'logprob').
            top_logprobs = [{"token": item["token"], "logprob": math.log(item["prob"])} for item in top_probs_sorted]
        except Exception as e:  # noqa: BLE001 - intentionally catching all exceptions to log JSON parsing failures
            error = f"Error processing top_probs: {e}"
            return None, error
        if top_logprobs is None:
            error = "Failed to extract top_logprobs"
            return None, error
        return top_logprobs, None

    # Thresholds for categorizing log probabilities, used for color-coding text in HTML visualization.
    LOGPROB_THRESHOLDS: ClassVar[list[float]] = [-4, -8, -12]

    @staticmethod
    def get_font_color(logprob: float) -> str:
        """Computes the font color based on log probability."""
        if logprob > BaseRAGTextCompleter.LOGPROB_THRESHOLDS[0]:
            return "#000000"  # Black for high confidence
        if logprob > BaseRAGTextCompleter.LOGPROB_THRESHOLDS[1]:
            return "#800000"  # Dark red for medium confidence
        return "#FF0000"  # Bright red for low confidence

    @staticmethod
    def generate_annotated_doc_html(
        text_completion_results: Iterable[dict[str, Any]],
        output_path: str,
    ) -> None:
        """Creates an HTML page that highlights words using font color based on log probability."""
        html_content = f"""
<html>
    <body>
        <p>LLM Prediction Confidence Visualization: Words are color-coded based on their log probabilities:</p>
        <ul>
            <li><span style="color: black;">High confidence (black): </span>logprob > {BaseRAGTextCompleter.LOGPROB_THRESHOLDS[0]}</li>
            <li><span style="color: #800000;">Medium confidence (dark red): </span>{BaseRAGTextCompleter.LOGPROB_THRESHOLDS[1]} < logprob <= {BaseRAGTextCompleter.LOGPROB_THRESHOLDS[0]}</li>
            <li><span style="color: red;">Low confidence (red): </span>{BaseRAGTextCompleter.LOGPROB_THRESHOLDS[2]} < logprob <= {BaseRAGTextCompleter.LOGPROB_THRESHOLDS[1]}</li>
            <li><span style="color: red; font-weight: bold;">Very low confidence (bold red): </span>logprob <= {BaseRAGTextCompleter.LOGPROB_THRESHOLDS[2]}</li>
        </ul>
        <hr>
        <p>
        """
        for result in text_completion_results:
            if result["text_with_mask"] == "":
                html_content += f'<span class="word">{result["target_content"]}</span>'
                continue
            target_word_logprob = result["target_content_logprob"]
            tooltip = ""
            if target_word_logprob == float("-inf"):
                tooltip = "Incorrect prediction, "
                if "predicted_content_top_logprobs" in result:
                    target_word_logprob = result["predicted_content_top_logprobs"][0][-1]["logprob"]
            color = BaseRAGTextCompleter.get_font_color(target_word_logprob)
            font_weight = "bold" if target_word_logprob <= BaseRAGTextCompleter.LOGPROB_THRESHOLDS[2] else "normal"
            tooltip += f"LogProb: {target_word_logprob:.3f}"
            if "predicted_content_top_logprobs" in result:
                top_logprobs_html = str(result["predicted_content_top_logprobs"]).replace('"', "&quot;")
                tooltip += (
                    f", top {len(result['predicted_content_top_logprobs'][0])} token predictions: {top_logprobs_html}"
                )
            html_content += f'<span class="word" style="color: {color}; font-weight: {font_weight};" title="{tooltip}">{result["target_content"]}</span>'
        html_content += """
        </p>
    </body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML file generated: {os.path.abspath(output_path)}")


class PredictionMethod(Enum):
    """Enumeration of different prediction methods."""

    LOGPROBS_FROM_LOGITS = "logprobs_from_logits"
    LOGPROBS_FROM_TEXT_RESPONSE = "logprobs_from_text_response"
    WORD_FROM_TEXT_RESPONSE = "word_from_text_response"

    def build_text_completion_prompt(self, text_with_mask: str, augmented_context: str, top_k: int) -> str:
        """
        Builds text completion prompt based on the prediction method.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            augmented_context: Additional context to help with the prediction.
            top_k: The number of top predictions to consider.
        """
        if self == PredictionMethod.LOGPROBS_FROM_LOGITS:
            return PredictionMethod.__build_logprobs_from_logits_prompt(
                text_with_mask=text_with_mask, augmented_context=augmented_context
            )
        if self == PredictionMethod.LOGPROBS_FROM_TEXT_RESPONSE:
            return self.__build_logprobs_from_text_response_prompt(
                text_with_mask=text_with_mask,
                augmented_context=augmented_context,
                top_k=top_k,
            )
        if self == PredictionMethod.WORD_FROM_TEXT_RESPONSE:
            return self.__build_word_from_text_response_prompt(
                text_with_mask=text_with_mask,
                augmented_context=augmented_context,
            )
        raise ValueError(f"Unknown prediction method: {self}")

    @staticmethod
    def __build_base_text_completion_prompt(text_with_mask: str, augmented_context: str) -> str:
        """
        Builds the base text completion prompt.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            augmented_context: Additional context to help with the prediction.
        """
        base_text_completion_prompt = f"""
You are a professional language model trained to assist with text completion tasks. Your goal is to accurately fill in missing parts of text using your understanding of language and context. You MUST always output a prediction. Never apologize, never refuse, never explain. Even if context is empty or insufficient, you MUST still output your best guess.

Below is the optional context retrieved to help you make a better prediction. It may be empty - that is fine, still make your best prediction. It is marked between <Begin Context> and <End Context>:
<Begin Context>
{augmented_context}
<End Context>

Now, complete the following text by predicting the missing alphanumeric word, represented by "{TextCompletionTaskBase.MASK}". The "{TextCompletionTaskBase.MASK}" is immediately followed by a non-alphanumeric character (e.g., space, -, ., ,, etc.) or the end of the text. Therefore, predicting a partial word that ends right before that delimiter is valid (e.g., "United" for "United Kingdom" or "Anglo" for "Anglo-Saxon"). The text is marked between <Begin Text> and <End Text>:
<Begin Text>
{text_with_mask}...<the rest of the text is not visible to you>
<End Text>
"""
        return base_text_completion_prompt

    @staticmethod
    def __build_logprobs_from_logits_prompt(text_with_mask: str, augmented_context: str) -> str:
        """
        Builds a prompt that asks the model to output the single best word prediction.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            augmented_context: Additional context to help with the prediction.
        """
        text_completion_prompt = f"""{
            PredictionMethod.__build_base_text_completion_prompt(
                text_with_mask=text_with_mask, augmented_context=augmented_context
            )
        }
What is the **single alphanumeric word** that best completes the masked position "{
            TextCompletionTaskBase.MASK
        }"? Please respond strictly with **only** the word. No explanation, no apology, no punctuation - just the word.

Answer: """
        return text_completion_prompt

    @staticmethod
    def __build_logprobs_from_text_response_prompt(
        text_with_mask: str,
        augmented_context: str,
        top_k: int,
    ) -> str:
        """
        Builds a prompt that asks the model to output top word predictions and their probs in JSON format.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            augmented_context: Additional context to help with the prediction.
            top_k: The number of top predictions to consider.
        """
        text_completion_prompt = f"""{
            PredictionMethod.__build_base_text_completion_prompt(
                text_with_mask=text_with_mask, augmented_context=augmented_context
            )
        }
Return exactly the top {top_k} single alphanumeric word predictions for "{
            TextCompletionTaskBase.MASK
        }" in **valid JSON format**.
Each element must be an object with:
  - "word": a string representing the predicted word.
  - "prob": a float representing its probability.
Your response must be a JSON array, like this:
[{{"word":"pred1","prob": 0.432}},{{"word":"pred2","prob": 0.312}},...]

Answer: """
        return text_completion_prompt

    @staticmethod
    def __build_word_from_text_response_prompt(
        text_with_mask: str,
        augmented_context: str,
    ) -> str:
        """
        Builds a prompt that asks the model to sample a single word from its probability distribution.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            augmented_context: Additional context to help with the prediction.
        """
        # TODO (allenwang-ms): The following prompt needs to be refined/tested to ensure the model
        # behaves as expected.
        text_completion_prompt = f"""{
            PredictionMethod.__build_base_text_completion_prompt(
                text_with_mask=text_with_mask, augmented_context=augmented_context
            )
        }
Sample the **single alphanumeric word** that completes the masked position "{
            TextCompletionTaskBase.MASK
        }" for the given text by drawing once from your internal probability distribution.
Rules:
- Do not choose the most likely word deterministically.
- Output exactly one **single alphanumeric word**.
- Do not normalize or re-rank; sample once from the native distribution.

Answer: """
        return text_completion_prompt


class BaseLocalLlmRAGTextCompleter(BaseRAGTextCompleter):
    """A base implementation of a RAG text completer using a local model."""

    model_id: str
    tokenizer: Any
    device: torch.device

    def __init__(
        self,
        model_id: str,
        gpu_id: int | None = None,
        prediction_method: PredictionMethod = PredictionMethod.LOGPROBS_FROM_LOGITS,
    ) -> None:
        """
        Initializes the text completer.

        Args:
            model_id: The model Id to use for text completion.
            gpu_id: The GPU Id to use for the model, if any.
            prediction_method: The method to use for predicting log probabilities.
        """
        super().__init__()
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.device = torch.device(
            "cuda" + ("" if gpu_id is None else f":{gpu_id}") if torch.cuda.is_available() else "cpu"
        )
        self.prediction_method = prediction_method

    def prepare_results_from_predicted_token_logits(
        self,
        target_content: str,
        predicted_token_logits: torch.Tensor,
        top_k: int,
        additional_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Processes the predicted token logits and prepares the results.

        Args:
            target_content: The expected content to fill in the mask.
            predicted_token_logits: The logits for the predicted tokens.
            top_k: The number of top predictions to consider.
            additional_info: Optional dictionary containing additional information.
        """
        # Compute log probabilities over the vocabulary.
        log_probs = torch.nn.functional.log_softmax(predicted_token_logits, dim=-1)

        # Extract top k token indices and their log probabilities.
        top_logprobs, top_indices = torch.topk(log_probs, top_k)
        top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]

        top_logprobs = [
            [
                {"token": token, "logprob": logprob}
                for token, logprob in zip(top_tokens, top_logprobs.tolist(), strict=True)
            ]
        ]

        return BaseRAGTextCompleter.prepare_results_from_top_logprobs(
            target_content=target_content,
            top_logprobs=top_logprobs,
            additional_info=additional_info,
        )


class BasePhiRAGTextCompleter(BaseLocalLlmRAGTextCompleter):
    """
    A base implementation of a RAG text completer using the Phi model. This class does not implement
    the `get_augmented_context` method, which should be provided by concrete subclasses to define
    how augmented context is retrieved.
    """

    TOKENS_TO_EXCLUDE: ClassVar[list[str]] = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    """Tokens that are excluded from the model's predictions."""

    token_ids_to_exclude: list[Any]
    model: Any

    def __init__(self, model_id: str = "microsoft/phi-4", gpu_id: int | None = None) -> None:
        """
        Initializes the text completer.

        Args:
            model_id: The Phi model Id to use for text completion.
            gpu_id: The GPU Id to use for the model, if any.
        """
        super().__init__(model_id=model_id, gpu_id=gpu_id)
        self.token_ids_to_exclude = [
            self.tokenizer.convert_tokens_to_ids(token) for token in BasePhiRAGTextCompleter.TOKENS_TO_EXCLUDE
        ]

        print(f"Loading {self.model_id} model on device: {self.device}.")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device} if gpu_id is not None else "auto",
        )

        print(f"Loaded {self.model_id} model on device: {self.device}.")

        self.model.eval()

    @override
    def complete_single_partial_query(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str = "",
        seed: int = 42,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """
        Processes a single partial query and returns the predicted content and log probabilities.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            target_content: The expected content to fill in the mask.
            augmented_context: Additional context to help with the prediction.
            seed: seed for the random number generator.
            top_k: The number of top predictions to consider.

        Returns:
            dict[str, Any]: A dictionary containing the following:
                - "predicted_content" (str): The predicted content for the masked position.
                - "target_content_logprob" (float): The log probability of the target content.
                - "predicted_content_top_logprobs" (list[list[dict[str, Any]]]): Each outer list
                contains predicitons at the same token position, and each inner list contains the
                top k predictions for that token position, including the token and its log
                probability.
        """
        text_completion_prompt = self.prediction_method.build_text_completion_prompt(
            text_with_mask=text_with_mask,
            augmented_context=augmented_context,
            top_k=top_k,
        )

        # Encode the prompt.
        input_ids = self.tokenizer.encode(text_completion_prompt, return_tensors="pt").to(self.device)

        # Get model outputs (logits for each token).
        with torch.no_grad():
            outputs = self.model(input_ids)

        # Extract logits for the next token (last position).
        next_token_logits = outputs.logits[0, -1, :]

        # Exclude bad tokens from the predictions.
        for token_id in self.token_ids_to_exclude:
            next_token_logits[token_id] = float("-inf")

        return self.prepare_results_from_predicted_token_logits(
            target_content=target_content,
            predicted_token_logits=next_token_logits,
            top_k=top_k,
        )


class BaseQwenRAGTextCompleter(BaseLocalLlmRAGTextCompleter):
    """
    A base implementation of a RAG text completer using the Qwen model. This class does not implement
    the `get_augmented_context` method, which should be provided by concrete subclasses to define
    how augmented context is retrieved.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-14B",
        gpu_id: int | None = None,
        prediction_method: PredictionMethod = PredictionMethod.LOGPROBS_FROM_LOGITS,
    ) -> None:
        """
        Initializes the text completer.

        Args:
            model_id: The Qwen model Id to use for text completion.
            gpu_id: The GPU Id to use for the model, if any.
            prediction_method: The method to use for predicting log probabilities.
        """
        super().__init__(model_id=model_id, gpu_id=gpu_id, prediction_method=prediction_method)

        print(f"Loading {self.model_id} model on device: {self.device}.")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device} if gpu_id is not None else "auto",
        )

        print(f"Loaded {self.model_id} model on device: {self.device}.")

        self.model.eval()

    @override
    def complete_single_partial_query(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str = "",
        seed: int = 42,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """
        Processes a single partial query and returns the predicted content and log probabilities.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            target_content: The expected content to fill in the mask.
            augmented_context: Additional context to help with the prediction.
            seed: seed for the random number generator.
            top_k: The number of top predictions to consider.

        Returns:
            dict[str, Any]: A dictionary containing the following:
                - "predicted_content" (str): The predicted content for the masked position.
                - "target_content_logprob" (float): The log probability of the target content.
                - "predicted_content_top_logprobs" (list[list[dict[str, Any]]]): Each outer list
                contains predicitons at the same token position, and each inner list contains the
                top k predictions for that token position, including the token and its log
                probability.
        """
        if self.prediction_method == PredictionMethod.LOGPROBS_FROM_LOGITS:
            return self.__get_logprobs_from_logits(
                text_with_mask=text_with_mask,
                target_content=target_content,
                augmented_context=augmented_context,
                top_k=top_k,
            )
        if self.prediction_method == PredictionMethod.LOGPROBS_FROM_TEXT_RESPONSE:
            return self.__get_logprobs_from_text_response(
                text_with_mask=text_with_mask,
                target_content=target_content,
                augmented_context=augmented_context,
                seed=seed,
                top_k=top_k,
            )
        if self.prediction_method == PredictionMethod.WORD_FROM_TEXT_RESPONSE:
            return self.__get_word_from_text_response(
                text_with_mask=text_with_mask,
                target_content=target_content,
                augmented_context=augmented_context,
                seed=seed,
            )
        raise ValueError(f"Unknown prediction method: {self.prediction_method}")

    def __get_logprobs_from_logits(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str,
        top_k: int,
    ) -> dict[str, Any]:
        """
        Processes a single partial query and returns the predicted content and log probabilities.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            target_content: The expected content to fill in the mask.
            augmented_context: Additional context to help with the prediction.
            top_k: The number of top predictions to consider.

        Returns:
            dict[str, Any]: A dictionary containing the following:
                - "predicted_content" (str): The predicted content for the masked position.
                - "target_content_logprob" (float): The log probability of the target content.
                - "predicted_content_top_logprobs" (list[list[dict[str, Any]]]): Each outer list
                contains predicitons at the same token position, and each inner list contains the
                top k predictions for that token position, including the token and its log
                probability.
        """
        text_completion_prompt = "/no_think" + self.prediction_method.build_text_completion_prompt(
            text_with_mask=text_with_mask,
            augmented_context=augmented_context,
            top_k=top_k,
        )

        # Encode the prompt (returns both input_ids and attention_mask).
        model_inputs = self.tokenizer(text_completion_prompt, return_tensors="pt").to(self.device)
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask

        # Generate tokens to let the model output whitespace/thinking if needed.
        outputs = cast(
            GenerateOutput,
            self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10_000,  # Allow a few tokens to skip potential whitespace.
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            ),
        )

        # Decode the generated text.
        generated_ids = outputs.sequences[0][input_ids.shape[1] :]
        raw_response = self.tokenizer.decode(generated_ids)

        # Find start of answer (after last occurrence of "</think>" or "<|im_start|>").
        think_marker = "</think>"
        im_start_marker = "<|im_start|>"
        last_think_pos = raw_response.rfind(think_marker)
        last_im_start_pos = raw_response.rfind(im_start_marker)
        start_search_pos = 0
        if last_think_pos == -1 and last_im_start_pos == -1:
            start_search_pos = 0
        elif last_think_pos > last_im_start_pos:
            start_search_pos = last_think_pos + len(think_marker)
        else:
            start_search_pos = last_im_start_pos + len(im_start_marker)
        # Find first non-whitespace char index.
        target_char_index = -1
        for idx in range(start_search_pos, len(raw_response)):
            if not raw_response[idx].isspace():
                target_char_index = idx
                break

        next_token_logits = None
        if target_char_index != -1 and outputs.scores is not None:
            # Find which token corresponds to this character.
            for i in range(len(generated_ids)):
                # Decode up to current token.
                decoded_prefix = self.tokenizer.decode(generated_ids[: i + 1])
                if len(decoded_prefix) > target_char_index:
                    # This token covers the target character.
                    next_token_logits = outputs.scores[i][0]  # logits for the i-th generated token
                    break
        error = None
        if next_token_logits is None:
            error = "Could not locate target token logits, falling back to last token logits."
            if outputs.scores:
                next_token_logits = outputs.scores[-1][0]
            else:
                with torch.no_grad():
                    outputs = self.model(input_ids)
                next_token_logits = outputs.logits[0, -1, :]

        additional_info = {"raw_model_output": raw_response}
        additional_info |= {"error": error} if error else {}

        return self.prepare_results_from_predicted_token_logits(
            target_content=target_content,
            predicted_token_logits=next_token_logits,
            top_k=top_k,
            additional_info=additional_info,
        )

    def __get_logprobs_from_text_response(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str,
        seed: int,
        top_k: int,
    ) -> dict[str, Any]:
        """Prompts the model to output the top word predictions and their probs in JSON format, parses the response, and returns the result dict."""
        text_completion_prompt = self.prediction_method.build_text_completion_prompt(
            text_with_mask=text_with_mask,
            augmented_context=augmented_context,
            top_k=top_k,
        )

        messages = [
            {"role": "user", "content": text_completion_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        # Retry to extract and parse the JSON list.
        max_retries = 3
        top_logprobs = None
        raw_response = ""
        error = None
        for attempt in range(max_retries):
            # Generate response from the model (no output_scores).
            set_seed(seed + attempt)
            outputs = cast(
                GenerateOutput,
                self.model.generate(
                    **inputs,
                    max_new_tokens=10_000,
                    temperature=0.7,
                    return_dict_in_generate=True,
                ),
            )

            # Decode the generated text.
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1] :]
            raw_response = self.tokenizer.decode(generated_ids)

            # Only look into the content between "</think>" and "<|im_end|>".
            start_marker = "</think>"
            end_marker = "<|im_end|>"
            start_idx = raw_response.find(start_marker)
            end_idx = raw_response.find(end_marker, start_idx + len(start_marker))
            if start_idx != -1 and end_idx != -1:
                response = raw_response[start_idx + len(start_marker) : end_idx]
            else:
                response = raw_response

            top_logprobs, error = BaseRAGTextCompleter.prepare_top_logprobs_from_json_response(
                json_response=response,
            )
            if error is not None:
                print(f"Attempt {attempt + 1}: {error}")
                continue
            if top_logprobs is not None:
                break

        top_logprobs = [top_logprobs] if top_logprobs is not None else None
        additional_info = {"raw_model_output": raw_response}
        additional_info |= {"error": error} if error else {}

        return BaseRAGTextCompleter.prepare_results_from_top_logprobs(
            target_content=target_content,
            top_logprobs=top_logprobs,
            additional_info=additional_info,
        )

    def __get_word_from_text_response(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str,
        seed: int,
    ) -> dict[str, Any]:
        """Prompts the model to output a single word prediction, parses the response, and returns the result dict."""
        text_completion_prompt = self.prediction_method.build_text_completion_prompt(
            text_with_mask=text_with_mask,
            augmented_context=augmented_context,
            top_k=1,
        )

        messages = [
            {"role": "user", "content": text_completion_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        # Retry to extract and parse the response.
        max_retries = 3
        predicted_word = None
        raw_response = ""
        error = None
        for attempt in range(max_retries):
            # Generate response from the model (no output_scores).
            set_seed(seed + attempt)
            outputs = cast(
                GenerateOutput,
                self.model.generate(
                    **inputs,
                    max_new_tokens=10_000,
                    temperature=0.7,
                    return_dict_in_generate=True,
                ),
            )

            # Decode the generated text.
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1] :]
            raw_response = self.tokenizer.decode(generated_ids)

            # Only look into the content between "</think>" and "<|im_end|>".
            start_marker = "</think>"
            end_marker = "<|im_end|>"
            start_idx = raw_response.find(start_marker)
            content_start = start_idx + len(start_marker) if start_idx != -1 else 0
            end_idx = raw_response.find(end_marker, content_start)
            predicted_word = raw_response[content_start:end_idx] if end_idx != -1 else raw_response[content_start:]

            if not predicted_word:
                error = "Could not find a valid word in the response."
                print(f"Attempt {attempt + 1}: {error}")
                continue
            error = None
            break

        top_logprobs = [[{"token": predicted_word, "logprob": 0.0}]] if predicted_word else None
        additional_info = {"raw_model_output": raw_response}
        additional_info |= {"error": error} if error else {}

        return BaseRAGTextCompleter.prepare_results_from_top_logprobs(
            target_content=target_content,
            top_logprobs=top_logprobs,
            additional_info=additional_info,
        )


class BaseGptOssRAGTextCompleter(BaseLocalLlmRAGTextCompleter):
    """
    A base implementation of a RAG text completer using the GPT-OSS model. This class does not implement
    the `get_augmented_context` method, which should be provided by concrete subclasses to define
    how augmented context is retrieved.
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-oss-120b",
        gpu_id: int | None = None,
        prediction_method: PredictionMethod = PredictionMethod.LOGPROBS_FROM_LOGITS,
    ) -> None:
        """
        Initializes the text completer.

        Args:
            model_id: The GPT-OSS model Id to use for text completion.
            gpu_id: The GPU Id to use for the model, if any.
            prediction_method: The method to use for predicting log probabilities.
        """
        super().__init__(model_id=model_id, gpu_id=gpu_id, prediction_method=prediction_method)

        print(f"Loading {self.model_id} model on device: {self.device}.")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": self.device} if gpu_id is not None else "auto",
        )

        print(f"Loaded {self.model_id} model on device: {self.device}.")

        self.model.eval()

    @override
    def complete_single_partial_query(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str = "",
        seed: int = 42,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """
        Processes a single partial query and returns the predicted content and log probabilities.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            target_content: The expected content to fill in the mask.
            augmented_context: Additional context to help with the prediction.
            seed:: seed for the random number generator.
            top_k: The number of top predictions to consider.

        Returns:
            dict[str, Any]: A dictionary containing the following:
                - "predicted_content" (str): The predicted content for the masked position.
                - "target_content_logprob" (float): The log probability of the target content.
                - "predicted_content_top_logprobs" (list[list[dict[str, Any]]]): Each outer list
                contains predicitons at the same token position, and each inner list contains the
                top k predictions for that token position, including the token and its log
                probability.
        """
        if self.prediction_method == PredictionMethod.LOGPROBS_FROM_TEXT_RESPONSE:
            return self.__get_logprobs_from_text_response(
                text_with_mask=text_with_mask,
                target_content=target_content,
                augmented_context=augmented_context,
                top_k=top_k,
            )
        if self.prediction_method == PredictionMethod.LOGPROBS_FROM_LOGITS:
            return self.__get_logprobs_from_logits(
                text_with_mask=text_with_mask,
                target_content=target_content,
                augmented_context=augmented_context,
                top_k=top_k,
            )
        raise ValueError(f"Unknown prediction method: {self.prediction_method}")

    def __get_logprobs_from_text_response(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str,
        top_k: int,
    ) -> dict[str, Any]:
        """Prompts the model to output the top word predictions and their probs in JSON format, parses the response, and returns the result dict."""
        text_completion_prompt = self.prediction_method.build_text_completion_prompt(
            text_with_mask=text_with_mask,
            augmented_context=augmented_context,
            top_k=top_k,
        )

        messages = [
            {"role": "user", "content": text_completion_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            reasoning_effort="low",
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        # Retry to extract and parse the JSON list.
        max_retries = 3
        top_logprobs = None
        decoded_text = ""
        error = None
        for attempt in range(max_retries):
            # Generate response from the model (no output_scores).
            set_seed(42 + attempt)
            outputs = cast(
                GenerateOutput,
                self.model.generate(
                    **inputs,
                    max_new_tokens=10_000,
                    temperature=0.7,
                    # do_sample=False,  # Uncomment to disable sampling.
                    return_dict_in_generate=True,
                ),
            )

            # Decode the generated text.
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1] :]
            decoded_text = self.tokenizer.decode(generated_ids)

            # Only look into the content between "<|start|>assistant<|channel|>final<|message|>" and "<|return|>".
            start_marker = "<|start|>assistant<|channel|>final<|message|>"
            end_marker = "<|return|>"
            start_idx = decoded_text.find(start_marker)
            end_idx = decoded_text.find(end_marker, start_idx + len(start_marker))
            if start_idx != -1 and end_idx != -1:
                relevant_content = decoded_text[start_idx + len(start_marker) : end_idx]
            else:
                relevant_content = decoded_text

            top_logprobs, error = BaseRAGTextCompleter.prepare_top_logprobs_from_json_response(
                json_response=relevant_content,
            )
            if error is not None:
                print(f"Attempt {attempt + 1}: {error}")
                continue
            if top_logprobs is not None:
                break

        top_logprobs = [top_logprobs] if top_logprobs is not None else None
        additional_info = {"raw_model_output": decoded_text}
        additional_info |= {"error": error} if error else {}

        return BaseRAGTextCompleter.prepare_results_from_top_logprobs(
            target_content=target_content,
            top_logprobs=top_logprobs,
            additional_info=additional_info,
        )

    def __get_logprobs_from_logits(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str,
        top_k: int,
    ) -> dict[str, Any]:
        """Prompts the model and extracts log probabilities from the output logits."""
        text_completion_prompt = self.prediction_method.build_text_completion_prompt(
            text_with_mask=text_with_mask,
            augmented_context=augmented_context,
            top_k=top_k,
        )
        messages = [
            {"role": "user", "content": text_completion_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            reasoning_effort="low",
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        # Retry logic for finding the start marker.
        start_marker = "<|start|>assistant<|channel|>final<|message|>"
        start_marker_tokens = self.tokenizer.encode(start_marker, add_special_tokens=False)
        max_retries = 5
        offset = None
        outputs = None
        decoded_text = ""

        for retry_count in range(max_retries):
            outputs = cast(
                GenerateOutput,
                self.model.generate(
                    **inputs,
                    max_new_tokens=10_000,
                    do_sample=False,
                    output_scores=True,  # return logits
                    return_dict_in_generate=True,  # return a dict including scores
                ),
            )

            # Slice out the generated portion.
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1] :]  # generated tokens only

            # Decode full generated text (optional, for inspection).
            decoded_text = self.tokenizer.decode(generated_ids)
            # print(f"Generated text (attempt {retry_count + 1}): {decoded_text}")  # Uncomment for debugging.

            for i in range(len(generated_ids) - len(start_marker_tokens) + 1):
                if all(generated_ids[i + j] == start_marker_tokens[j] for j in range(len(start_marker_tokens))):
                    offset = i
                    break

            if offset is not None:
                print(f"Found start marker at offset {offset} on attempt {retry_count + 1}")
                break
            print(f"Start marker not found on attempt {retry_count + 1}, retrying...")

        if offset is None or outputs is None or outputs.scores is None:
            print(f"Could not find start marker '{start_marker}' after {max_retries} attempts")
            # Output results.
            return {
                "predicted_content": "<Not Finished Correctly>",
                "target_content_logprob": float("-inf"),
                "predicted_content_top_logprobs": [[{"token": target_content, "logprob": float("-inf")}]],
                "raw_model_output": decoded_text,
            }

        next_token_idx = offset + len(start_marker_tokens)

        # Extract logits for the next token.
        next_token_logits = outputs.scores[next_token_idx][0]

        return self.prepare_results_from_predicted_token_logits(
            target_content=target_content,
            predicted_token_logits=next_token_logits,
            top_k=top_k,
        )
