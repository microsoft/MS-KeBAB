# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import cast

import numpy as np
from scipy.special import logsumexp


class FairKeywordScorer:
    """Fair keyword evaluator for text completion tasks."""

    def __init__(self, a: float = -4.0, b: float = 1.0) -> None:
        """Initialize the evaluator with threshold parameters a and b."""
        self.a = a
        self.b = b

    def calculate_t(
        self,
        predicted_content_top_logprobs: list[list[str | float]],
        target_content: str,
    ) -> float:
        """TODO (allenwang-ms)."""
        return FairKeywordScorer.__calculate_t(
            predicted_content_top_logprobs=predicted_content_top_logprobs,
            target_content=target_content,
            a=self.a,
            b=self.b,
        )

    def calculate_score(
        self,
        predicted_content_top_logprobs: list[list[str | float]],
        target_content: str,
        t: float,
    ) -> float:
        """TODO (allenwang-ms)."""
        predicted_content_top_logprobs.sort(key=lambda x: x[1], reverse=True)

        # Calculate cumulative probabilities and find largest n such that:
        # log(p_n/c_n) - c_{n-1}/p_n * log(1 + p_n/c_{n-1}) - t > 0
        k = 0
        cumulative_probs = []

        for i, (_, logprob) in enumerate(predicted_content_top_logprobs):
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
        truncated_top_logprobs = []
        for i in range(k):
            token, logprob = predicted_content_top_logprobs[i]
            normalized_prob = np.exp(logprob) / c_k
            normalized_logprob = np.log(normalized_prob)
            truncated_top_logprobs.append((token, normalized_logprob))

        # Calculate score using the truncated distribution.
        truncated_target_content_logprob = FairKeywordScorer.get_target_content_logprob_from_top_logprobs(
            predicted_content_top_logprobs=truncated_top_logprobs,
            target_content=target_content,
        )
        score = max(0, truncated_target_content_logprob - t)

        return score

    @staticmethod
    def get_target_content_logprob_from_top_logprobs(
        predicted_content_top_logprobs: list[list[str | float]],
        target_content: str,
        allow_target_content_as_prefix: bool = False,
    ) -> float:
        """Get the log probability of the target content from the top log probabilities."""
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
        for token, logprob in predicted_content_top_logprobs:
            token_lower = str(token).strip().lower()
            if (
                # Check if the predicted token matches the target content or a prefix of it.
                target_content_lower.startswith(token_lower)
                # When `allow_target_content_as_prefix` is True, also allow the target content to be a prefix of the predicted token.
                or (allow_target_content_as_prefix and token_lower.startswith(target_content_lower))
            ) and token_lower:  # Ensure non-empty token.
                prefix_logprobs.append(float(logprob))
        if prefix_logprobs:
            # Convert to numpy array for logsumexp operations.
            prefix_array = np.array(prefix_logprobs, dtype=float)
            target_content_logprob = cast(float, logsumexp(prefix_array))

        return target_content_logprob

    @staticmethod
    def __calculate_t(
        predicted_content_top_logprobs: list[list[str | float]], target_content: str, a: float, b: float
    ) -> float:
        """
        Calculate threshold ensuring baseline has zero score.
        t = min(0, max(a, base_logprob + b, c))
        where c is calculated from the position of target content in the distribution.
        """
        base_logprob = FairKeywordScorer.get_target_content_logprob_from_top_logprobs(
            predicted_content_top_logprobs, target_content
        )

        # Get the target content and distribution.
        target_content = target_content.strip().lower()

        # Sort by logprob descending.
        predicted_content_top_logprobs.sort(key=lambda x: x[1], reverse=True)

        # Find position of target content (match first token that equals or is prefix).
        target_position = -1
        for i, (token, _) in enumerate(predicted_content_top_logprobs):
            token_lower = str(token).strip().lower()
            # Check if token is a prefix of target content.
            if target_content.startswith(token_lower):
                target_position = i
                break

        # Calculate cumulative probabilities.
        cumulative_probs = []
        for i, (_, logprob) in enumerate(predicted_content_top_logprobs):
            prob = np.exp(logprob)
            if i == 0:
                cumulative_probs.append(prob)
            else:
                cumulative_probs.append(cumulative_probs[-1] + prob)

        # Calculate c based on whether target was found.
        if target_position != -1:
            # Target found at position n.
            n = target_position
            p_n = np.exp(predicted_content_top_logprobs[n][1])
            c_n = cumulative_probs[n]
            c_prev = cumulative_probs[n - 1] if n > 0 else 0

            # c = log(p_n/c_n) - c_{n-1}/p_n * log(1 + p_n/c_{n-1})
            term1 = np.log(p_n / c_n)
            term2 = (c_prev / p_n) * np.log(1 + p_n / c_prev) if c_prev > 0 else 0

            c = term1 - term2
        else:
            # Target not found, use k = length of distribution.
            k = len(predicted_content_top_logprobs) - 1  # Last position (0-indexed)
            p_k = np.exp(predicted_content_top_logprobs[k][1])
            c_k = cumulative_probs[k]

            # c = log(p_k/(c_k+p_k)) - c_k/p_k * log(1 + p_k/c_k)
            term1 = np.log(p_k / (c_k + p_k))
            term2 = (c_k / p_k) * np.log(1 + p_k / c_k)

            c = term1 - term2

        # Final threshold: t = min(0, max(a, baseline_logprob + b, c))
        t = min(0, max(a, base_logprob + b, c))

        return t
