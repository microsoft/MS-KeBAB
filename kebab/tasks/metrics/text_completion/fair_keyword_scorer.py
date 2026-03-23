# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

import numpy as np
from kebab.tasks.metrics.text_completion.utils import get_target_content_prob_from_top_probs, process_top_probs


class FairKeywordScorer:
    """Fair keyword evaluator for text completion tasks."""

    def __init__(self, base_predictions: Path, a: float = -4.0, b: float = 1.0) -> None:
        """Initialize the evaluator."""
        self.base_predictions = base_predictions
        self.a = a
        self.b = b

    def calculate_t(
        self,
        predicted_content_top_probs: dict[str, float],
        target_content: str,
    ) -> float:
        """
        Calculate threshold t for a word based on the base predicted content distribution.

        Args:
            predicted_content_top_probs: The top probabilities for the predicted content.
            target_content: The groundtruth content.
        """
        return FairKeywordScorer.__calculate_t(
            predicted_content_top_probs=predicted_content_top_probs,
            target_content=target_content,
            a=self.a,
            b=self.b,
        )

    @staticmethod
    def calculate_score(
        predicted_content_top_probs: dict[str, float],
        target_content: str,
        t: float,
    ) -> float:
        """
        Calculate the score for a prediction based on the truncated predicted content distribution.

        Args:
            predicted_content_top_probs: The top probabilities for the predicted content.
            target_content: The groundtruth content.
            t: The threshold for evaluation.
        """
        top_probs = list(predicted_content_top_probs.items())
        top_probs.sort(key=lambda x: x[1], reverse=True)

        # Calculate cumulative probabilities and find largest n such that:
        # log(p_n/c_n) - c_{n-1}/p_n * log(1 + p_n/c_{n-1}) - t > 0
        k = 0
        cumulative_probs = []

        for i, (_, prob) in enumerate(top_probs):
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
            else:
                break

        # If no k found, keep at least one element.
        if k == 0:
            k = 1

        # Truncate and normalize: (p_1/c_k, ..., p_k/c_k)
        c_k = cumulative_probs[k - 1]
        truncated_top_probs = {}
        for i in range(k):
            word, prob = top_probs[i]
            normalized_prob = prob / c_k
            truncated_top_probs[word] = normalized_prob

        # Calculate score using the truncated distribution.
        truncated_target_content_prob = get_target_content_prob_from_top_probs(
            predicted_content_top_probs=truncated_top_probs,
            target_content=target_content,
        )
        score = max(0, np.log(truncated_target_content_prob) - t) if truncated_target_content_prob > 0 else 0

        return score

    @staticmethod
    def __calculate_t(predicted_content_top_probs: dict[str, float], target_content: str, a: float, b: float) -> float:
        """
        Calculate threshold ensuring baseline has zero score.
        t = min(0, max(a, base_logprob + b, c))
        where c is calculated from the position of target content in the distribution.
        """
        predicted_content_top_probs = process_top_probs(predicted_content_top_probs)

        with np.errstate(divide="ignore"):
            base_logprob = np.log(get_target_content_prob_from_top_probs(predicted_content_top_probs, target_content))

        # Get the target content and distribution.
        target_content = target_content.strip().lower()
        top_probs = list(predicted_content_top_probs.items())

        # Find position of target content (match first token that equals or is prefix).
        target_position = -1
        for i, (token, _) in enumerate(top_probs):
            token_lower = str(token).strip().lower()
            if target_content == token_lower:
                target_position = i
                break

        # Calculate cumulative probabilities.
        cumulative_probs = []
        for i, (_, prob) in enumerate(top_probs):
            if i == 0:
                cumulative_probs.append(prob)
            else:
                cumulative_probs.append(cumulative_probs[-1] + prob)

        # Calculate c based on whether target was found.
        if target_position != -1:
            # Target found at position n.
            n = target_position
            p_n = top_probs[n][1]
            c_n = cumulative_probs[n]
            c_prev = cumulative_probs[n - 1] if n > 0 else 0

            # c = log(p_n/c_n) - c_{n-1}/p_n * log(1 + p_n/c_{n-1})
            term1 = np.log(p_n / c_n)
            term2 = (c_prev / p_n) * np.log(1 + p_n / c_prev) if c_prev > 0 else 0

            c = term1 - term2
        else:
            # Target not found, use k = length of distribution.
            k = len(top_probs) - 1  # last position (0-indexed)
            p_k = top_probs[k][1]
            c_k = cumulative_probs[k]

            # c = log(p_k/(c_k+p_k)) - c_k/p_k * log(1 + p_k/c_k)
            term1 = np.log(p_k / (c_k + p_k))
            term2 = (c_k / p_k) * np.log(1 + p_k / c_k)

            c = term1 - term2

        # Final threshold: t = min(0, max(a, baseline_logprob + b, c))
        t = min(0, max(a, base_logprob + b, c))

        return t
