# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Any


def process_top_probs(
    top_probs: dict[str, float],
) -> dict[str, float]:
    """Combine duplicate keys by summing probabilities, sort descending, and normalize if total
    exceeds 1.

    Args:
        top_probs: A dict mapping words/tokens to their probabilities.

    Returns:
        A dict sorted by probability descending, normalized so the total does not exceed 1.
    """
    combined: dict[str, float] = {}
    for word, prob in top_probs.items():
        combined[word] = combined.get(word, 0.0) + prob
    combined = dict(sorted(combined.items(), key=lambda item: item[1], reverse=True))
    total_prob = sum(combined.values())
    if total_prob > 1:
        combined = {k: v / total_prob for k, v in combined.items()}
    return combined


def process_logprobs_to_probs(
    top_logprobs: list[dict[str, Any]],
) -> dict[str, float]:
    """Process a list of token log-probability dicts to a combined, sorted probability dict.
    Duplicate tokens are combined by summing their probabilities. The result is sorted by
    probability descending and normalized if the total exceeds 1.

    Args:
        top_logprobs: A list of dicts, each with ``"word"`` (str) and ``"logprob"`` (float) keys.

    Returns:
        A dict sorted by probability descending, normalized so the total does not exceed 1.
    """
    probs: dict[str, float] = {}
    for logprob in top_logprobs:
        word = logprob["word"]
        prob = math.exp(logprob["logprob"])
        probs[word] = probs.get(word, 0.0) + prob
    return process_top_probs(probs)


def get_target_content_prob_from_top_probs(
    predicted_content_top_probs: dict[str, float],
    target_content: str,
    allow_be_prefix_of_target_content: bool = False,
    allow_target_content_as_prefix: bool = False,
) -> float:
    """
    Get the probability of the target content from the top predicted probabilities. By default, only exact (case-insensitive) matches are counted. The two optional flags relax the matching.

    Args:
        predicted_content_top_probs: A list of dicts containing the top probabilities.
        target_content: The expected content to fill in the mask.
        allow_be_prefix_of_target_content: Whether to allow a predicted word/token to be a prefix of the target content.
        allow_target_content_as_prefix: Whether to allow the target content to be a prefix of the predicted word/token.

    Returns:
        float: The probability of the target content based on the top predicted probabilities.
    """
    target_content_prob = 0
    target_content_lower = target_content.strip().lower()

    prefix_probs = []
    for token, prob in predicted_content_top_probs.items():
        token_lower = str(token).strip().lower()
        if token_lower == target_content_lower or (
            (
                # Check if the predicted token matches the target content or a prefix of it.
                (allow_be_prefix_of_target_content and target_content_lower.startswith(token_lower))
                # When `allow_target_content_as_prefix` is True, also allow the target content to be a prefix of the predicted token.
                or (allow_target_content_as_prefix and token_lower.startswith(target_content_lower))
            )
            and token_lower
        ):
            prefix_probs.append(prob)
    if prefix_probs:
        target_content_prob = sum(prefix_probs)

    return target_content_prob


def get_target_content_prob_with_fallback(result: dict[str, Any], top_k: int = 20) -> float:
    """
    Gets the target content probability with fallback.

    Args:
        result: A dictionary containing the result of text completion.
        top_k: The number of top predictions to consider.
    """
    if result.get("target_content_prob", 0) != 0:
        return result["target_content_prob"]
    # Fallback to the smallest probability among the top k predicted tokens when the target
    # content probability is 0.
    target_content_prob = min(result["predicted_content_top_probs"].values())
    # Fallback to a small value to avoid -inf log probability or when the prob is larger than 1 / top_k
    # which usually means a shorter list of predictions being returned.
    if target_content_prob > 1 / top_k or target_content_prob == 0:
        target_content_prob = 1 / 100_000
    return target_content_prob


def calculate_brier_score_for_prediction(
    predicted_content_top_probs: dict[str, float],
    target_content: str,
    top_k: int,
) -> float:
    """Calculate the Brier score for a prediction over the top-k predicted tokens.

    The target's probability is the sum of the probabilities of all top-k tokens
    that match the target content (case-insensitive exact match). Matched tokens
    are treated as a single "target" bucket; remaining tokens contribute their
    squared probabilities.

    Args:
        target_content: The expected content to fill in the mask.
        predicted_content_top_probs: A dict mapping predicted tokens to their probabilities.
        top_k: The number of top predicted tokens to consider.

    Returns:
        The Brier score: ``(p_target - 1)^2 + sum_{i != target} p_i^2``. Returns
        ``2.0`` when the target is not found within the top-k predictions.
    """
    # Combine duplicates, sort descending, normalize, then slice to top_k.
    top_k_probs = dict(list(process_top_probs(predicted_content_top_probs).items())[:top_k])
    target_content_prob = get_target_content_prob_from_top_probs(top_k_probs, target_content)

    # Target not found in top-k -> return 2.
    if target_content_prob == 0:
        return 2.0

    # Brier score: matched tokens form the target bucket; the rest contribute p^2.
    target_lower = target_content.strip().lower()
    brier_score = (target_content_prob - 1) ** 2 + sum(
        p**2 for token, p in top_k_probs.items() if token.strip().lower() != target_lower
    )
    return brier_score
