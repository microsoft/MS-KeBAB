# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from kebab.tasks.metrics.text_completion.utils import get_target_content_prob_with_fallback
from sklearn.isotonic import IsotonicRegression


@dataclass
class ModelInfo:
    """Information about a model for the alternating estimator."""

    predictions_path: Path
    ability: float = 0.0
    word_logprobs: list[float] = field(default_factory=list)


class AlternatingEstimator:
    """Alternating estimator for evaluating the model abilities based on word informativeness."""

    def __init__(
        self,
        good_model_infos: dict[str, ModelInfo],
        bad_model_infos: dict[str, ModelInfo],
        dont_know_model_infos: dict[str, ModelInfo],
    ) -> None:
        """Initialize the estimator with model infos for good, bad, and don't know models."""
        self.model_infos = AlternatingEstimator.__init_model_infos(
            good_model_infos, bad_model_infos, dont_know_model_infos
        )

    def run(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
    ) -> tuple[list[float], list[float], list[float], int]:
        """Run the alternating estimator to estimate model abilities and word informativeness."""
        update_model_abilities: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = (
            AlternatingEstimator.__update_model_abilities_keywords_auto
        )

        model_abilities = [info.ability for info in self.model_infos.values()]
        word_logprobs = [info.word_logprobs for info in self.model_infos.values()]

        logprobs = np.array(word_logprobs)  # (num_models, num_words)
        avg_logprobs = logprobs.mean(axis=0)  # (num_words,)
        abilities = np.array(model_abilities)  # (num_models,)
        informativeness = np.zeros(logprobs.shape[1])
        max_change = float("inf")

        num_iterations = 0
        for _ in range(max_iterations):
            num_iterations += 1

            # Estimate word informativeness from model abilities using isotonic regression.
            # For each word, fit a monotonically increasing function from abilities to logprobs,
            # then measure informativeness as signed R² (fraction of variance explained).
            informativeness = AlternatingEstimator.__estimate_informativeness_isotonic(abilities, logprobs)

            # Estimate model abilities from word informativeness.
            new_abilities = update_model_abilities(abilities, informativeness, logprobs)

            # Check convergence.
            max_change = float(np.max(np.abs(new_abilities - abilities)))
            abilities = new_abilities
            if max_change < convergence_threshold:
                break
        else:
            print(f"Warning: did not converge after {max_iterations} iterations (max_change={max_change:.2e}).")

        return abilities.tolist(), informativeness.tolist(), avg_logprobs.tolist(), num_iterations

    @staticmethod
    def __estimate_informativeness_isotonic(
        abilities: np.ndarray,
        logprobs: np.ndarray,
    ) -> np.ndarray:
        """Estimate word informativeness via isotonic regression R².

        For each word, fits both increasing and decreasing monotonic regressions
        from model abilities to logprobs. Informativeness is the signed R²:
        positive if the increasing fit is better (good models score higher),
        negative if the decreasing fit is better (anti-correlated).
        """
        num_words = logprobs.shape[1]
        informativeness = np.zeros(num_words)
        sort_order = np.argsort(abilities)
        sorted_abilities = abilities[sort_order]

        iso_inc = IsotonicRegression(increasing=True)
        iso_dec = IsotonicRegression(increasing=False)

        for w in range(num_words):
            word_logprobs = logprobs[:, w]
            total_var = np.var(word_logprobs)
            if total_var == 0.0:
                continue

            sorted_logprobs = word_logprobs[sort_order]

            fitted_inc = iso_inc.fit_transform(sorted_abilities, sorted_logprobs)
            residual_var_inc = np.mean((sorted_logprobs - fitted_inc) ** 2)
            r2_inc = 1.0 - residual_var_inc / total_var

            fitted_dec = iso_dec.fit_transform(sorted_abilities, sorted_logprobs)
            residual_var_dec = np.mean((sorted_logprobs - fitted_dec) ** 2)
            r2_dec = 1.0 - residual_var_dec / total_var

            if r2_inc >= r2_dec:
                informativeness[w] = r2_inc
            else:
                informativeness[w] = -r2_dec

        return informativeness

    @staticmethod
    def __update_model_abilities_weighted_avg(
        model_abilities: np.ndarray,
        word_informativeness: np.ndarray,
        word_logprobs: np.ndarray,
    ) -> np.ndarray:
        # ability[m] = weighted average of logprobs, weighted by max(0, informativeness).
        weights = np.maximum(0.0, word_informativeness)  # (num_words,)
        total_weight = weights.sum()
        if total_weight == 0.0:
            new_abilities = np.zeros_like(model_abilities)
        else:
            new_abilities = word_logprobs @ weights / total_weight  # (num_models,)
        return new_abilities

    @staticmethod
    def __update_model_abilities_keywords_by_cutoff(
        model_abilities: np.ndarray,
        word_informativeness: np.ndarray,
        word_logprobs: np.ndarray,
        informativeness_percentage_cutoff: float = 0.5,
    ) -> np.ndarray:
        # Select top informativeness_percentage_cutoff of the positively informative words.
        # For each model, ability is average logprob of the selected words.
        positive_mask = word_informativeness > 0
        positive_informativeness = word_informativeness[positive_mask]
        if len(positive_informativeness) == 0:
            return np.zeros_like(model_abilities)

        num_to_keep = max(1, int(len(positive_informativeness) * informativeness_percentage_cutoff))
        threshold = np.sort(positive_informativeness)[-num_to_keep]
        selected_mask = word_informativeness >= threshold

        return word_logprobs[:, selected_mask].mean(axis=1)

    @staticmethod
    def __update_model_abilities_keywords_auto(
        model_abilities: np.ndarray,
        word_informativeness: np.ndarray,
        word_logprobs: np.ndarray,
    ) -> np.ndarray:
        # sort words in descending informativeness, keeping only positive informativeness.  Find the n that maximizes mean(informativeness[1:n])*sqrt(n).
        positive_mask = word_informativeness > 0
        positive_informativeness = word_informativeness[positive_mask]
        if len(positive_informativeness) == 0:
            return np.zeros_like(model_abilities)

        sorted_desc = np.sort(positive_informativeness)[::-1]
        cumulative_mean = np.cumsum(sorted_desc) / np.arange(1, len(sorted_desc) + 1)
        sqrt_n = np.sqrt(np.arange(1, len(sorted_desc) + 1))
        scores = cumulative_mean * sqrt_n
        best_n = int(np.argmax(scores)) + 1
        threshold = sorted_desc[best_n - 1]
        selected_mask = word_informativeness >= threshold

        return word_logprobs[:, selected_mask].mean(axis=1)

    @staticmethod
    def __init_model_infos(
        good_model_infos: dict[str, ModelInfo],
        bad_model_infos: dict[str, ModelInfo],
        dont_know_model_infos: dict[str, ModelInfo],
    ) -> dict[str, ModelInfo]:
        all_names = list(good_model_infos) + list(bad_model_infos) + list(dont_know_model_infos)
        if not all_names:
            raise ValueError("At least one model info dict must be non-empty.")
        if len(set(all_names)) != len(all_names):
            duplicates = {name for name in all_names if all_names.count(name) > 1}
            raise ValueError(f"Duplicate model names across groups: {duplicates}")

        init_good_model_ability = 1.0
        init_bad_model_ability = -1.0
        init_dont_know_model_ability = 0.0

        all_model_infos: dict[str, ModelInfo] = {}
        for model_infos, ability in [
            (good_model_infos, init_good_model_ability),
            (bad_model_infos, init_bad_model_ability),
            (dont_know_model_infos, init_dont_know_model_ability),
        ]:
            for model_name, info in model_infos.items():
                with open(info.predictions_path, encoding="utf-8") as f:
                    predictions = json.load(f)
                word_logprobs = [
                    math.log(get_target_content_prob_with_fallback(item))
                    for item in predictions
                    if item["text_with_mask"]
                ]
                all_model_infos[model_name] = replace(info, ability=ability, word_logprobs=word_logprobs)

        return all_model_infos
