# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from itertools import permutations
from typing import cast

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class TaskInstance:
    data: np.ndarray
    base_model: int
    evaluator_model: int
    informativeness: np.ndarray | None = None


EstimatorMethod = Callable[[], np.ndarray]


class EstimatorContext:
    """Shared, cached intermediate values plus estimator methods."""

    def __init__(self, task: TaskInstance, t: float = 0.0, verbose: bool = False):
        self.task = task
        self.t = t
        self.verbose = verbose
        self._cache: dict[str, np.ndarray] = {}

    def cached(self, key: str, build: Callable[[], np.ndarray]) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def data(self) -> np.ndarray:
        return self.task.data

    @property
    def delta(self) -> np.ndarray:
        return self.cached("delta", lambda: self.data - self.data[self.task.base_model, None])

    @property
    def clipped(self) -> np.ndarray:
        """Clip the data to be non-negative"""
        return self.cached("clipped", lambda: np.clip(self.delta - self.t, 0, None))

    @property
    def is_positive(self) -> np.ndarray:
        return self.cached("is_positive", lambda: self.delta - self.t > 0)

    def estimator_mean(self) -> np.ndarray:
        return self.data.mean(axis=1)

    def estimator_mean_clipped(self) -> np.ndarray:
        return self.clipped.mean(axis=1)

    def estimator_mean_clipped2(self) -> np.ndarray:
        return np.mean(self.clipped * self.clipped, axis=1)

    def estimator_mean_clipped3(self) -> np.ndarray:
        return np.mean(self.clipped * self.clipped * self.clipped, axis=1)

    def estimator_var_clipped(self) -> np.ndarray:
        return np.var(self.clipped, axis=1)

    def estimator_mean_clipped_plus(self) -> np.ndarray:
        return self.clipped.mean(axis=1) + self.is_positive.mean(axis=1)

    def estimator_subset_mean_clipped(self) -> np.ndarray:
        """Mean of the positive values only"""
        return safe_divide(self.clipped.sum(axis=1), self.is_positive.sum(axis=1))

    def estimator_weighted_clip(self) -> np.ndarray:
        c = np.clip(self.delta, 0, None)
        return safe_divide(np.sum(c * np.sqrt(c), axis=1), np.sum(c, axis=1))

    def estimator_weighted_clip2(self) -> np.ndarray:
        c = np.clip(self.delta, 0, None)
        return safe_divide(np.sum(c * c, axis=1), np.sum(c, axis=1))

    def estimator_weighted_clip3(self) -> np.ndarray:
        c = np.clip(self.delta, 0, None)
        return safe_divide(np.sum(c * c * c, axis=1), np.sum(c * c, axis=1))

    def estimator_longppl(self) -> np.ndarray:
        is_keyword = top(self.delta[self.task.evaluator_model, :])
        if self.verbose:
            print(f"Proportion of keywords: {np.mean(is_keyword)}")
        return self.data[:, is_keyword].mean(axis=1)

    def estimator_longppl2(self) -> np.ndarray:
        max_word_effect = np.max(self.delta, axis=0)
        is_keyword2 = top(max_word_effect)
        if self.verbose:
            print(f"Proportion of keywords (method 2): {np.mean(is_keyword2)}")
            # print(f"Proportion of common keywords: {np.mean(is_keyword & is_keyword2)}")
        longppl2_estimator = self.data[:, is_keyword2].mean(axis=1)
        return longppl2_estimator

    def estimator_longppl3(self) -> np.ndarray:
        word_effect = np.mean(self.delta, axis=0)
        is_keyword3 = top(word_effect)
        if self.verbose:
            print(f"Proportion of keywords (method 3): {np.mean(is_keyword3)}")
        return self.data[:, is_keyword3].mean(axis=1)

    def estimator_ideal_longppl(self) -> np.ndarray:
        is_keyword_ideal = top(self.task.informativeness)
        if self.verbose:
            print(f"Proportion of keywords (ideal): {np.mean(is_keyword_ideal)}")
        return self.data[:, is_keyword_ideal].mean(axis=1)

    def estimator_ideal(self) -> np.ndarray:
        if self.task.informativeness is None:
            raise ValueError("Informativeness is required for the ideal estimator")
        positive_words = self.task.informativeness[self.task.informativeness > 0]
        w = positive_words * positive_words
        return safe_divide(
            np.dot(self.data[:, self.task.informativeness > 0] / positive_words[None, :], w),
            np.sum(w),
        )

    def estimator_alternating(self) -> np.ndarray:
        m, n = self.data.shape
        ability_estimate = np.zeros(m)
        ability_estimate[self.task.evaluator_model] = 1
        ability_estimate[self.task.base_model] = -1
        if False:
            ability_estimate = self.estimator_exhaustive_increasing()
        # word_std = np.std(self.data, axis=0)
        word_mean = np.mean(self.data, axis=0)
        standardized = self.data - word_mean[None, :]  # / (word_std[None, :] + 1e-8)
        if False:
            i = 1
            ability_estimate[0 : i - 1] = -1
            ability_estimate[i:] = 1
        if False:  # special first iteration
            has_nonzero_ability = ability_estimate != 0
            subset_mean = np.mean(self.data[has_nonzero_ability, :], axis=0)
            subset_word_effect = np.dot(
                ability_estimate[has_nonzero_ability],
                self.data[has_nonzero_ability, :] - subset_mean[None, :],
            )
            # is_keyword_subset = top(subset_word_effect)
            # longppl_iter_estimator = data[:, is_keyword_subset].mean(axis=1)
            weight = np.maximum(0, subset_word_effect)
            ability_estimate = np.dot(self.data, weight) / np.sum(weight)
        for _ in range(25):
            # ability_estimate's mean doesn't need to be subtracted because standardized has zero mean
            informativeness = np.dot(ability_estimate, standardized)
            # is_keyword_iter = topk(word_effect_iter, n // pct)
            if False:
                word_effect_iter2 = -distance_to_sorted_all(ability_estimate, standardized)
                is_keyword_iter2 = topk(word_effect_iter2, n // 4)
                is_keyword_iter = topk(informativeness, n // 4)
                in1not2 = is_keyword_iter & ~is_keyword_iter2
                in2not1 = is_keyword_iter2 & ~is_keyword_iter
                print(f"in1not2: {np.mean(in1not2)} in2not1: {np.mean(in2not1)}")
                # is_keyword_iter = top(word_effect_iter)
                # print(
                #     f"Proportion of keywords (iterative method): {np.mean(is_keyword_iter)} Propportion positive: {np.mean(word_effect_iter > 0)}"
                # )
                ability_estimate = self.data[:, is_keyword_iter].mean(axis=1)
            else:
                weight = np.maximum(0, informativeness)
                # standarization is irrelevant here since it will shift all estimates by a constant
                ability_estimate = np.dot(self.data, weight) / np.sum(weight)
        return ability_estimate

    def estimator_alternating_iso(self) -> np.ndarray:
        m, n = self.data.shape
        ability_estimate = np.zeros(m)
        ability_estimate[self.task.evaluator_model] = 1
        ability_estimate[self.task.base_model] = -1
        word_mean = np.mean(self.data, axis=0)
        standardized = self.data - word_mean[None, :]
        iso_inc = IsotonicRegression(increasing=True)
        for _ in range(25):
            sort_order = np.argsort(ability_estimate)
            sorted_abilities = ability_estimate[sort_order]
            sorted_matrix = standardized[sort_order, :]
            informativeness = np.zeros(n)
            for c in range(n):
                sorted_logprobs = sorted_matrix[:, c]
                fitted_inc = iso_inc.fit_transform(sorted_abilities, sorted_logprobs)
                if True:
                    informativeness[c] = fitted_inc[-1]
                else:
                    residual_var = np.mean((sorted_logprobs - fitted_inc) ** 2)
                    total_var = np.var(sorted_logprobs)
                    informativeness[c] = 1 - residual_var / total_var

            weight = np.maximum(0, informativeness)
            # standarization is irrelevant here since it will shift all estimates by a constant
            ability_estimate = np.dot(self.data, weight) / np.sum(weight)
        return ability_estimate

    @property
    def orderings(self):
        model_count = self.data.shape[0]
        return list(permutations(range(model_count)))

    def score_abilities_linear(self, abilities, standardized):
        if abilities[self.task.base_model] > abilities[self.task.evaluator_model]:
            return -np.inf
        informativeness = np.dot(abilities, standardized) / np.dot(abilities, abilities)
        dist = (standardized - abilities[:, None] * informativeness[None, :]) ** 2
        return np.sum(-dist)

    def estimator_exhaustive_linear(self) -> np.ndarray:
        word_mean = np.mean(self.data, axis=0)
        standardized = self.data - word_mean[None, :]
        ordering_scores = [
            self.score_abilities_linear(np.asarray(ordering), standardized) for ordering in self.orderings
        ]
        return np.asarray(self.orderings[np.argmax(ordering_scores)])

    def score_ordering(self, ordering, matrix) -> float:
        if ordering[self.task.base_model] > ordering[self.task.evaluator_model]:
            return -np.inf
        dist = distance_to_sorted_all(ordering, matrix)
        # threshold = 1
        # return np.sum(np.clip(threshold - dist, 0, None))
        return np.sum(-dist)

    def estimator_exhaustive_increasing(self) -> np.ndarray:
        ordering_scores = [self.score_ordering(np.asarray(ordering), matrix=self.data) for ordering in self.orderings]
        if self.verbose:
            print(f"Ordering scores: {ordering_scores}")
            # dist = distance_to_sorted_all(np.asarray([3, 1, 2, 0]), self.data)
            # print(f"Distances to isotonic fit for ordering [3,1,2,0]: {dist}")
        return np.asarray(self.orderings[np.argmax(ordering_scores)])

    def score_ordering_iso(self, ordering, matrix):
        if ordering[self.task.base_model] > ordering[self.task.evaluator_model]:
            return -np.inf
        dist = distance_to_sorted_all_iso(np.asarray(ordering), matrix)
        return np.sum(-dist)

    def estimator_exhaustive_iso(self) -> np.ndarray:
        ordering_scores = [self.score_ordering_iso(ordering, matrix=self.data) for ordering in self.orderings]
        if self.verbose:
            print(f"Ordering scores iso: {ordering_scores}")
            # dist = distance_to_sorted_all_iso(np.asarray([3, 1, 2, 0]), self.data)
            # print(f"Distances to isotonic fit for ordering [3,1,2,0]: {dist}")
        return np.asarray(self.orderings[np.argmax(ordering_scores)])


def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def topk(arr, k):
    k = int(k)
    if k <= 0:
        return np.zeros(arr.shape, dtype=bool)
    k = min(k, arr.size)
    threshold = np.sort(arr)[arr.size - k]
    return arr >= threshold


def find_threshold(arr):
    if len(arr) < 1:
        raise ValueError("Array must have at least 1 element")
    sorted_arr = np.sort(arr)
    if sorted_arr[-1] < 0:
        raise ValueError("All elements are negative")
    z = np.zeros_like(arr)
    for i in range(len(sorted_arr)):
        m = np.mean(sorted_arr[-i - 1 :])
        v = i + 1
        z[i] = m * np.sqrt(v)
    i = np.argmax(z)
    threshold = sorted_arr[-i - 1]
    if threshold < 0:
        raise ValueError("Threshold is negative")
    return threshold


def top(arr):
    t = find_threshold(arr)
    return arr >= t


def distance_to_sorted(arr):
    """Compute the total amount that array elements would need to be increased to make the array sorted in non-decreasing order."""
    cost = 0
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
        else:
            cost += max_val - arr[i]
    return cost


def distance_to_sorted_all(ability, matrix: np.ndarray) -> np.ndarray:
    sorted_indices = np.argsort(ability)
    ncol = np.shape(matrix)[1]
    cost = np.zeros(ncol)
    max_val = matrix[sorted_indices[0], :]
    for i in range(1, len(sorted_indices)):
        cost += (np.maximum(0, max_val - matrix[sorted_indices[i], :])) ** 2
        max_val = np.maximum(max_val, matrix[sorted_indices[i], :])
    return cost


def distance_to_sorted_all_iso(ability, matrix: np.ndarray) -> np.ndarray:
    iso_inc = IsotonicRegression(increasing=True)
    ncol = np.shape(matrix)[1]
    sort_order = np.argsort(ability)
    sorted_abilities = ability[sort_order]
    sorted_matrix = matrix[sort_order, :]
    dist = np.zeros(ncol)
    for c in range(ncol):
        sorted_logprobs = sorted_matrix[:, c]
        fitted_inc = iso_inc.fit_transform(sorted_abilities, sorted_logprobs)
        dist[c] = np.mean((sorted_logprobs - fitted_inc) ** 2)
    return dist


def discover_estimators(ctx: EstimatorContext, prefix: str = "estimator_") -> dict[str, EstimatorMethod]:
    """
    Auto-discover estimators from EstimatorContext methods named estimator_*.
    """
    discovered: dict[str, EstimatorMethod] = {}
    for method_name, method in inspect.getmembers(ctx, predicate=callable):
        if not method_name.startswith(prefix):
            continue
        default_name = method_name.removeprefix(prefix)
        discovered[default_name] = cast(EstimatorMethod, method)
    return discovered


def run_all_estimators(
    task: TaskInstance,
    t: float = 0.0,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Run discovered estimators and return mapping name -> estimator vector."""
    ctx = EstimatorContext(task=task, t=t, verbose=verbose)
    estimators = discover_estimators(ctx)
    return {name: fn() for name, fn in estimators.items()}
