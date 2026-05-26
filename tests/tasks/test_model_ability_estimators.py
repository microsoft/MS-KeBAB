from collections import defaultdict
from typing import cast

import numpy as np
import pytest
from scipy.stats import kendalltau
from tests.tasks.model_ability_estimators import TaskInstance, run_all_estimators


# To view the output of the print statements, run this test with:
# pytest -s tests/tasks/test_perplexity_estimators.py
@pytest.mark.parametrize("nonlinear", [True, False])
@pytest.mark.parametrize("minimum_informativeness", [-0.9])  # [-0.9, -1, -1.1])
def test_model_ability_estimators(nonlinear: bool, minimum_informativeness: float) -> None:
    """Compare estimators for model performance in a setting where we have m models and n words,
    and the performance of each model on each word is given by a linear function of the model and word parameters plus Gaussian noise.
    We want to see which estimator of model performance has the highest Kendall correlation with the true model parameters."""

    trial_count = 10
    correlations = defaultdict(lambda: [(0.0, 0.0)] * trial_count)
    rng = np.random.default_rng(1)
    for trial in range(trial_count):
        m = 4  # Number of models
        n = 1000  # Number of words
        if True:
            # Deliberately imperfect choice of base and evaluator models to make the task non-trivial but not impossible
            base_model = 1
            evaluator_model = m - 2
        else:
            base_model = 0
            evaluator_model = m - 1

        # Generate m evenly spaced numbers between 0 and 1
        abilities = np.linspace(0, 1, m)
        # Generate n evenly spaced numbers between -1 and 1, with the option to make them mostly negative to test estimator robustness in that setting
        informativeness = np.linspace(minimum_informativeness, 1, n)
        difficulty = rng.normal(0, 1, n)  # Random difficulty for each word
        if nonlinear:
            # construct a matrix where the first row is zero, the last row is one, and middle rows increase.
            # we do this by taking the cumulative sum of positive random numbers.
            increments = rng.gamma(1, 1, (m, n)) / rng.gamma(1, 1, (m, n))
            increments[0, :] = 0
            data = np.cumsum(increments, axis=0)
            data = data / data[m - 1, :] * informativeness[None, :]
        else:
            # for each model and word, compute the product of the model and word and add Gaussian noise with mean 0 and standard deviation 1
            data = abilities[:, None] * informativeness[None, :]
        data = difficulty[None, :] + rng.normal(data, 9e-1)
        task = TaskInstance(
            data=data,
            base_model=base_model,
            evaluator_model=evaluator_model,
            informativeness=informativeness,
        )
        estimates = run_all_estimators(task, verbose=trial_count == 1)

        if trial_count == 1:
            print(f"True model parameters: {abilities}")
            for name, estimate in estimates.items():
                print(f"{name} estimator: {estimate}")

        # Compute the correlation between the true model parameters and all estimators
        for name, estimate in estimates.items():
            tau = cast(float, kendalltau(abilities, estimate)[0])
            rho = cast(float, np.corrcoef(abilities, estimate)[0, 1])
            correlations[name][trial] = (tau, rho)
    avg_correlations = {name: np.mean(pairs, axis=0) for name, pairs in correlations.items()}
    for name, avg in avg_correlations.items():
        print(f"Average correlation for {name} estimator: {avg}")
        if not name.startswith("ideal") and not name.startswith("exhaustive"):
            # Assert that alternating estimator beats all other non-ideal estimators
            for i in range(len(avg)):
                assert avg_correlations["alternating"][i] >= avg[i]
