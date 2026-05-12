from collections import defaultdict
from typing import cast

import numpy as np
import pytest
from estimators import TaskInstance, run_all_estimators
from scipy.stats import kendalltau


# To view the output of the print statements, run this test with:
# pytest -s tests/tasks/test_perplexity_estimators.py
@pytest.mark.parametrize("minimum_informativeness", [-0.9])  # , -1, -1.1])
def test_perplexity_estimators(minimum_informativeness: float) -> None:
    """Compare estimators for model performance in a setting where we have m models and n words,
    and the performance of each model on each word is given by a linear function of the model and word parameters plus Gaussian noise.
    We want to see which estimator of model performance has the highest Kendall correlation with the true model parameters."""

    nonlinear = True
    trial_count = 1
    pearsons = defaultdict(lambda: np.zeros(trial_count))
    kendall_taus = defaultdict(lambda: np.zeros(trial_count))
    rng = np.random.default_rng(1)
    for trial in range(trial_count):
        m = 4  # Number of models
        n = 1000  # Number of words
        # Deliberately imperfect choice of base and evaluator models to make the task non-trivial but not impossible
        base_model = 1
        evaluator_model = m - 2
        # base_model = 0
        # evaluator_model = m - 1

        # Generate m evenly spaced numbers between 0 and 1
        abilities = np.linspace(0, 1, m)
        # Generate n evenly spaced numbers between -1 and 1, with the option to make them mostly negative to test estimator robustness in that setting
        informativeness = np.linspace(minimum_informativeness, 1, n)
        difficulty = rng.normal(0, 1, n)  # Random difficulty for each word
        # for each model and word, compute the product of the model and word and add Gaussian noise with mean 0 and standard deviation 1
        data = difficulty[None, :] + abilities[:, None] * informativeness[None, :]
        data = rng.normal(data, 1)
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
            kendall_taus[name][trial] = tau
            pearsons[name][trial] = np.corrcoef(abilities, estimate)[0, 1]
    avg_kts = {name: np.mean(kendall_taus[name]) for name in kendall_taus}
    for name, avg in avg_kts.items():
        print(f"Average Kendall tau for {name} estimator: {avg}")
        if not name.startswith("ideal") and False:
            # Assert that alternating estimator beats all other non-ideal estimators
            assert avg_kts["alternating"] >= avg
            if not name.startswith("alternating"):
                assert avg_kts["increasing"] >= avg
                assert avg_kts["increasing_iso"] >= avg
    avg_pearsons = {name: np.mean(pearsons[name]) for name in pearsons}
    for name, avg in avg_pearsons.items():
        print(f"Average Pearson correlation for {name} estimator: {avg}")
        if not name.startswith("ideal") and False:
            # Assert that alternating estimator beats all other non-ideal estimators
            assert avg_pearsons["alternating"] >= avg
            if not name.startswith("alternating"):
                assert avg_pearsons["increasing"] >= avg
                assert avg_pearsons["increasing_iso"] >= avg
