# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the benchmark entry point."""

# ruff: noqa: RUF015, PLR2004

from __future__ import annotations

from pathlib import Path

from kebab import mskebab
from kebab.contracts.task import TaskType


def assert_dicts_equal(dict1: dict, dict2: dict, epsilon: float = 1e-6):
    """Assert that two dictionaries are equal."""
    assert dict1.keys() == dict2.keys()
    for key, value1 in dict1.items():
        if isinstance(value1, dict):
            assert_dicts_equal(value1, dict2[key], epsilon)
        elif isinstance(value1, float):
            assert abs(value1 - dict2[key]) < epsilon
        else:
            assert value1 == dict2[key]


def test_task_interface():
    """Test task interface."""
    benchmark = mskebab.Benchmark(Path(__file__).parent / "data" / "test_task_instances.json")

    tasks = benchmark.tasks
    assert set(tasks.keys()) == {TaskType.Extraction, TaskType.Linking, TaskType.EntityGeneration, TaskType.Clustering}
    for task_type, task in tasks.items():
        assert task_type == task.task_type

    task_instances = benchmark.task_instances
    assert set(task_instances.keys()) == {
        "Extraction-ReDocRED-Train",
        "Extraction-ReDocRED-Test",
        "Extraction-Heldout",
        "Linking-TREx-Train",
        "Linking-TREx-Test",
        "Linking-MAVE",
        "Linking-Heldout",
        "Entity-Generation-REBEL",
        "Clustering-Heldout",
    }
    for task_instance_name, task_instance in task_instances.items():
        assert task_instance_name == task_instance.name
    task_instance_data = {
        "Extraction-Heldout": {
            "predictions": "tests/data/extraction/re_docred_dev_extracted_entities.jsonl",
            "metrics": {
                "aesop": {
                    "avg_property_precision": 0.23,
                    "avg_property_recall": 0.345,
                    "matched_count": 8,
                    "property_precision": {
                        "name": 0.692,
                        "type": 0.0,
                        "descriptions": 0.0,
                    },
                    "property_recall": {
                        "name": 0.692,
                        "type": 0.0,
                        "descriptions": None,
                    },
                    "unmatched_counts": {"extra_gt_entities": 47, "extra_pred_entities": 0, "pairs": 0},
                    "unmatched_fractions": {"extra_gt_entities": 0.854, "extra_pred_entities": 0.0, "pairs": 0.0},
                }
            },
        },
        "Linking-Heldout": {
            "predictions": "tests/data/linking/predictions.jsonl",
            "metrics": {
                "false_negative": 0.0,
                "false_positive": 0.0,
                "precision": 1.0,
                "recall": 1.0,
                "tnr": 1.0,
                "total": 2.0,
                "tpr": 1.0,
                "true_negative": 1.0,
                "true_positive": 1.0,
            },
        },
        "Clustering-Heldout": {
            "predictions": "tests/data/clustering/predictions.jsonl",
            "metrics": {
                "f1": 1.0,
                "fragments": 2,
                "ground_truth_clusters": 2,
                "precision": 1.0,
                "predicted_clusters": 2,
                "recall": 1.0,
            },
        },
    }
    for task in tasks.values():
        for task_instance in task.instances.values():
            assert task_instance.task == task
            if task_instance.name.endswith("-Heldout"):
                metrics = task_instance.evaluate(Path(task_instance_data[task_instance.name]["predictions"]))
                assert_dicts_equal(task_instance_data[task_instance.name]["metrics"], metrics, epsilon=1e-3)


def test_cache():
    """Test cache."""
    cache_dir = Path(".local_cache").resolve()
    cache = mskebab.cache(cache_dir)
    cache.clear()

    assert (
        cache.get_cached_path("http://w3.erss.univ-tlse2.fr/UETAL/2022-2023/sigirfp093-mitra.pdf")
        == cache_dir / "sigirfp093-mitra.pdf"
    )

    # The following assert is intentionally the same as the previous one to simulate
    # the case where the method is called for the same URL multiple times. The second
    # call should directly return the local path to the already downloaded and cached file
    assert (
        cache.get_cached_path("http://w3.erss.univ-tlse2.fr/UETAL/2022-2023/sigirfp093-mitra.pdf")
        == cache_dir / "sigirfp093-mitra.pdf"
    )

    assert (
        cache.get_cached_path(
            "https://www.microsoft.com/en-us/research/wp-content/uploads/2015/08/sigirfp093-mitra.pdf"
        )
        == cache_dir / "sigirfp093-mitra_1.pdf"
    )
