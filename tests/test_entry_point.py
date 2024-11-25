# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the benchmark entry point."""

# ruff: noqa: RUF015, PLR2004

from __future__ import annotations

from pathlib import Path

from kebab import mskebab
from kebab.contracts.task import TaskType


def test_task_interface():
    """Test task interface."""
    benchmark = mskebab.benchmark()

    tasks = benchmark.tasks
    assert set(tasks.keys()) == {TaskType.Extraction, TaskType.Linking}
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
    }
    for task_instance_name, task_instance in task_instances.items():
        assert task_instance_name == task_instance.name

    for task in tasks.values():
        for task_instance in task.instances.values():
            assert task_instance.task == task
            if task_instance.name.endswith("-Heldout"):
                metrics = task_instance.evaluate(Path("some_output_file"))
                assert metrics[f"primary_{task.task_type.name.lower()}_metric"] == 0.8
                assert metrics[f"secondary_{task.task_type.name.lower()}_metric"] == 0.6


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
