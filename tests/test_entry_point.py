# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the benchmark entry point."""

# ruff: noqa: RUF015, PLR2004

from pathlib import Path
import kebab.mskebab as mskebab

def test_task_interface():
  benchmark = mskebab.benchmark()

  tasks = benchmark.tasks
  assert set(tasks.keys()) == set(["Extraction", "Linking"])
  for task_name, task in tasks.items():
    assert task_name == task.name

  task_instances = benchmark.task_instances
  assert set(task_instances.keys()) == set(["Extraction-ReDocRED",
                                            "Extraction-Heldout",
                                            "Linking-TREx",
                                            "Linking-MAVE",
                                            "Linking-Heldout"])
  for task_instance_name, task_instance in task_instances.items():
    assert task_instance_name == task_instance.name

  for task in tasks.values():
    for task_instance in task.instances.values():
      assert task_instance.parent == task
      for set_type in ["train", "dev", "test"]:
        if set_type != "test" or not task_instance.heldout:
          metrics = task_instance.evaluate("some_output_file", set_type)
          assert metrics["primary_{}_metric".format(task.name.lower())] == 0.8
          assert metrics["secondary_{}_metric".format(task.name.lower())] == 0.6

def test_cache_manager():
    cache_dir = Path(".local_cache").resolve()
    cache = mskebab.cache(cache_dir)
    cache.clear()
    assert cache.get_cached_path("http://w3.erss.univ-tlse2.fr/UETAL/2022-2023/sigirfp093-mitra.pdf") == cache_dir / "sigirfp093-mitra.pdf"
    assert cache.get_cached_path("http://w3.erss.univ-tlse2.fr/UETAL/2022-2023/sigirfp093-mitra.pdf") == cache_dir / "sigirfp093-mitra.pdf"
    assert cache.get_cached_path("https://www.microsoft.com/en-us/research/wp-content/uploads/2015/08/sigirfp093-mitra.pdf") == cache_dir / "sigirfp093-mitra_1.pdf"
