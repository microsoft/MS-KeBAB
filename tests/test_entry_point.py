# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the benchmark entry point."""

# ruff: noqa: RUF015, PLR2004

import kebab.mskebab as mskebab

def test_entry_point():
  benchmark = mskebab.benchmark()

  tasks = benchmark.tasks
  assert set(tasks.keys()) == set(["extraction", "linking"])
  for task_id, task in tasks.items():
    assert task_id == task.id

  task_instances = benchmark.task_instances
  assert set(task_instances.keys()) == set(["extraction_redocred", "linking_trex", "linking_mave"])
  for task_instance_id, task_instance in task_instances.items():
    assert task_instance_id == task_instance.id

  for task in tasks.values():
    for task_instance in task.instances.values():
      assert task_instance.parent_id == task.id
      assert set(task_instance.data.keys()) == set(task.req_data)



