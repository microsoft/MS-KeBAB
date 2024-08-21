# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the benchmark entry point."""

# ruff: noqa: RUF015, PLR2004

import kebab.mskebab as mskebab

def test_task_interface():
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
      test_ground_truth_file = [data_file for data_file in task_instance.data.keys() if data_file.startswith("test_ground_truth")][0]
      metrics = task_instance.evaluate(output="some_output_file", ground_truth=test_ground_truth_file)
      assert metrics["primary_{}_metric".format(task.id)] == 0.8
      assert metrics["secondary_{}_metric".format(task.id)] == 0.6
