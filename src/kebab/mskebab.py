# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Entry point to the MS-KeBAB package."""

from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Self
from . import metrics

def safe_merge_list_of_dictionary(dict_list: list[dict[str, Any]]) -> dict[str, Any]:
  """Merge a list of dictionaries into a single dictionary ensuring no duplicate keys in input."""
  merged_dict = {}
  for d in dict_list:
    for k,v in d.items():
      assert k not in merged_dict
      merged_dict[k] = v
  return merged_dict

@dataclass
class Benchmark:
  """The entry point class."""

  tasks: dict[str, Task]
  task_instances: dict[str, Task]

  @classmethod
  def init(cls, path) -> Self:
    """Load configuration"""
    with open(path) as f:
      config = json.load(f)
      tasks = {k: Task.from_dict({**{"id": k}, **v}) for k,v in config["tasks"].items()}
      task_instances = safe_merge_list_of_dictionary([task.instances for task in tasks.values()])
    return cls(
      tasks=tasks,
      task_instances=task_instances
    )

@dataclass
class Task:
  """Represents a benchmark task with its task instances."""

  id: str
  req_data: list
  instances: dict

  @classmethod
  def from_dict(cls, data_dict: dict[str, Any]) -> Self:
    """Create a task from a dictionary."""
    task_id=data_dict["id"]
    req_data=data_dict["req_data"]
    instances={k: TaskInstance.from_dict({**{"id": k}, **{"parent_id": task_id}, **v}, req_data) for k,v in data_dict["instances"].items()}
    return cls(
      id=task_id,
      req_data=req_data,
      instances=instances
    )

@dataclass
class TaskInstance:
  """Represents a benchmark task instance with its data files."""

  id: str
  parent_id: str
  data: dict

  @classmethod
  def from_dict(cls, data_dict: dict[str, Any], req_data: list) -> Self:
    """Create a task instance from a dictionary."""
    instance_id=data_dict["id"]
    parent_id=data_dict["parent_id"]
    data=data_dict["data"]
    assert len(data) == len(req_data)
    for k in data.keys():
      assert k in req_data
    return cls(
      id=instance_id,
      parent_id=parent_id,
      data=data
    )

  def evaluate(self, **kwargs):
    eval_func = getattr(metrics, "evaluate_{}".format(self.parent_id))
    return eval_func(**kwargs)

def benchmark() -> Benchmark:
  return Benchmark.init(Path(__file__).parents[0] / "config.json")
