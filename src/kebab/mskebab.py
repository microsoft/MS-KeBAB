# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Entry point to the MS-KeBAB package."""

from __future__ import annotations

import json
import shutil
import urllib.parse
import urllib.request
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

@dataclass
class Cache:
  """The cache manager class"""

  cache_dir: Path
  file_map_path: Path
  file_map: dict[str, Path]

  @classmethod
  def init(cls, cache_dir_path: Path) -> Self:
    """Load configuration"""
    file_map_path = cache_dir_path / "map.json"
    if file_map_path.exists():
        with open(file_map_path) as f:
          file_map = {k: Path(v) for k,v in json.load(f).items()}
    else:
      file_map = {}
    obj = cls(
      cache_dir=cache_dir_path,
      file_map_path=file_map_path,
      file_map=file_map
    )
    obj.validate_and_save_map()
    return obj

  def get_cached_path(self, path: str) -> Path:
    if urllib.parse.urlparse(str(path)).scheme == "": # Is URL
      return Path(path)
    if not path in self.file_map or not self.file_map[path].exists():
      self.file_map[path] = self.cache_file(path)
      self.validate_and_save_map()
    return self.file_map[path]

  def cache_file(self, url: str) -> Path:
    path = Path(url)
    local_path = self.cache_dir / path.name
    if not local_path.exists():
      self.download_file(url, local_path)
      return local_path
    i = 1
    while True:
      local_path = self.cache_dir / (Path(path.stem + "_" + str(i) + path.suffix))
      if not local_path.exists():
        self.download_file(url, local_path)
        return local_path
      i += 1

  def download_file(self, url: str, path: Path):
    with urllib.request.urlopen(url) as response:
      with open(str(path), 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

  def clear(self):
    shutil.rmtree(self.cache_dir)
    self.file_map.clear()
    self.validate_and_save_map()

  def validate_and_save_map(self):
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    for k,v in self.file_map.items():
      if not v.exists() or not v.is_file() or v.parents[0] != self.cache_dir:
        del self.file_map[k]
    file_map_str_dict = {k: str(v) for k,v in self.file_map.items()}
    with open(self.file_map_path, 'w') as f:
      json.dump(file_map_str_dict, f)

def benchmark() -> Benchmark:
  return Benchmark.init(Path(__file__).parents[0] / "config.json")

def cache(cache_dir_path: Path = Path.cwd() / ".cache") -> Cache:
  return Cache.init(Path(cache_dir_path))
