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
from . import task_lib

@dataclass
class Benchmark:
  """The entry point class."""

  _tasks: dict[str, task_lib.Task]
  _task_instances: dict[str, task_lib.Task]

  @property
  def tasks(self) -> dict[str, task_lib.Task]:
    return self._tasks

  @property
  def task_instances(self) -> dict[str, task_lib.Task]:
    return self._task_instances

  @classmethod
  def init(cls, path) -> Self:
    """Initialize entry point"""
    task_list = [task_lib.ExtractionTask(), task_lib.LinkingTask()]
    tasks = {task.name: task for task in task_list}
    task_instances = {}
    with open(path) as f:
      task_instance_config = json.load(f)
    for instance_name, instance_config in task_instance_config.items():
      if instance_config["task"] == "Extraction":
        instance = task_lib.ExtractionTaskInstance(instance_name, instance_config["data"], tasks["Extraction"], instance_config["heldout"])
      elif instance_config["task"] == "Linking":
        instance = task_lib.LinkingTaskInstance(instance_name, instance_config["data"], tasks["Linking"], instance_config["heldout"])
      else:
        raise ValueError("Unknown task type for task instance")
      task_instances[instance.name] = instance
      instance.parent.add_instance(instance)
    return cls(
      _tasks=tasks,
      _task_instances=task_instances
    )

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
  return Benchmark.init(Path(__file__).parents[0] / "instances.json")

def cache(cache_dir_path: Path = Path.cwd() / ".cache") -> Cache:
  return Cache.init(Path(cache_dir_path))
