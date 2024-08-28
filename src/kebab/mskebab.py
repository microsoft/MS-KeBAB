# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Entry point to the MS-KeBAB package."""

from __future__ import annotations

import json
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from . import task_lib


@dataclass
class Benchmark:
    """The entry point class."""

    _tasks: dict[str, task_lib.Task]
    _task_instances: dict[str, task_lib.TaskInstance]

    @property
    def tasks(self) -> dict[str, task_lib.Task]:
        """Return copy of task list."""
        return (
            self._tasks.copy()
        )  # To disallow modifying this dictionary from outside of this class

    @property
    def task_instances(self) -> dict[str, task_lib.TaskInstance]:
        """Return copy of task instance list."""
        return (
            self._task_instances.copy()
        )  # To disallow modifying this dictionary from outside of this class

    @classmethod
    def init(cls, path: Path) -> Self:
        """Initialize entry point."""
        tasks = {}
        task_instances = {}
        with open(path) as f:
            task_instance_config = json.load(f)
        for instance_name, instance_config in task_instance_config.items():
            task_type = instance_config["task"]
            task, instance = task_lib.TaskInstance.create_task_instance(
                instance_name, instance_config, tasks.get(task_type)
            )
            tasks[task_type] = task
            task_instances[instance.name] = instance
        return cls(_tasks=tasks, _task_instances=task_instances)


@dataclass
class Cache:
    """The cache class."""

    _cache_dir: Path
    _file_map_path: Path
    _file_map: dict[str, Path]

    @classmethod
    def init(cls, cache_dir_path: Path) -> Self:
        """Initialize cache."""
        file_map_path = cache_dir_path / "map.json"
        if file_map_path.exists():
            with open(file_map_path) as f:
                file_map = {k: Path(v) for k, v in json.load(f).items()}
        else:
            file_map = {}
        obj = cls(
            _cache_dir=cache_dir_path, _file_map_path=file_map_path, _file_map=file_map
        )
        obj.validate_and_save_map()
        return obj

    def get_cached_path(self, path: str) -> Path:
        """Get path to locally cached file."""
        if urllib.parse.urlparse(str(path)).scheme == "":  # Is URL
            return Path(path)
        if path not in self._file_map or not self._file_map[path].exists():
            self._file_map[path] = self.cache_file(path)
            self.validate_and_save_map()
        return self._file_map[path]

    def cache_file(self, url: str) -> Path:
        """Download file to cache and return local path."""
        path = Path(url)
        local_path = self._cache_dir / path.name
        if not local_path.exists():
            self.download_file(url, local_path)
            return local_path
        i = 1
        while True:
            local_path = self._cache_dir / (Path(f"{path.stem}_{i}{path.suffix}"))
            if not local_path.exists():
                self.download_file(url, local_path)
                return local_path
            i += 1

    def download_file(self, url: str, path: Path):
        """Download file to cache."""
        if not url.startswith(("http:", "https:")):
            raise ValueError("URL must start with 'http:' or 'https:'")
        with urllib.request.urlopen(url) as response, open(path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

    def clear(self):
        """Clear the cache."""
        shutil.rmtree(self._cache_dir)
        self._file_map.clear()
        self.validate_and_save_map()

    def validate_and_save_map(self):
        """Validate and save the cache map."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        for k, v in self._file_map.items():
            if not v.exists() or not v.is_file() or v.parents[0] != self._cache_dir:
                del self._file_map[k]
        file_map_str_dict = {k: str(v) for k, v in self._file_map.items()}
        with open(self._file_map_path, "w") as f:
            json.dump(file_map_str_dict, f)


def benchmark() -> Benchmark:
    """Initialize and return benchmark object."""
    return Benchmark.init(Path(__file__).parents[0] / "instances.json")


def cache(cache_dir_path: Path = Path(".cache")) -> Cache:
    """Initialize and return cache object."""
    return Cache.init(Path(cache_dir_path))
