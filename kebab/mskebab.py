# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Entry point to the MS-KeBAB package."""

from __future__ import annotations

import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from kebab.contracts.task import Task, TaskInstance, TaskType
from kebab.tasks.entity_generation import EntityGenerationTaskInstance
from kebab.tasks.extraction.task import ExtractionTaskInstance
from kebab.tasks.linking import LinkingTaskInstance


class Benchmark:
    """The entry point class."""

    __tasks: dict[TaskType, Task]
    __task_instances: dict[str, TaskInstance]

    @property
    def tasks(self) -> dict[TaskType, Task]:
        """Return copy of task list."""
        return self.__tasks.copy()  # To disallow modifying this dictionary from outside of this class

    @property
    def task_instances(self) -> dict[str, TaskInstance]:
        """Return copy of task instance list."""
        return self.__task_instances.copy()  # To disallow modifying this dictionary from outside of this class

    def __init__(self, path: Path):
        """Initialize entry point."""
        self.__tasks = {}
        self.__task_instances = {}
        Benchmark.__register_task_types()
        with open(path) as f:
            task_instance_config = json.load(f)
        for instance_name, instance_config in task_instance_config.items():
            task_type = TaskType[instance_config["task"]]
            if task_type not in self.__tasks:
                task = Task(task_type)
                self.__tasks[task_type] = task
            else:
                task = self.__tasks[task_type]
            self.__task_instances[instance_name] = TaskInstance.create_task_instance(
                instance_name, task, instance_config["data"]
            )

    @staticmethod
    def __register_task_types() -> None:
        """Register available task types with their corresponding classes."""
        TaskInstance.register_task_type(TaskType.EntityGeneration, EntityGenerationTaskInstance)
        TaskInstance.register_task_type(TaskType.Extraction, ExtractionTaskInstance)
        TaskInstance.register_task_type(TaskType.Linking, LinkingTaskInstance)


class Cache:
    """The cache class."""

    __cache_dir: Path
    __file_map_path: Path
    __file_map: dict[str, Path]

    def __init__(self, cache_dir: Path):
        """Initialize cache."""
        self.__cache_dir = cache_dir
        self.__file_map_path = cache_dir / "map.json"
        if self.__file_map_path.exists():
            with open(self.__file_map_path) as f:
                self.__file_map = {k: Path(v) for k, v in json.load(f).items()}
        else:
            self.__file_map = {}
        self.validate_and_save_map()

    def get_cached_path(self, path: str) -> Path:
        """Get path to locally cached file."""
        if urllib.parse.urlparse(str(path)).scheme == "":  # Is URL
            return Path(path)
        if path not in self.__file_map or not self.__file_map[path].exists():
            self.__file_map[path] = self.cache_file(path)
            self.validate_and_save_map()
        return self.__file_map[path]

    def cache_file(self, url: str) -> Path:
        """Download file to cache and return local path."""
        path = Path(url)
        local_path = self.__cache_dir / path.name
        if not local_path.exists():
            self.download_file(url, local_path)
            return local_path
        i = 1
        while True:
            local_path = self.__cache_dir / (Path(f"{path.stem}_{i}{path.suffix}"))
            if not local_path.exists():
                self.download_file(url, local_path)
                return local_path
            i += 1

    def download_file(self, url: str, path: Path):
        """Download file to cache."""
        if not url.startswith(("http:", "https:")):
            raise ValueError("URL must start with 'http:' or 'https:'")

        # Suppressing ruff check S310 for the next line of code because of a known issue.
        # See: https://github.com/astral-sh/ruff/issues/7918
        with urllib.request.urlopen(url) as response, open(path, "wb") as out_file:  # noqa: S310
            shutil.copyfileobj(response, out_file)

    def clear(self):
        """Clear the cache."""
        shutil.rmtree(self.__cache_dir)
        self.__file_map.clear()
        self.validate_and_save_map()

    def validate_and_save_map(self):
        """Validate and save the cache map."""
        self.__cache_dir.mkdir(parents=True, exist_ok=True)
        for k, v in self.__file_map.items():
            if not v.exists() or not v.is_file() or v.parents[0] != self.__cache_dir:
                del self.__file_map[k]
        file_map_str_dict = {k: str(v) for k, v in self.__file_map.items()}
        with open(self.__file_map_path, "w") as f:
            json.dump(file_map_str_dict, f)


def benchmark() -> Benchmark:
    """Initialize and return benchmark object."""
    return Benchmark(Path(__file__).parents[0] / "configs" / "instances.json")


def cache(cache_dir_path: Path = Path(".cache")) -> Cache:
    """Initialize and return cache object."""
    return Cache(Path(cache_dir_path))
