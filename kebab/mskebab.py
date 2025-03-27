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
from kebab.tasks.clustering import ClusteringTaskInstance


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

        with open(path) as f:
            task_instance_config = json.load(f)

        for instance_name, instance_config in task_instance_config.items():
            task_type = TaskType[instance_config["task"]]
            task = Task(task_type)
            self.__tasks[task_type] = task

            match task_type:
                case TaskType.EntityGeneration:
                    task_class = EntityGenerationTaskInstance
                case TaskType.Extraction:
                    task_class = ExtractionTaskInstance
                case TaskType.Linking:
                    task_class = LinkingTaskInstance
                case TaskType.Clustering:
                    task_class = ClusteringTaskInstance
                case _:
                    raise ValueError(f"No registered class for task type: {task.task_type}.")
            self.__task_instances[instance_name] = task_class(
                instance_name, task, **instance_config["data"]
            )

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

        self.__file_map = {
            k: v for k, v in self.__file_map.items() if v.exists() and v.is_file() and v.parents[0] == self.__cache_dir
        }

        file_map_str_dict = {k: str(v) for k, v in self.__file_map.items()}

        with open(self.__file_map_path, "w") as f:
            json.dump(file_map_str_dict, f)


def benchmark(path: Path | None = None) -> Benchmark:
    """Initialize and return benchmark object."""
    if not path:
        path = Path(__file__).parents[0] / "configs" / "instances.json"
    return Benchmark(path)


def cache(cache_dir_path: Path = Path(".cache")) -> Cache:
    """Initialize and return cache object."""
    return Cache(Path(cache_dir_path))
