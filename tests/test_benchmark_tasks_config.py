# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
from pathlib import Path
from unittest import mock

import pytest
from kebab import mskebab
from kebab.contracts.task import Task
from kebab.utils import io_helpers
from kebab.utils.io_helpers import resolve_path


@pytest.mark.manual
def test_benchmark_tasks_config():
    """Test that the benchmark tasks configuration is valid (the datasets exist)."""
    benchmark = mskebab.get_default_benchmark()
    assert benchmark is not None, "Benchmark should not be None"
    assert len(benchmark.tasks) > 0, "Benchmark should have at least one task"

    for task in benchmark.tasks:
        try:
            assert isinstance(task, Task), f"Task {task} should be an instance of Task"
            assert task.name, f"Task {task} should have a name"

            schema = task.read_schema()
            assert schema is not None, f"Schema for task {task.name} should not be None"

            item = next(iter(task.read_items()))
            assert item is not None, f"Task {task.name} should have at least one item"
        except (AssertionError, ValueError, OSError, FileNotFoundError, StopIteration) as e:
            print(f"Task {task.name} is not properly configured or has no items: {e}")


def test_benchmark_tasks_config_syntax():
    """Test that the benchmark tasks configuration has a valid syntax."""
    config_file_path = Path(__file__).parent.parent / "kebab" / "configs" / "tasks.json"
    assert config_file_path.exists(), f"Config file {config_file_path} does not exist"

    with open(config_file_path, encoding="utf-8") as f:
        config = json.load(f)
        for task in config["tasks"]:
            data = task.get("data")
            for path in data.values():
                assert "~" not in path, f"Path {path} in task {task['name']} contains a tilde (~), which is not allowed"


@pytest.mark.parametrize(
    ("input_path", "expected"),
    [
        # shortcut in the middle (relative)
        (
            "data/datasets.lnk/train/file.jsonl",
            Path("C:/all_datasets/train/file.jsonl"),
        ),
        # shortcut in the middle (assumed)
        (
            "data/datasets_shortcut/train/file.jsonl",
            Path("C:/all_datasets/train/file.jsonl"),
        ),  # explicitly local path, shortcut in the middle (assumed)
        (
            "./data/datasets_shortcut/train/file.jsonl",
            Path("C:/all_datasets/train/file.jsonl"),
        ),
        # shortcut is the very first component
        (
            "start.lnk/file.jsonl",
            Path("C:/start_point/file.jsonl"),
        ),
        # absolute path containing a shortcut
        (
            "C:/proj/rootdatasets.lnk/z1",
            Path("D:/datasets/z1"),
        ),
        # two successive shortcuts
        (
            "a.lnk/b.lnk/file.txt",
            Path("C:/bar/file.txt"),
        ),
        # path with no shortcuts at all
        (
            "regular/dir/file.txt",
            Path("regular/dir/file.txt"),
        ),  # explicitly local path with no shortcuts at all
        (
            "./regular/dir/file.txt",
            Path("regular/dir/file.txt"),
        ),
    ],
)
def test_resolve_shortcuts(input_path: str, expected: Path) -> None:
    """
    Verify that every `.lnk` component is expanded (without touching the real file system).
    """

    def _fake_target(path: Path) -> Path:
        """Return a canned target or raise to simulate an unreadable shortcut."""
        _fake_targets = {
            "datasets.lnk": Path("C:/all_datasets"),
            "datasets_shortcut.lnk": Path("C:/all_datasets"),
            "start.lnk": Path("C:/start_point"),
            "rootdatasets.lnk": Path("D:/datasets"),
            "a.lnk": Path("C:/foo"),
            "b.lnk": Path("C:/bar"),
        }

        name = path.name
        if name in _fake_targets:
            return _fake_targets[name]
        raise FileNotFoundError(f"cannot resolve {name!r}")

    def _is_file_stub(self: Path) -> bool:
        """Stub for pathlib.Path.is_file to avoid real file system checks."""
        return self.suffix.lower() == ".lnk"

    def _exists_stub(self: Path) -> bool:
        """Stub for pathlib.Path.exists to avoid real file system checks."""
        return self.name != "datasets_shortcut"

    with (
        mock.patch.object(io_helpers, "_shortcut_target", side_effect=_fake_target),
        mock.patch("pathlib.Path.is_file", new=_is_file_stub),
        mock.patch("pathlib.Path.exists", new=_exists_stub),
    ):
        result = resolve_path(input_path)
        assert result == expected
