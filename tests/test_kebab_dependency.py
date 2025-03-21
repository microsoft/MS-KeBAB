from __future__ import annotations

from kebab import mskebab
from kebab.contracts.entity import ValueType
from kebab.contracts.task import Task, TaskType


def test_kebab_import() -> None:
    """Test the import of the kebab module."""
    v = ValueType.TEXT.value
    assert v == "Text"

    task_instances = mskebab.benchmark().task_instances
    assert task_instances is not None
    assert len(task_instances) > 0

    Task.clear_created_task_types()


def test_kebab_import_limited() -> None:
    """Test the import of the kebab module with limited tasks."""
    task_instances = mskebab.benchmark(allowed_tasks=[TaskType.Linking]).task_instances
    assert task_instances is not None
    assert len(task_instances) > 0
    assert {instance.task.task_type for instance in task_instances.values()} == {TaskType.Linking}

    Task.clear_created_task_types()
