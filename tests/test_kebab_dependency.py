from __future__ import annotations

from kebab import mskebab
from kebab.contracts.entity import ValueType


def test_kebab_import() -> None:
    """Test the import of the kebab module."""
    v = ValueType.TEXT.value
    assert v == "Text"

    task_instances = mskebab.benchmark().task_instances
    assert task_instances is not None
    assert len(task_instances) > 0
