# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path
from typing import Any

from kebab.contracts.document import Document
from kebab.contracts.entity import Entity


class CustomEncoder(json.JSONEncoder):
    """JSON encoder for entities and documents."""

    def default(self, o: Any) -> Any:  # noqa: ANN401, D102
        if isinstance(o, Entity):
            return o.to_dict()
        if isinstance(o, Document):
            return o.to_dict()
        return super().default(o)


def save_to_json(data: Any, file_path: Path) -> None:
    """
    Save data to a JSON file.

    Args:
        data: The serializable data to save.
        file_path: The path where the data should be saved.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
