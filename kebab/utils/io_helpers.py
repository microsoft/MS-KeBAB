# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
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
