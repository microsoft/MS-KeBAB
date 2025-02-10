# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# ruff: noqa: ANN401
from __future__ import annotations

import re
from typing import Any

from dateutil.parser import parse as parse_date

from kebab.contracts.entity import Property, ValueType


def normalize_property_value(value: Any, property_: Property) -> Any:
    """Normalize property value.
    For ValueType.TEXT replaces all space like symbols to spaces, removes leading and trailing spaces and double spaces.
    For ValueType.DATE, if the date is a string, converts it to ISO 8601 format.
    Other value types remain unchanged.
    """
    if property_.data_type.value_type is ValueType.TEXT:
        return normalize_string(value)
    if property_.data_type.value_type is ValueType.DATE and isinstance(value, str):
        return parse_date(value).isoformat()
    return value


def normalize_string(value: str) -> str:
    """Replace all space like symbols to spaces, removes leading and trailing spaces and double spaces."""
    return re.sub(r"\s+", " ", value.strip()).lower()
