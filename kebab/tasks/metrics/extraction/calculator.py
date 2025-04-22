# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from kebab.contracts.document import Document
from kebab.contracts.entity import Entity, PropertySchema


@dataclass
class ExtractionOutput:
    """Extraction output."""

    document: Document
    entities: list[Entity]


class MetricConfig:
    """Metric configuration."""

    @staticmethod
    @abc.abstractmethod
    def from_dict(config: dict[str, Any], property_schema: PropertySchema) -> MetricConfig:
        """Create metric configuration from dictionary."""
        raise NotImplementedError


class MetricCalculator(abc.ABC):
    """Extraction metric calculator."""

    @abc.abstractmethod
    def run(self, prediction: Iterable[ExtractionOutput], ground_truth: Iterable[ExtractionOutput]) -> dict[str, Any]:
        """Calculate extraction metrics."""
        raise NotImplementedError
