# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from kebab.contracts.document import Document
from kebab.contracts.entity import Entity


@dataclass
class ExtractionOutput:
    """Extraction output."""
    document: Document
    entities: list[Entity]


class MetricsCalculator(abc.ABC):
    """Extraction metrics calculator."""

    @abc.abstractmethod
    def run(self, prediction: list[ExtractionOutput], ground_truth:list[ExtractionOutput]) -> dict[str, Any]:
        """Calculate extraction metrics."""
        raise NotImplementedError
