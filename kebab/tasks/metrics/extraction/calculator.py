# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kebab.contracts.document import Document
from kebab.contracts.entity import Entity, PropertySchema


@dataclass
class ExtractionOutput:
    """Extraction output."""

    document: Document
    entities: list[Entity]


class MetricCalculator(abc.ABC):
    """Extraction metric calculator."""

    def __init__(
        self,
        config: dict[str, Any],
        property_schema: PropertySchema,
        logger: logging.Logger | None = None,
        debug_output_path: Path | None = None,
    ) -> None:
        """Initialize metric calculator.

        Args:
            config: Metric configuration.
            property_schema: Property schema for the entities.
            logger: Optional logger instance.
            debug_output_path: Optional path to save debug information.
        """
        self.config = config
        self.property_schema = property_schema
        self.logger = logger
        self.debug_output_path = debug_output_path

    @abc.abstractmethod
    def run(self, prediction: Iterable[ExtractionOutput], ground_truth: Iterable[ExtractionOutput]) -> dict[str, Any]:
        """Calculate extraction metrics."""
        raise NotImplementedError
