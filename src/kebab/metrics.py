# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Metrics implementation for all MS-KeBAB benchmark tasks."""

from __future__ import annotations

def evaluate_extraction(**kwargs) -> dict[str, float]:
  """Evaluate an output for the extraction task."""

  # TODO: Implement actual metric computation
  return {
    "primary_extraction_metric": 0.8,
    "secondary_extraction_metric": 0.6
    }

def evaluate_linking(**kwargs) -> dict[str, float]:
  """Evaluate an output for the linking task."""

  # TODO: Implement actual metric computation
  return {
    "primary_linking_metric": 0.8,
    "secondary_linking_metric": 0.6
    }
