# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script to run the REBEL dataset subsampler to pick challenging examples and produce Test/Validation/Train splits.

Output layout (within output_dir):
    output_dir/linking/<Split>/rebel_linking_dataset.jsonl
    output_dir/linking/<Split>/rebel_linking_ground_truth.jsonl
    output_dir/linking/<Split>/rebel_linking_used_entities.jsonl
    output_dir/clustering/<Split>/rebel_clustering_dataset.jsonl
    output_dir/clustering/<Split>/rebel_clustering_ground_truth.jsonl
    output_dir/entity_generation/<Split>/rebel_entity_generation_dataset.jsonl
    output_dir/fragment_generation/<Split>/rebel_fragment_generation_dataset.jsonl
"""

from __future__ import annotations

import pathlib

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.rebel.rebel_dataset_subsampler import (
    RebelDatasetSubsampler,
    TypeFilter,
    default_splits,
)


@click.option(
    "--base-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "output",
    help=(
        "Path to the base dataset directory produced by RebelDatasetBuilder. "
        "This directory must contain the base linking/clustering/generation jsonl files."
    ),
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "output",
    help="Path to the target output directory",
)
@click.option(
    "--required-type-id",
    type=str,
    default="",
    help=(
        "Optional Wikidata type ID to include. If provided, only entities whose types intersect this (or its descendants, "
        "when --include-descendants is set) will be kept."
    ),
)
@click.option(
    "--exclude-type-id",
    type=str,
    multiple=True,
    help=(
        "Optional Wikidata type IDs to exclude. Any entity whose types intersect these (or their descendants, "
        "when --include-descendants is set) will be skipped. Can be passed multiple times."
    ),
)
@click.option(
    "--type-hierarchy-path",
    type=pathlib.Path,
    default=pathlib.Path.cwd()
    / "Datasets"
    / "Wikidata"
    / "Type Hierarchy"
    / "2025-01-30"
    / "wikidata_type_hierarchy.jsonl",
    help="Path to the Wikidata type hierarchy jsonl file (required when using type filters).",
)
@click.option(
    "--include-descendants/--no-include-descendants",
    is_flag=True,
    default=False,
    help="Whether to include descendants of required/excluded type IDs in type filtering.",
)
@click.command()
def main(
    base_dir: pathlib.Path,
    output_dir: pathlib.Path,
    required_type_id: str,
    exclude_type_id: tuple[str, ...],
    type_hierarchy_path: pathlib.Path,
    include_descendants: bool,
) -> None:
    """Run REBEL subsampler to produce Test/Validation/Train splits in one go."""
    logging_helpers.configure_logging()

    # Build optional type filter only if any option was provided
    type_filter: TypeFilter | None = None
    if required_type_id or exclude_type_id:
        type_filter = TypeFilter(
            type_hierarchy_path=type_hierarchy_path,
            required_type_id=required_type_id or None,
            exclude_type_ids=list(exclude_type_id) or None,
            include_descendants=include_descendants,
        )

    sampler = RebelDatasetSubsampler(
        base_dir=base_dir,
        output_dir=output_dir,
        splits=default_splits(),
        type_filter=type_filter,
    )

    sampler.run()


if __name__ == "__main__":
    main()
