# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script to run the REBEL dataset split sampler to pick challenging examples and produce Test/Validation/Train splits.

Output layout (within output_dir):
    output_dir/linking/<split>/rebel_linking_dataset.jsonl
    output_dir/linking/<split>/rebel_linking_ground_truth.jsonl
    output_dir/linking/<split>/rebel_linking_used_entities.jsonl
    output_dir/clustering/<split>/rebel_clustering_dataset.jsonl
    output_dir/clustering/<split>/rebel_clustering_ground_truth.jsonl
    output_dir/entity_generation/<split>/rebel_entity_generation_dataset.jsonl
    output_dir/fragment_generation/<split>/rebel_fragment_generation_dataset.jsonl
"""

from __future__ import annotations

from pathlib import Path

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.rebel.rebel_split_sampler import (
    RebelSplitSampler,
    TypeFilter,
    default_splits,
)


@click.option(
    "--input",
    "input_dir",
    type=Path,
    default=Path.cwd() / "data" / "REBEL",
    help=(
        "Path to the dataset directory produced by RebelDatasetBuilder. "
        "This directory must contain the base linking/clustering/generation jsonl files."
    ),
)
@click.option(
    "--output-dir",
    type=Path,
    default=Path.cwd() / "data" / "REBEL",
    help="Path to the target output directory",
)
@click.option(
    "--required-type-id",
    type=str,
    multiple=True,
    help=(
        "Optional Wikidata type IDs to include. If provided, only entities whose types intersect these (or their descendants, "
        "when --include-descendants is set) will be kept. Can be passed multiple times."
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
    type=Path,
    default=Path.cwd() / "Datasets" / "Wikidata" / "type_hierarchy" / "wikidata_type_hierarchy.jsonl",
    help="Path to the Wikidata type hierarchy jsonl file (required when using type filters).",
)
@click.option(
    "--include-descendants/--no-include-descendants",
    is_flag=True,
    default=False,
    help="Whether to include descendants of required/excluded type IDs in type filtering.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    show_default=True,
    help=("Random seed to make sampling deterministic across runs."),
)
@click.command()
def main(
    input_dir: Path,
    output_dir: Path,
    required_type_id: tuple[str, ...],
    exclude_type_id: tuple[str, ...],
    type_hierarchy_path: Path,
    include_descendants: bool,
    seed: int | None,
) -> None:
    """Run REBEL split sampler to produce Test/Validation/Train splits."""
    logging_helpers.configure_logging()

    # Build optional type filter only if any option was provided
    type_filter: TypeFilter | None = None
    if required_type_id or exclude_type_id:
        type_filter = TypeFilter(
            type_hierarchy_path=type_hierarchy_path,
            required_type_ids=list(required_type_id) or None,
            exclude_type_ids=list(exclude_type_id) or None,
            include_descendants=include_descendants,
        )

    sampler = RebelSplitSampler(
        input_dir=input_dir,
        output_dir=output_dir,
        splits=default_splits(seed=seed),
        type_filter=type_filter,
    )

    sampler.run()


if __name__ == "__main__":
    main()
