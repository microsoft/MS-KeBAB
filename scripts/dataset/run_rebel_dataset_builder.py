# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the REBEL linking and clustering datasets building steps."""

from __future__ import annotations

import pathlib

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.rebel.rebel_dataset_builder import MergeDistributionMode, RebelDatasetBuilder


@click.option(
    "--rebel-fragments-path",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "data" / "REBEL" / "Fragments" / "Resolved" / "rebel_entity_fragments.jsonl",
    help="Path to the file containing REBEL fragments.",
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "output",
    help="Path to the output directory (dataset *.jsonl files will be created there).",
)
@click.option(
    "--max-count",
    type=int,
    default=5_000_000,
    help="The maximum number of pairs to include in the dataset. A negative value is ignored.",
)
@click.option(
    "--max-merge-fragments",
    type=int,
    default=10,
    help="The maximum number of fragments to merge into a single entity. A negative value is ignored.",
)
@click.option(
    "--merge-distribution",
    type=click.Choice([mode.value for mode in MergeDistributionMode], case_sensitive=False),
    default=MergeDistributionMode.ZIPF.value,
    help="The distribution mode to use for merging fragments into entities.",
)
@click.option(
    "--deduplicate/--no-deduplicate",
    is_flag=True,
    default=True,
    help="Whether to deduplicate property values in the merged fragments. Default is True.",
)
@click.command()
def main(
    rebel_fragments_path: pathlib.Path,
    output_dir: pathlib.Path,
    max_count: int,
    max_merge_fragments: int,
    merge_distribution: str,
    deduplicate: bool,
) -> None:
    """Run REBEL linking and clustering datasets building steps."""
    logging_helpers.configure_logging()

    builder = RebelDatasetBuilder(
        fragments_path=rebel_fragments_path,
        output_dir=output_dir,
        max_count=max_count if max_count > 0 else None,
        max_merge_fragments=max_merge_fragments,
        merge_distribution=MergeDistributionMode(merge_distribution) if merge_distribution else None,
        deduplicate_values=deduplicate,
    )

    builder.run()


if __name__ == "__main__":
    main()
