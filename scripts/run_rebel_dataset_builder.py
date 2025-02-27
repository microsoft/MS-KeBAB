# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the REBEL linking and clustering datasets building steps."""

from __future__ import annotations

import pathlib

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.rebel.rebel_dataset_builder import RebelDatasetBuilder


@click.option(
    "--rebel-fragments-path",
    type=pathlib.Path,
    default=pathlib.Path.home()
    / "OneDrive - Microsoft"
    / "Benchmark"
    / "Datasets"
    / "REBEL"
    / "Fragments Resolved"
    # / "sample"
    / "rebel_entity_fragments.jsonl",
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
    default=100_000,
    help="The maximum number of pairs to include in the dataset. A negative value is ignored.",
)
@click.command()
def main(rebel_fragments_path: pathlib.Path, output_dir: pathlib.Path, max_count: int) -> None:
    """Run REBEL linking and clustering datasets building steps."""
    logging_helpers.configure_logging()

    builder = RebelDatasetBuilder(
        fragments_path=rebel_fragments_path,
        output_dir=output_dir,
        max_count=max_count if max_count > 0 else None,
    )

    builder.run()


if __name__ == "__main__":
    main()
