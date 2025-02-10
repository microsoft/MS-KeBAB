# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the Re-DocRED extraction dataset creation steps."""

from __future__ import annotations

import pathlib

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.re_docred.re_docred_dataset_builder import ReDocRedDatasetBuilder


@click.option(
    "--re-docred-dir",
    type=pathlib.Path,
    default=pathlib.Path.home()
    / "OneDrive - Microsoft"
    / "Benchmark"
    / "Datasets"
    / "Re-DocRED"
    / "Original Dataset"
    / "full",
    # / "dev",
    # / "test",
    # / "train",
    # / "single",
    help="Path to the Re-DocRED data directory (will be *.json-globbed against).",
)
@click.option(
    "--wikidata-properties-path",
    type=pathlib.Path,
    default=pathlib.Path.home()
    / "OneDrive - Microsoft"
    / "Benchmark"
    / "Datasets"
    / "Wikidata"
    / "Properties"
    / "2024-06-27"
    / "wikidata_properties.json",
    help="Path to the Wikidata properties file.",
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "output" / "re_docred_extractions" / "full",
    help="Path to the output directory ('extracts.jsonl' and 'entities.jsonl' will be created there).",
)
@click.command()
def main(
    re_docred_dir: pathlib.Path,
    wikidata_properties_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    """Run Re-DocRED extraction dataset creation steps."""
    logging_helpers.configure_logging()

    builder = ReDocRedDatasetBuilder(
        re_docred_dir=re_docred_dir,
        wikidata_properties_path=wikidata_properties_path,
        output_dir=output_dir,
    )

    builder.run()


if __name__ == "__main__":
    main()
