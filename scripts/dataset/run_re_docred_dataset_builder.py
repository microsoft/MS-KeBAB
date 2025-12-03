# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the Re-DocRED extraction dataset creation steps."""

from __future__ import annotations

import logging
from pathlib import Path

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.re_docred.re_docred_dataset_builder import ReDocRedDatasetBuilder


@click.option(
    "--re-docred-dir",
    type=Path,
    default=Path.cwd() / "data" / "Re-DocRED" / "Original Dataset" / "dev",
    help="Path to the Re-DocRED data directory (will be *.json-globbed against).",
)
@click.option(
    "--wikidata-properties-path",
    type=Path,
    default=Path.cwd() / "data" / "Wikidata" / "Properties" / "wikidata_properties.json",
    help="Path to the Wikidata properties file.",
)
@click.option(
    "--output-dir",
    type=Path,
    default=Path.cwd() / "data" / "Re-DocRED" / "Extractions"/ "dev_updated",
    help="Path to the output directory ('extracts.jsonl' and 'entities.jsonl' will be created there).",
)
@click.option(
    "--log-level",
    type=str,
    default="INFO",
    help="Logging level to use.",
)
@click.command()
def main(
    re_docred_dir: Path,
    wikidata_properties_path: Path,
    output_dir: Path,
    log_level: str,
) -> None:
    """Run Re-DocRED extraction dataset creation steps."""
    logging_helpers.configure_logging(level=getattr(logging, log_level.upper(), logging.INFO))
    builder = ReDocRedDatasetBuilder(
        re_docred_dir=re_docred_dir,
        wikidata_properties_path=wikidata_properties_path,
        output_dir=output_dir,
    )

    builder.run()


if __name__ == "__main__":
    main()
