# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Script to run the Wikidata names dataset builder."""

from __future__ import annotations

import pathlib

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.wikidata.wikidata_names_dataset_builder import WikidataNamesDatasetBuilder


@click.option(
    "--wikidata-simple-entities-path",
    type=pathlib.Path,
    default=pathlib.Path.home()
    / "OneDrive - Microsoft"
    / "Benchmark"
    / "Datasets"
    / "Wikidata"
    / "Simple Entities"
    / "2024-06-27"
    / "wikidata_simple_entities.jsonl",
    help="The path to the extracted Wikidata simple entities.",
)
@click.option(
    "--wikidata-type-hierarchy-path",
    type=pathlib.Path,
    default=pathlib.Path.home()
    / "OneDrive - Microsoft"
    / "Benchmark"
    / "Datasets"
    / "Wikidata"
    / "Type Hierarchy"
    / "2024-06-27"
    / "wikidata_type_hierarchy.jsonl",
    help="The path to the Wikidata type hierarchy.",
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "output" / "wikidata_names",
    help="The directory to save the extracted names.",
)
@click.option(
    "--target-type-id",
    type=str,
    default="Q154954",
    help="The type ID to extract names for. E.g. Q154954 for person.",
)
@click.command()
def main(
    wikidata_simple_entities_path: pathlib.Path,
    wikidata_type_hierarchy_path: pathlib.Path,
    output_dir: pathlib.Path,
    target_type_id: str,
):
    """Run Wikidata name extraction steps."""
    logging_helpers.configure_logging()

    extractor = WikidataNamesDatasetBuilder(
        wikidata_entities_path=wikidata_simple_entities_path,
        wikidata_type_hierarchy_path=wikidata_type_hierarchy_path,
        output_dir=output_dir,
        type_ids=[target_type_id],
    )

    extractor.run()


if __name__ == "__main__":
    main()
