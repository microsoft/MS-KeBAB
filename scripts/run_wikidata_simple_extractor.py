# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the Wikidata simple entity/property extraction steps."""

from __future__ import annotations

import pathlib

import click

from kebab import utils
from kebab.dataset.wikidata.wikidata_simple_extractor import WikidataSimpleExtractor


@click.option(
    "--run-entity-extraction/--no-run-entity-extraction",
    # default=False,
    default=True,
    help="Whether to run the simple entity extraction step.",
)
@click.option(
    "--run-property-scrape/--no-run-property-scrape",
    default=False,
    help="Whether to run the property scraping step.",
)
@click.option(
    "--wikidata-json-dump-path",
    type=pathlib.Path,
    # default=pathlib.Path.cwd() / "data" / "wikidata" / "latest-all.json",
    default=pathlib.Path(
        r"C:\Users\pmyshkov\OneDrive - Microsoft\Benchmark\Datasets\Wikidata\Dump\Sample 10k\2024-06-27\sample_10k.json"
    ),
    help="The path to the Wikidata JSON dump file.",
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "output" / "wikidata",
    help="The directory to save the extracted data.",
)
@click.command()
def main(
    run_entity_extraction: bool,
    run_property_scrape: bool,
    wikidata_json_dump_path: pathlib.Path,
    output_dir: pathlib.Path,
):
    """Run Wikidata hierarchy extraction steps."""
    utils.configure_logging()

    extractor = WikidataSimpleExtractor(
        wikidata_json_dump_path=wikidata_json_dump_path,
        output_dir=output_dir,
    )

    if run_entity_extraction:
        extractor.extract_simple_entities()

    if run_property_scrape:
        extractor.scrape_properties()


if __name__ == "__main__":
    main()
