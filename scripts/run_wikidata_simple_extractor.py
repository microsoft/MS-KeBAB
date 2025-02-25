# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the Wikidata WikidataEntity/property extraction steps."""

from __future__ import annotations

import pathlib

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.wikidata.wikidata_simple_extractor import WikidataSimpleExtractor


@click.option(
    "--run-entity-extraction/--no-run-entity-extraction",
    default=False,
    help="Whether to run the entity extraction step.",
)
@click.option(
    "--run-property-fetch/--no-run-property-fetch",
    default=False,
    help="Whether to run the property fetching step.",
)
@click.option(
    "--wikidata-json-dump-path",
    type=pathlib.Path,
    default=pathlib.Path.home() / "latest-all.json",
    help="The path to the standard Wikidata JSON dump file.",
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
    run_property_fetch: bool,
    wikidata_json_dump_path: pathlib.Path,
    output_dir: pathlib.Path,
):
    """Run Wikidata hierarchy extraction steps."""
    logging_helpers.configure_logging()

    extractor = WikidataSimpleExtractor(
        wikidata_json_dump_path=wikidata_json_dump_path,
        run_entity_extraction=run_entity_extraction,
        run_property_fetch=run_property_fetch,
        output_dir=output_dir,
    )

    extractor.run()


if __name__ == "__main__":
    main()
