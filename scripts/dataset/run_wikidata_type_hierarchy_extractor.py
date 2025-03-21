# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the Wikidata type hierarchy extraction steps."""

from __future__ import annotations

import pathlib

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.wikidata.wikidata_type_hierarchy_extractor import WikidataTypeHierarchyExtractor


@click.option(
    "--wikidata-json-dump-path",
    type=pathlib.Path,
    default=pathlib.Path.home() / "latest-all.json",
    help="The path to the standard Wikidata JSON dump file.",
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "output" / "wikidata_hierarchy",
    help="The directory to save the extracted hierarchy.",
)
@click.command()
def main(
    wikidata_json_dump_path: pathlib.Path,
    output_dir: pathlib.Path,
):
    """Run Wikidata hierarchy extraction steps."""
    logging_helpers.configure_logging()

    extractor = WikidataTypeHierarchyExtractor(
        wikidata_json_dump_path=wikidata_json_dump_path,
        output_dir=output_dir,
    )

    extractor.run()


if __name__ == "__main__":
    main()
