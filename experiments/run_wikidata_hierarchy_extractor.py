# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the Wikidata hierarchy extraction steps."""

from __future__ import annotations

import pathlib

import click

from kebab import utils
from kebab.dataset.wikidata.wikidata_hierarchy_extractor import WikidataHierarchyExtractor


@click.option(
    "--path-to-wikidata-json-dump",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "data" / "wikidata" / "latest-all.json",
    help="The path to the Wikidata JSON dump file.",
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "data" / "wikidata_hierarchy",
    help="The directory to save the extracted hierarchy.",
)
@click.command()
def main(
    path_to_wikidata_json_dump: pathlib.Path,
    output_dir: pathlib.Path,
):
    """Run Wikidata hierarchy extraction steps."""
    utils.configure_logging()

    extractor = WikidataHierarchyExtractor(
        path_to_wikidata_json_dump=path_to_wikidata_json_dump,
        output_dir=output_dir,
    )

    extractor.run()


if __name__ == "__main__":
    main()
