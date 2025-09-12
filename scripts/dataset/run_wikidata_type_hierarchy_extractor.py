# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script to run the Wikidata type hierarchy extraction steps.

Requires a standard Wikidata JSON dump file, which can be downloaded from
https://dumps.wikimedia.org/wikidatawiki/entities/
"""

from __future__ import annotations

from pathlib import Path

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.wikidata.wikidata_type_hierarchy_extractor import WikidataTypeHierarchyExtractor


@click.option(
    "--wikidata-json-dump-path",
    type=Path,
    default=Path.cwd() / "data" / "latest-all.json",
    help="The path to the standard Wikidata JSON dump file.",
)
@click.option(
    "--output-dir",
    type=Path,
    default=Path.cwd() / "output" / "wikidata_hierarchy",
    help="The directory to save the extracted hierarchy.",
)
@click.command()
def main(
    wikidata_json_dump_path: Path,
    output_dir: Path,
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
