# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the Wikidata hierarchy extraction steps."""

from __future__ import annotations

import pathlib

import click
from kebab.utils import logging
from kebab.utils.dataset.wikidata.wikidata_hierarchy_extractor import WikidataHierarchyExtractor


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
    default=pathlib.Path.cwd() / "output" / "wikidata_hierarchy",
    help="The directory to save the extracted hierarchy.",
)
@click.command()
def main(
    wikidata_json_dump_path: pathlib.Path,
    output_dir: pathlib.Path,
):
    """Run Wikidata hierarchy extraction steps."""
    logging.configure_logging()

    extractor = WikidataHierarchyExtractor(
        wikidata_json_dump_path=wikidata_json_dump_path,
        output_dir=output_dir,
    )

    extractor.run()


if __name__ == "__main__":
    main()
