# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Script to run the Wikidata name extractor."""

from __future__ import annotations

import pathlib

import click

from kebab import utils
from kebab.dataset.wikidata.wikidata_names_extractor import WikidataNamesExtractor


@click.option(
    "--path-to-wikidata-concrete-entities",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "data" / "wikidata" / "wikidata_concrete_entities.jsonl",
    help="The path to the Wikidata concrete entities.",
)
@click.option(
    "--path-to-wikidata-type-hierarchy",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "data" / "wikidata" / "wikidata_type_hierarchy.json",
    help="The path to the Wikidata type hierarchy.",
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "data" / "wikidata_names",
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
    path_to_wikidata_concrete_entities: pathlib.Path,
    path_to_wikidata_type_hierarchy: pathlib.Path,
    output_dir: pathlib.Path,
    target_type_id: str,
):
    """Run Wikidata name extraction steps."""
    utils.configure_logging()

    extractor = WikidataNamesExtractor(
        path_to_wikidata_concrete_entities=path_to_wikidata_concrete_entities,
        path_to_wikidata_type_hierarchy=path_to_wikidata_type_hierarchy,
        output_dir=output_dir,
        type_ids=[target_type_id],
    )

    extractor.run()


if __name__ == "__main__":
    main()
