# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the T-REx fragment extraction steps."""

from __future__ import annotations

import pathlib

import click

from kebab import utils
from kebab.dataset.trex.t_rex_fragment_extractor import TRexFragmentExtractor


@click.option(
    "--trex-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "data" / "trex_dataset",
    help="Path to the T-REx data directory (will be *.json-globbed against).",
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "data" / "trex_fragments",
    help="Path to the output directory ('all_entities_fragments.jsonl' will be created there).",
)
@click.option(
    "--path-to-wikidata-properties",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "data" / "wikidata" / "wikidata_properties.json",
    help="Path to the Wikidata properties file (optional).",
)
@click.command()
def main(
    trex_dir: pathlib.Path,
    output_dir: pathlib.Path,
    path_to_wikidata_properties: pathlib.Path,
) -> None:
    """Run T-REx fragment extraction steps."""
    utils.configure_logging()

    extractor = TRexFragmentExtractor(
        trex_dir=trex_dir,
        output_dir=output_dir,
        path_to_wikidata_properties=path_to_wikidata_properties,
    )

    extractor.run()


if __name__ == "__main__":
    main()
