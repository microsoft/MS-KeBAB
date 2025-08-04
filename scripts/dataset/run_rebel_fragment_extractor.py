# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the REBEL fragment extraction steps."""

from __future__ import annotations

import pathlib

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.rebel.rebel_fragment_extractor import RebelFragmentExtractor


@click.option(
    "--rebel-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "data" / "REBEL" / "Original Dataset" / "full",
    help="Path to the REBEL data directory (will be *json/*.jsonl-globbed against).",
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "output",
    help="Path to the output directory ('rebel_entity_fragments.jsonl' will be created there).",
)
@click.command()
def main(
    rebel_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    """Run REBEL fragment extraction steps."""
    logging_helpers.configure_logging()

    extractor = RebelFragmentExtractor(
        rebel_dir=rebel_dir,
        output_dir=output_dir,
    )

    extractor.run()


if __name__ == "__main__":
    main()
