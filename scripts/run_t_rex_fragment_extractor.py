# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the T-REx fragment extraction steps."""

from __future__ import annotations

import pathlib

import click

from kebab.utils import logging
from kebab.utils.dataset.t_rex.t_rex_fragment_extractor import TRexFragmentExtractor


@click.option(
    "--t-rex-dir",
    type=pathlib.Path,
    default=pathlib.Path.home()
    / "OneDrive - Microsoft"
    / "Benchmark"
    / "Datasets"
    / "T-REx"
    / "Original Dataset"
    # / "single",
    # / "sample",
    / "full",
    help="Path to the T-REx data directory (will be *.json-globbed against).",
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "output" / "t_rex_fragments",
    help="Path to the output directory ('t_rex_entity_fragments.jsonl' will be created there).",
)
@click.command()
def main(
    t_rex_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    """Run T-REx fragment extraction steps."""
    logging.configure_logging()

    extractor = TRexFragmentExtractor(
        t_rex_dir=t_rex_dir,
        output_dir=output_dir,
    )

    extractor.run()


if __name__ == "__main__":
    main()
