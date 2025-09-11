# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the REBEL fragment filter to drop degenerate fragments/values."""

from __future__ import annotations

from pathlib import Path

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.rebel.rebel_fragment_filter import RebelFragmentFilter


@click.option(
    "--rebel-fragments-path",
    type=Path,
    default=Path.cwd() / "data" / "REBEL" / "Fragments" / "Surface Forms" / "Resolved" / "rebel_entity_fragments.jsonl",
    help="Path to the input REBEL fragments file (JSONL).",
)
@click.option(
    "--output-dir",
    type=Path,
    default=Path.cwd() / "output",
    help="Path to the output directory (filtered fragments JSONL will be created there).",
)
@click.command()
def main(
    rebel_fragments_path: Path,
    output_dir: Path,
):
    """Run the REBEL fragment filter."""
    logging_helpers.configure_logging()

    runner = RebelFragmentFilter(
        fragments_path=rebel_fragments_path,
        output_dir=output_dir,
    )

    runner.run()


if __name__ == "__main__":
    main()
