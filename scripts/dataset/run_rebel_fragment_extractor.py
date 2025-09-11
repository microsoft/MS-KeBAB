# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the REBEL fragment extraction steps."""

from __future__ import annotations

from pathlib import Path

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.rebel.rebel_fragment_extractor import RebelFragmentExtractor


@click.option(
    "--rebel-dir",
    type=Path,
    default=Path.cwd() / "data" / "REBEL" / "Original Dataset" / "full",
    help="Path to the REBEL data directory (will be *json/*.jsonl-globbed against).",
)
@click.option(
    "--output-dir",
    type=Path,
    default=Path.cwd() / "output",
    help="Path to the output directory ('rebel_entity_fragments.jsonl' will be created there).",
)
@click.option(
    "--extract-surface-forms",
    is_flag=True,
    default=True,
    help="Extract surface forms instead of URIs for property values.",
)
@click.command()
def main(
    rebel_dir: Path,
    output_dir: Path,
    extract_surface_forms: bool,
) -> None:
    """Run REBEL fragment extraction steps."""
    logging_helpers.configure_logging()

    extractor = RebelFragmentExtractor(
        rebel_dir=rebel_dir,
        output_dir=output_dir,
        extract_surface_forms=extract_surface_forms,
    )

    extractor.run()


if __name__ == "__main__":
    main()
