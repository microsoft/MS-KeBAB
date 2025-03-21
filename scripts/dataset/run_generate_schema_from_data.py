from collections.abc import Iterable
from pathlib import Path

import click
from kebab.utils import logging_helpers
from kebab.utils.io_helpers import generate_draft_property_schema_from_data


@click.option(
    "--path",
    "-p",
    type=Path,
    multiple=True,
    help="Path to the directory containing entities.jsonl files.",
)
@click.option(
    "--output",
    "-o",
    type=Path,
    default=Path.cwd() / "property_schema.generated.json",
    help="Output filename for the generated property schema.",
)
@click.command()
def main(path: Iterable[Path], output: Path) -> None:
    """Generate a draft property schema from data."""
    logging_helpers.configure_logging()
    filenames = [p / "entities.jsonl" for p in path]
    generate_draft_property_schema_from_data(filenames, output)


if __name__ == "__main__":
    main()
