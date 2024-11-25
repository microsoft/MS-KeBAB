# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the T-REx entity dataset creation steps."""

from __future__ import annotations

import pathlib

import click

from kebab.utils import utils
from kebab.utils.dataset.t_rex.t_rex_dataset_builder import TRexDatasetBuilder


@click.option(
    "--t-rex-fragments-path",
    type=pathlib.Path,
    default=pathlib.Path.home()
    / "OneDrive - Microsoft"
    / "Benchmark"
    / "Datasets"
    / "T-REx"
    / "Fragments"
    / "2024-06-27"
    # / "single"
    # / "sample"
    / "full"
    / "t_rex_entity_fragments.jsonl",
    help="Path to the T-REx fragments file.",
)
@click.option(
    "--wikidata-simple-entities-path",
    type=pathlib.Path,
    default=pathlib.Path.home()
    / "OneDrive - Microsoft"
    / "Benchmark"
    / "Datasets"
    / "Wikidata"
    / "Simple Entities"
    / "2024-06-27"
    / "wikidata_simple_entities.jsonl",
    help="The path to the extracted Wikidata simple entities.",
)
@click.option(
    "--wikidata-properties-path",
    type=pathlib.Path,
    default=pathlib.Path.home()
    / "OneDrive - Microsoft"
    / "Benchmark"
    / "Datasets"
    / "Wikidata"
    / "Properties"
    / "2024-06-27"
    / "wikidata_properties.json",
    help="Path to the Wikidata properties file.",
)
@click.option(
    "--wikidata-type-hierarchy-path",
    type=pathlib.Path,
    default=pathlib.Path.home()
    / "OneDrive - Microsoft"
    / "Benchmark"
    / "Datasets"
    / "Wikidata"
    / "Type Hierarchy"
    / "2024-06-27"
    / "wikidata_type_hierarchy.jsonl",
    help="The path to the Wikidata type hierarchy.",
)
@click.option(
    "--output-dir",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "output" / "t_rex_entities",
    help="Path to the output directory ('t_rex_entities_dataset.jsonl' will be created there).",
)
@click.command()
def main(
    t_rex_fragments_path: pathlib.Path,
    wikidata_simple_entities_path: pathlib.Path,
    wikidata_properties_path: pathlib.Path,
    wikidata_type_hierarchy_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    """Run T-REx entity dataset creation steps."""
    utils.configure_logging()

    builder = TRexDatasetBuilder(
        fragments_path=t_rex_fragments_path,
        wikidata_simple_entities_path=wikidata_simple_entities_path,
        wikidata_properties_path=wikidata_properties_path,
        wikidata_type_hierarchy_path=wikidata_type_hierarchy_path,
        output_dir=output_dir,
    )

    builder.run()


if __name__ == "__main__":
    main()
