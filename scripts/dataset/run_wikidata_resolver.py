# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script to run the Wikidata Resolver to substitute property names and actual property values into entities."""

from __future__ import annotations

from pathlib import Path

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.wikidata.wikidata_resolver import WikidataResolver


@click.option(
    "--entities-path",
    type=Path,
    default=Path.cwd() / "data" / "REBEL" / "Fragments" / "Surface Forms" / "rebel_entity_fragments.jsonl",
    help="Path to the input entities file.",
)
@click.option(
    "--wikidata-simple-entities-path",
    type=Path,
    default=Path.cwd() / "data" / "Wikidata" / "Simple Entities" / "2025-01-30" / "wikidata_simple_entities.jsonl",
    help="The path to the extracted Wikidata simple entities.",
)
@click.option(
    "--wikidata-properties-path",
    type=Path,
    default=Path.cwd() / "data" / "Wikidata" / "Properties" / "2025-01-30" / "wikidata_properties.json",
    help="Path to the Wikidata properties file.",
)
@click.option(
    "--wikidata-type-hierarchy-path",
    type=Path,
    default=Path.cwd() / "data" / "Wikidata" / "Type Hierarchy" / "2025-01-30" / "wikidata_type_hierarchy.jsonl",
    help="The path to the Wikidata type hierarchy.",
)
@click.option(
    "--attach-types/--no-attach-types",
    default=True,
    help="Whether to attach actual Wikidata types to the entities.",
)
@click.option(
    "--resolve-types/--no-resolve-types",
    default=True,
    help="Whether to resolve Wikidata types for the attached types.",
)
@click.option(
    "--resolve-property-names/--no-resolve-property-names",
    default=True,
    help="Whether to resolve property names.",
)
@click.option(
    "--resolve-property-values/--no-resolve-property-values",
    default=True,
    help="Whether to resolve property values.",
)
@click.option(
    "--query-api/--no-query-api",
    default=True,
    help="Whether to query the Wikidata API to resolve property values when not found in the dump.",
)
@click.option(
    "--output-dir",
    type=Path,
    default=Path.cwd() / "output" / "resolved_entities",
    help="Path to the output directory (the output jsonl file will be created there).",
)
@click.command()
def main(
    entities_path: Path,
    wikidata_simple_entities_path: Path,
    wikidata_properties_path: Path,
    wikidata_type_hierarchy_path: Path,
    attach_types: bool,
    resolve_types: bool,
    resolve_property_names: bool,
    resolve_property_values: bool,
    query_api: bool,
    output_dir: Path,
) -> None:
    """Run the Wikidata Resolver to substitute property names and actual property values into entities."""
    logging_helpers.configure_logging()

    resolver = WikidataResolver(
        entities_path=entities_path,
        wikidata_simple_entities_path=wikidata_simple_entities_path,
        wikidata_properties_path=wikidata_properties_path,
        wikidata_type_hierarchy_path=wikidata_type_hierarchy_path,
        attach_types=attach_types,
        resolve_attached_types=resolve_types,
        resolve_property_names=resolve_property_names,
        resolve_property_values=resolve_property_values,
        query_api=query_api,
        output_dir=output_dir,
    )

    resolver.run()


if __name__ == "__main__":
    main()
