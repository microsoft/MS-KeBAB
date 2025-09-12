# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Build the linking, clustering and generation datasets based on REBEL and Wikidata.

Steps:
1) Optionally extract Wikidata properties, simple entities, and type hierarchy.
2) Extract REBEL fragments.
3) Resolve Wikidata types/props/values where possible.
4) Filter degenerate fragments.
5) Sample pairs to build base linking/clustering datasets.
6) Split into train/dev/test datasets for all tasks.

Inputs are discovered under a single working directory (default: ./data).

Expected input locations under ``<working_dir>``:
- REBEL original dataset: ``REBEL/rebel_original_dataset/full``
- Wikidata properties (optional): ``Wikidata/properties/wikidata_properties.json``
- Wikidata simple entities (optional): ``Wikidata/simple_entities/wikidata_simple_entities.jsonl``
- Wikidata type hierarchy (optional): ``Wikidata/type_hierarchy/wikidata_type_hierarchy.jsonl``

Notes:
- By default, Wikidata artifacts are not extracted. Enabling extraction requires a standard Wikidata JSON dump
  (``latest-all.json``). You can download it from  https://dumps.wikimedia.org/wikidatawiki/entities/
  (unpacked size is > 1 TB).
- If Wikidata inputs are not available (and not extracted), some functionality is skipped: certain degenerate-fragment
  filtering will not be applied, and IDs (property IDs, entity IDs, type IDs) will not be resolved to human-readable values.
"""

from __future__ import annotations

from pathlib import Path

import click
from kebab.utils import logging_helpers
from kebab.utils.dataset.rebel.rebel_fragment_extractor import RebelFragmentExtractor
from kebab.utils.dataset.rebel.rebel_fragment_filter import RebelFragmentFilter
from kebab.utils.dataset.rebel.rebel_pair_sampler import (
    MergeDistributionMode,
    RebelPairSampler,
)
from kebab.utils.dataset.rebel.rebel_split_sampler import RebelSplitSampler, default_splits
from kebab.utils.dataset.wikidata.wikidata_resolver import WikidataResolver
from kebab.utils.dataset.wikidata.wikidata_simple_extractor import WikidataSimpleExtractor
from kebab.utils.dataset.wikidata.wikidata_type_hierarchy_extractor import (
    WikidataTypeHierarchyExtractor,
)


def _exists(p: Path) -> bool:
    """Check if a path exists."""
    try:
        return p.exists()
    except OSError:
        return False


@click.command()
@click.option(
    "--working-dir",
    type=Path,
    default=Path.cwd() / "data",
    help="Root data directory to read inputs and write outputs.",
)
@click.option(
    "--extract-properties/--no-extract-properties",
    default=False,
    help=(
        "Extract Wikidata properties into <working_dir>/Wikidata/properties. "
        "If not extracting, we'll look for 'wikidata_properties.json' there; "
        "if missing, property names will not be resolved from IDs."
    ),
)
@click.option(
    "--extract-simple-entities/--no-extract-simple-entities",
    default=False,
    help=(
        "Extract Wikidata simple entities into <working_dir>/Wikidata/simple_entities. "
        "If not extracting, we'll look for 'wikidata_simple_entities.jsonl' there; "
        "if missing, property values will not be resolved from IDs."
    ),
)
@click.option(
    "--extract-type-hierarchy/--no-extract-type-hierarchy",
    default=False,
    help=(
        "Extract Wikidata type hierarchy into <working_dir>/Wikidata/type_hierarchy. "
        "If not extracting, we'll look for 'wikidata_type_hierarchy.jsonl' there; "
        "if missing, types will not be attached/resolved and type-based filtering "
        "of degenerate fragments will not be applied."
    ),
)
@click.option(
    "--wikidata-json-dump-path",
    type=Path,
    default=None,
    help=(
        "Path to Wikidata latest-all.json dump (required when any extraction flag is enabled). "
        "Defaults to <working-dir>/latest-all.json if not provided. Download from "
        "https://dumps.wikimedia.org/wikidatawiki/entities/."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help=("Random seed to make sampling deterministic across runs."),
)
def main(
    working_dir: Path,
    extract_properties: bool,
    extract_simple_entities: bool,
    extract_type_hierarchy: bool,
    wikidata_json_dump_path: Path | None,
    seed: int,
) -> None:
    """Run the full REBEL dataset builder pipeline with sensible defaults."""
    logging_helpers.configure_logging()

    # Layout roots
    rebel_root = working_dir / "REBEL"
    wikidata_root = working_dir / "Wikidata"

    # Wikidata I/O paths
    dump_path = wikidata_json_dump_path or (working_dir / "latest-all.json")
    wd_properties_out = wikidata_root / "properties"
    wd_simple_entities_out = wikidata_root / "simple_entities"
    wd_type_hierarchy_out = wikidata_root / "type_hierarchy"

    # REBEL I/O paths
    rebel_original_dir = rebel_root / "rebel_original_dataset" / "full"
    rebel_fragments_root = rebel_root / "rebel_fragments"
    rebel_fragments_extracted = rebel_fragments_root / "extracted"
    rebel_fragments_resolved = rebel_fragments_root / "resolved"
    rebel_fragments_filtered = rebel_fragments_root / "filtered"

    # 1) Optional Wikidata extractions
    if extract_properties or extract_simple_entities:
        extractor = WikidataSimpleExtractor(
            wikidata_json_dump_path=dump_path,
            run_entity_extraction=extract_simple_entities,
            run_property_fetch=extract_properties,
            output_dir=wikidata_root,
        )
        extractor.run()

    if extract_type_hierarchy:
        hierarchy_extractor = WikidataTypeHierarchyExtractor(
            wikidata_json_dump_path=dump_path,
            output_dir=wd_type_hierarchy_out,
        )
        hierarchy_extractor.run()

    # Check availability of optional Wikidata artifacts (used for resolver),
    # regardless of whether we produced them in this run.
    properties_path = wd_properties_out / "wikidata_properties.json"
    simple_entities_path = wd_simple_entities_out / "wikidata_simple_entities.jsonl"
    type_hierarchy_path = wd_type_hierarchy_out / "wikidata_type_hierarchy.jsonl"

    properties_available = _exists(properties_path)
    simple_entities_available = _exists(simple_entities_path)
    type_hierarchy_available = _exists(type_hierarchy_path)

    # 2) Extract fragments from REBEL dataset
    fragment_extractor = RebelFragmentExtractor(
        rebel_dir=rebel_original_dir,
        output_dir=rebel_fragments_extracted,
        extract_surface_forms=True,
    )
    fragment_extractor.run()

    # 3) Resolve Wikidata names/types/values where possible given available inputs
    resolver = WikidataResolver(
        entities_path=rebel_fragments_extracted / "rebel_entity_fragments.jsonl",
        wikidata_simple_entities_path=simple_entities_path,
        wikidata_properties_path=properties_path,
        wikidata_type_hierarchy_path=type_hierarchy_path,
        attach_types=type_hierarchy_available,
        resolve_attached_types=type_hierarchy_available,
        resolve_property_names=properties_available,
        resolve_property_values=simple_entities_available,
        query_api=True,
        output_dir=rebel_fragments_resolved,
    )
    resolver.run()

    # 4) Filter out degenerate fragments
    fragment_filter = RebelFragmentFilter(
        fragments_path=rebel_fragments_resolved / "rebel_entity_fragments.jsonl",
        output_dir=rebel_fragments_filtered,
        drop_fragments_without_type=bool(type_hierarchy_available),
    )
    fragment_filter.run()

    # 5) Build base datasets (linking/clustering)
    pair_sampler = RebelPairSampler(
        fragments_path=rebel_fragments_filtered / "rebel_entity_fragments.jsonl",
        output_dir=rebel_root,
        max_count=100_000,
        max_merge_fragments=10,
        merge_distribution=MergeDistributionMode.ZIPF,
        deduplicate_values=True,
        seed=seed,
    )
    pair_sampler.run()

    # 6) Produce splits for all tasks
    split_sampler = RebelSplitSampler(
        input_dir=rebel_root,
        output_dir=rebel_root,
        splits=default_splits(seed=seed),
        type_filter=None,
    )
    split_sampler.run()


if __name__ == "__main__":
    main()
