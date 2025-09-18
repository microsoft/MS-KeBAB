# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Build the linking, clustering and generation datasets based on REBEL and Wikidata.

Steps:
0) (Optional) Download and unpack the public REBEL dataset archive.
1) (Optional) Extract Wikidata properties, simple entities, and type hierarchy.
2) Extract REBEL fragments.
3) Resolve Wikidata types/props/values where possible.
4) Filter degenerate fragments.
5) Sample pairs to build base linking/clustering datasets.
6) Split into train/dev/test datasets for all tasks.

Inputs are discovered under a single working directory (default: ./data).

Expected input locations under ``<working_dir>``:
- REBEL original dataset: ``REBEL/rebel_original_dataset``
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

import logging
import tempfile
import urllib.request
import zipfile
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


REBEL_ZIP_URL = "https://huggingface.co/datasets/Babelscape/rebel-dataset/resolve/main/rebel_dataset.zip"


def _exists(p: Path) -> bool:
    """Check if a path exists."""
    try:
        return p.exists()
    except OSError:
        return False


def _dir_has_files(dir_path: Path) -> bool:
    """Return True if directory exists and contains at least one file (non-recursive)."""
    if not dir_path.is_dir():
        return False
    return any(dir_path.iterdir())


def _download_and_unpack_rebel(logger: logging.Logger, dest_dir: Path, force: bool) -> None:
    """Download and unzip the REBEL dataset archive into the destination directory."""
    if _dir_has_files(dest_dir) and not force:
        logger.info(
            f"REBEL original dataset already present at {dest_dir} (skip download; use --force-download-rebel to re-download)"
        )
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading REBEL dataset archive from {REBEL_ZIP_URL}")
    try:
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_zip = Path(tmpd) / "rebel_dataset.zip"
            urllib.request.urlretrieve(REBEL_ZIP_URL, tmp_zip)  # noqa: S310 (trusted constant URL)
            logger.info(f"Download complete ({tmp_zip.stat().st_size} bytes); extracting -> {dest_dir}")
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                zf.extractall(dest_dir)
    except Exception:
        logger.exception("Failed to download or extract REBEL dataset")
        raise
    else:
        logger.info(f"REBEL dataset extracted successfully -> {dest_dir}")


@click.command()
@click.option(
    "--working-dir",
    type=Path,
    default=Path.cwd() / "data",
    help="Root data directory to read inputs and write outputs.",
)
@click.option(
    "--download-rebel/--no-download-rebel",
    default=True,
    help=(
        "Download and unpack the public REBEL dataset archive if not already present under "
        "<working_dir>/REBEL/rebel_original_dataset."
    ),
)
@click.option(
    "--force-download-rebel/--no-force-download-rebel",
    default=False,
    help="Force re-download of REBEL dataset even if target directory is non-empty.",
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
    "--run-extract-fragments/--no-run-extract-fragments",
    default=True,
    help=(
        "Whether to run step 2: extract REBEL fragments. If disabled, expects "
        "existing outputs under <working_dir>/REBEL/rebel_fragments/extracted."
    ),
)
@click.option(
    "--run-resolve/--no-run-resolve",
    default=True,
    help=(
        "Whether to run step 3: resolve Wikidata names/types/values. If disabled, expects "
        "existing outputs under <working_dir>/REBEL/rebel_fragments/resolved or will use extracted."
    ),
)
@click.option(
    "--run-filter/--no-run-filter",
    default=True,
    help=(
        "Whether to run step 4: filter degenerate fragments. If disabled, expects "
        "existing outputs under <working_dir>/REBEL/rebel_fragments/filtered or will use prior step."
    ),
)
@click.option(
    "--run-sample-pairs/--no-run-sample-pairs",
    default=True,
    help=(
        "Whether to run step 5: sample base pairs for linking/clustering. If disabled, "
        "expects existing base datasets under <working_dir>/REBEL/<dataset>/base."
    ),
)
@click.option(
    "--run-split/--no-run-split",
    default=True,
    help=("Whether to run step 6: produce dataset splits."),
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
    download_rebel: bool,
    force_download_rebel: bool,
    extract_properties: bool,
    extract_simple_entities: bool,
    extract_type_hierarchy: bool,
    wikidata_json_dump_path: Path | None,
    seed: int,
    run_extract_fragments: bool,
    run_resolve: bool,
    run_filter: bool,
    run_sample_pairs: bool,
    run_split: bool,
) -> None:
    """Run the full REBEL dataset builder pipeline with sensible defaults."""
    logging_helpers.configure_logging()
    logger = logging.getLogger(Path(__file__).stem)

    logger.info(
        f"Starting REBEL dataset build: "
        f"working_dir={working_dir}, download_rebel={download_rebel}, force_download_rebel={force_download_rebel}, "
        f"extract_properties={extract_properties}, extract_simple_entities={extract_simple_entities}, extract_type_hierarchy={extract_type_hierarchy}, "
        f"run_extract_fragments={run_extract_fragments}, run_resolve={run_resolve}, run_filter={run_filter}, "
        f"run_sample_pairs={run_sample_pairs}, run_split={run_split}, seed={seed}"
    )

    # Layout roots
    rebel_root = working_dir / "REBEL"
    wikidata_root = working_dir / "Wikidata"

    # Wikidata I/O paths
    dump_path = wikidata_json_dump_path or (working_dir / "latest-all.json")
    wd_properties_out = wikidata_root / "properties"
    wd_simple_entities_out = wikidata_root / "simple_entities"
    wd_type_hierarchy_out = wikidata_root / "type_hierarchy"

    # REBEL I/O paths
    rebel_original_dir = rebel_root / "rebel_original_dataset"

    # 0) Download REBEL dataset
    if download_rebel:
        _download_and_unpack_rebel(logger, rebel_original_dir, force_download_rebel)
    else:
        logger.info("Skipping REBEL dataset download (flag disabled)")
    rebel_fragments_root = rebel_root / "rebel_fragments"
    rebel_fragments_extracted = rebel_fragments_root / "extracted"
    rebel_fragments_resolved = rebel_fragments_root / "resolved"
    rebel_fragments_filtered = rebel_fragments_root / "filtered"

    # 1) Wikidata extractions (optional)
    if extract_properties or extract_simple_entities:
        logger.info(
            f"Running Wikidata simple/property extraction (entities={extract_simple_entities}, properties={extract_properties}) "
            f"from dump={dump_path} -> out={wikidata_root}"
        )
        extractor = WikidataSimpleExtractor(
            wikidata_json_dump_path=dump_path,
            run_entity_extraction=extract_simple_entities,
            run_property_fetch=extract_properties,
            output_dir=wikidata_root,
        )
        extractor.run()
    else:
        logger.info("Skipping Wikidata simple/property extraction (flags disabled)")

    if extract_type_hierarchy:
        logger.info(f"Running Wikidata type hierarchy extraction from dump={dump_path} -> out={wd_type_hierarchy_out}")
        hierarchy_extractor = WikidataTypeHierarchyExtractor(
            wikidata_json_dump_path=dump_path,
            output_dir=wd_type_hierarchy_out,
        )
        hierarchy_extractor.run()
    else:
        logger.info("Skipping Wikidata type hierarchy extraction (flag disabled)")

    # Check availability of optional Wikidata artifacts (used for resolver),
    # regardless of whether we produced them in this run.
    properties_path = wd_properties_out / "wikidata_properties.json"
    simple_entities_path = wd_simple_entities_out / "wikidata_simple_entities.jsonl"
    type_hierarchy_path = wd_type_hierarchy_out / "wikidata_type_hierarchy.jsonl"

    properties_available = _exists(properties_path)
    simple_entities_available = _exists(simple_entities_path)
    type_hierarchy_available = _exists(type_hierarchy_path)

    logger.info(
        f"Optional inputs availability: properties={properties_available} ({properties_path}), "
        f"simple_entities={simple_entities_available} ({simple_entities_path}), "
        f"type_hierarchy={type_hierarchy_available} ({type_hierarchy_path})"
    )

    # 2) Extract fragments from REBEL dataset (optional)
    if run_extract_fragments:
        logger.info(
            f"Extracting REBEL fragments from {rebel_original_dir} -> {rebel_fragments_extracted} (surface_forms={True})"
        )
        fragment_extractor = RebelFragmentExtractor(
            rebel_dir=rebel_original_dir,
            output_dir=rebel_fragments_extracted,
            extract_surface_forms=True,
        )
        fragment_extractor.run()
    else:
        logger.info("Skipping REBEL fragment extraction (flag disabled)")

    # Determine input path for subsequent steps depending on what was run
    current_fragments_path = rebel_fragments_extracted / "rebel_entity_fragments.jsonl"

    # 3) Resolve Wikidata names/types/values where possible given available inputs (optional)
    if run_resolve:
        logger.info(
            f"Resolving Wikidata fields (types attach={type_hierarchy_available} resolve={type_hierarchy_available}, "
            f"prop_names={properties_available}, prop_values={simple_entities_available}, query_api={True}) -> {rebel_fragments_resolved}"
        )
        resolver = WikidataResolver(
            entities_path=current_fragments_path,
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
        current_fragments_path = rebel_fragments_resolved / "rebel_entity_fragments.jsonl"
    else:
        logger.info("Skipping Wikidata resolution (flag disabled)")

    # 4) Filter out degenerate fragments (optional)
    if run_filter:
        logger.info(
            f"Filtering degenerate fragments (drop_without_type={bool(type_hierarchy_available)}) -> {rebel_fragments_filtered}"
        )
        fragment_filter = RebelFragmentFilter(
            fragments_path=current_fragments_path,
            output_dir=rebel_fragments_filtered,
            drop_fragments_without_type=bool(type_hierarchy_available),
        )
        fragment_filter.run()
        current_fragments_path = rebel_fragments_filtered / "rebel_entity_fragments.jsonl"
    else:
        logger.info("Skipping fragment filtering (flag disabled)")

    # 5) Build base datasets (linking/clustering) (optional)
    max_pair_count = 10_000_000
    if run_sample_pairs:
        logger.info(
            f"Sampling base pairs (max_count={max_pair_count}, max_merge_fragments={10}, merge_distribution={MergeDistributionMode.ZIPF}, "
            f"dedup_values={True}, seed={seed}) -> {rebel_root}"
        )
        pair_sampler = RebelPairSampler(
            fragments_path=current_fragments_path,
            output_dir=rebel_root,
            max_count=max_pair_count,
            max_merge_fragments=10,
            merge_distribution=MergeDistributionMode.ZIPF,
            deduplicate_values=True,
            seed=seed,
        )
        pair_sampler.run()
    else:
        logger.info("Skipping pair sampling (flag disabled)")

    # 6) Produce splits for all tasks (optional)
    if run_split:
        logger.info(f"Producing dataset splits -> {rebel_root}")
        split_sampler = RebelSplitSampler(
            input_dir=rebel_root,
            output_dir=rebel_root,
            splits=default_splits(seed=seed),
            type_filter=None,
        )
        split_sampler.run()
    else:
        logger.info("Skipping dataset split generation (flag disabled)")

    logger.info(f"REBEL dataset build completed successfully: out={rebel_root}")


if __name__ == "__main__":
    main()
