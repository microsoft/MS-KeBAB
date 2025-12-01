# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Build the linking, clustering and generation datasets based on REBEL and (optionally) Wikidata.

Steps:
0) (Optional) Download and unpack the public REBEL dataset archive.
1) (On demand) Ensure Wikidata artifacts required for requested resolution are present; extract if missing.
2) Extract REBEL fragments.
3) Resolve Wikidata types / property names / property values as requested.
4) Filter degenerate fragments.
5) Sample pairs to build base linking/clustering datasets.
6) Split into train/dev/test datasets for all tasks.

Inputs are discovered under a single working directory (default: ./data).

Expected input locations under ``<working_dir>``:
- REBEL original dataset: ``REBEL/rebel_original_dataset``
- Wikidata properties: ``Wikidata/properties/wikidata_properties.json``
- Wikidata simple entities: ``Wikidata/simple_entities/wikidata_simple_entities.jsonl``
- Wikidata type hierarchy: ``Wikidata/type_hierarchy/wikidata_type_hierarchy.jsonl``

Resolution behaviour:
- "Resolve property names" (default: enabled) will use the properties file if present; otherwise it is extracted.
- "Resolve property values" (default: disabled) will use / extract simple entities as needed.
- "Resolve / attach types" (default: disabled) will use / extract a type hierarchy as needed.

Example usage (from repository root):
    # Public build
    uv run ./scripts/dataset/run_download_and_build_rebel_based_datasets.py

    # Public rebuild
    uv run ./scripts/dataset/run_download_and_build_rebel_based_datasets.py --no-download-rebel

    # Quick run from pair-sampling onwards to verify the logic
    uv run ./scripts/dataset/run_download_and_build_rebel_based_datasets.py --no-download-rebel --no-run-extract-fragments --no-resolve-property-names --no-resolve-property-values --no-run-filter --max-pair-count 100000

    # With type resolution enabled (internal use case)
    uv run ./scripts/dataset/run_download_and_build_rebel_based_datasets.py --resolve-types --no-verify-md5

    # From pair sampling onwards (expects existing prior outputs)
    uv run ./scripts/dataset/run_download_and_build_rebel_based_datasets.py --no-download-rebel --no-run-extract-fragments --no-resolve-property-names --no-resolve-property-values --no-run-filter

    # From splitting onwards (expects existing base datasets)
    uv run ./scripts/dataset/run_download_and_build_rebel_based_datasets.py --no-download-rebel --no-run-extract-fragments --no-resolve-property-names --no-resolve-property-values --no-run-filter --no-run-sample-pairs

Notes:
- Extraction of missing Wikidata artifacts requires a standard Wikidata JSON dump (``latest-all.json``) which can be
    downloaded from https://dumps.wikimedia.org/wikidatawiki/entities/ (unpacked size > 1 TB). Only artifacts required
    by the chosen resolution flags are extracted.
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
from kebab.utils.io_helpers import create_md5_file, verify_md5_file


REBEL_ZIP_URL = "https://huggingface.co/datasets/Babelscape/rebel-dataset/resolve/main/rebel_dataset.zip"
MD5_LINE_PARTS = 2  # kept for backward compatibility of existing md5 files


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


def _collect_rebel_output_files(rebel_root: Path) -> list[Path]:
    """Collect all output files that should be included in MD5 verification."""
    output_files = []
    fragments_root = rebel_root / "rebel_fragments"
    for stage in ["extracted", "resolved", "filtered"]:
        stage_dir = fragments_root / stage
        if stage_dir.exists():
            output_files.extend(list(stage_dir.glob("*.jsonl")))
    for task in ["linking", "clustering", "entity_generation", "fragment_generation"]:
        task_dir = rebel_root / task / "base"
        if task_dir.exists():
            output_files.extend(list(task_dir.glob("*.jsonl")))
    for task in [
        "linking",
        "clustering",
        "entity_generation",
        "fragment_generation",
    ]:
        task_dir = rebel_root / task
        if task_dir.exists():
            for split_dir in task_dir.iterdir():
                if split_dir.is_dir() and split_dir.name != "base":
                    output_files.extend(list(split_dir.glob("*.jsonl")))
    return output_files


def _create_md5_file(md5_file: Path, rebel_root: Path) -> None:  # wrapper to maintain script interface
    files = _collect_rebel_output_files(rebel_root)
    create_md5_file(md5_file, files, rebel_root)


def _verify_md5_file(md5_file: Path, rebel_root: Path, logger: logging.Logger) -> None:  # wrapper
    verify_md5_file(md5_file, rebel_root, logger)


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
    "--resolve-property-names/--no-resolve-property-names",
    default=True,
    help=(
        "Resolve Wikidata property names. Uses <working_dir>/Wikidata/properties/wikidata_properties.json if it exists; "
        "otherwise fetches it from Wikidata."
    ),
)
@click.option(
    "--resolve-property-values/--no-resolve-property-values",
    default=False,
    help=(
        "Resolve Wikidata property values. Uses <working_dir>/Wikidata/simple_entities/"
        "wikidata_simple_entities.jsonl if present; otherwise extracts it (requires latest-all.json)."
    ),
)
@click.option(
    "--resolve-types/--no-resolve-types",
    default=False,
    help=(
        "Attach and resolve Wikidata types. Uses <working_dir>/Wikidata/type_hierarchy/wikidata_type_hierarchy.jsonl "
        "if present; otherwise extracts it (requires latest-all.json). Enables type-based filtering."
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
        "Whether to run the step: extract REBEL fragments. If disabled, expects "
        "existing outputs under <working_dir>/REBEL/rebel_fragments/extracted."
    ),
)
@click.option(
    "--run-filter/--no-run-filter",
    default=True,
    help=(
        "Whether to run the step: filter degenerate fragments. If disabled, expects "
        "existing outputs under <working_dir>/REBEL/rebel_fragments/filtered or will use prior step."
    ),
)
@click.option(
    "--run-sample-pairs/--no-run-sample-pairs",
    default=True,
    help=(
        "Whether to run the step: sample base pairs for linking/clustering. If disabled, "
        "expects existing base datasets under <working_dir>/REBEL/<dataset>/base."
    ),
)
@click.option(
    "--run-split/--no-run-split",
    default=True,
    help=("Whether to run the step: produce dataset splits."),
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help=("Random seed to make sampling deterministic across runs."),
)
@click.option(
    "--verify-md5/--no-verify-md5",
    default=True,
    help=("Whether to verify/create MD5 checksums for all output files."),
)
@click.option(
    "--max-pair-count",
    type=int,
    default=20_000_000,
    show_default=True,
    help=("Maximum number of pairs to sample when building base datasets."),
)
def main(
    working_dir: Path,
    download_rebel: bool,
    force_download_rebel: bool,
    resolve_property_names: bool,
    resolve_property_values: bool,
    resolve_types: bool,
    wikidata_json_dump_path: Path | None,
    seed: int,
    max_pair_count: int,
    run_extract_fragments: bool,
    run_filter: bool,
    run_sample_pairs: bool,
    run_split: bool,
    verify_md5: bool,
) -> None:
    """Run the full REBEL dataset builder pipeline with sensible defaults."""
    logging_helpers.configure_logging()
    logger = logging.getLogger(Path(__file__).stem)

    logger.info(
        "Starting REBEL dataset build: "
        f"working_dir={working_dir}, download_rebel={download_rebel}, force_download_rebel={force_download_rebel}, "
        f"resolve_property_names={resolve_property_names}, resolve_property_values={resolve_property_values}, resolve_types={resolve_types}, "
        f"run_extract_fragments={run_extract_fragments}, run_filter={run_filter}, "
        f"run_sample_pairs={run_sample_pairs}, run_split={run_split}, verify_md5={verify_md5}, seed={seed}"
    )

    # Layout roots
    rebel_root = working_dir / "REBEL"
    wikidata_root = working_dir / "Wikidata"

    # The truth MD5 file is located under the repository root: <repo_root>/data/REBEL/rebel_outputs.md5.
    repo_root = Path(__file__).resolve().parents[2]
    canonical_md5_file = repo_root / "data" / "REBEL" / "rebel_outputs.md5"

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

    # 1) Ensure Wikidata artifacts needed for requested resolution are present (extract on demand)
    properties_path = wd_properties_out / WikidataSimpleExtractor.WIKIDATA_PROPERTIES_FILENAME
    simple_entities_path = wd_simple_entities_out / WikidataSimpleExtractor.WIKIDATA_SIMPLE_ENTITIES_FILENAME
    type_hierarchy_path = wd_type_hierarchy_out / WikidataTypeHierarchyExtractor.WIKIDATA_HIERARCHY_FILENAME

    if resolve_property_names:
        if _exists(properties_path):
            logger.info(f"Using existing Wikidata properties file: {properties_path}")
        else:
            logger.info(
                f"Property name resolution requested; properties file missing -> fetching from Wikidata -> out={wd_properties_out}"
            )
            property_extractor = WikidataSimpleExtractor(
                run_entity_extraction=False,
                run_property_fetch=True,
                output_dir=wd_properties_out,
            )
            property_extractor.run()
    else:
        logger.info("Property name resolution disabled (will not extract or use properties file)")

    if resolve_property_values:
        if _exists(simple_entities_path):
            logger.info(f"Using existing Wikidata simple entities file: {simple_entities_path}")
        else:
            logger.info(
                f"Property value resolution requested; simple entities file missing -> extracting from dump={dump_path} -> out={wd_simple_entities_out}"
            )
            entity_extractor = WikidataSimpleExtractor(
                wikidata_json_dump_path=dump_path,
                run_entity_extraction=True,
                run_property_fetch=False,
                output_dir=wd_simple_entities_out,
            )
            entity_extractor.run()
    else:
        logger.info("Property value resolution disabled (will not extract or use simple entities file)")

    if resolve_types:
        if _exists(type_hierarchy_path):
            logger.info(f"Using existing Wikidata type hierarchy file: {type_hierarchy_path}")
        else:
            logger.info(
                f"Type resolution requested; hierarchy file missing -> extracting from dump={dump_path} -> out={wd_type_hierarchy_out}"
            )
            hierarchy_extractor = WikidataTypeHierarchyExtractor(
                wikidata_json_dump_path=dump_path,
                output_dir=wd_type_hierarchy_out,
            )
            hierarchy_extractor.run()
    else:
        logger.info("Type resolution disabled (will not extract or use type hierarchy file)")

    # Validate that all requested resolution artifacts actually exist.
    if resolve_property_names and not _exists(properties_path):
        raise RuntimeError(
            "Requested property name resolution but properties file not found: "
            f"{properties_path}. Provide a Wikidata dump or disable --no-resolve-property-names."
        )

    if resolve_property_values and not _exists(simple_entities_path):
        raise RuntimeError(
            "Requested property value resolution but simple entities file not found: "
            f"{simple_entities_path}. Provide a Wikidata dump or disable --no-resolve-property-values."
        )

    if resolve_types and not _exists(type_hierarchy_path):
        raise RuntimeError(
            "Requested type resolution but type hierarchy file not found: "
            f"{type_hierarchy_path}. Provide a Wikidata dump or disable --no-resolve-types."
        )

    logger.info(
        "Resolution inputs: "
        f"property_names={resolve_property_names} (path={properties_path}), "
        f"property_values={resolve_property_values} (path={simple_entities_path}), "
        f"types={resolve_types} (path={type_hierarchy_path})"
    )

    # 2) Extract fragments from REBEL dataset
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

    # 3) Resolve Wikidata names/types/values where possible given available inputs
    if resolve_property_names or resolve_property_values or resolve_types:
        logger.info(
            "Resolving Wikidata fields ("
            f"attach_types={resolve_types}, resolve_types={resolve_types}, "
            f"property_names={resolve_property_names}, property_values={resolve_property_values}, query_api={True}) -> {rebel_fragments_resolved}"
        )
        resolver = WikidataResolver(
            entities_path=current_fragments_path,
            wikidata_simple_entities_path=simple_entities_path,
            wikidata_properties_path=properties_path,
            wikidata_type_hierarchy_path=type_hierarchy_path,
            attach_types=resolve_types,
            resolve_attached_types=resolve_types,
            resolve_property_names=resolve_property_names,
            resolve_property_values=resolve_property_values,
            query_api=True,
            output_dir=rebel_fragments_resolved,
        )
        resolver.run()
        current_fragments_path = rebel_fragments_resolved / "rebel_entity_fragments.jsonl"
    else:
        logger.info("Skipping Wikidata resolution (flag disabled)")
        # If we skip resolution but resolved outputs already exist, advance to them
        # so that later steps (e.g., filtering/sampling) operate on the most
        # processed data.
        resolved_path = rebel_fragments_resolved / "rebel_entity_fragments.jsonl"
        if _exists(resolved_path):
            current_fragments_path = resolved_path

    # 4) Filter out degenerate fragments
    if run_filter:
        logger.info(
            f"Filtering degenerate fragments (drop_without_type={bool(resolve_types)}) -> {rebel_fragments_filtered}"
        )
        fragment_filter = RebelFragmentFilter(
            fragments_path=current_fragments_path,
            output_dir=rebel_fragments_filtered,
            drop_fragments_without_type=bool(resolve_types),
        )
        fragment_filter.run()
        current_fragments_path = rebel_fragments_filtered / "rebel_entity_fragments.jsonl"
    else:
        logger.info("Skipping fragment filtering (flag disabled)")
        # If we skip filtering but filtered outputs already exist, advance to them
        # so that subsequent steps (e.g., sampling) operate on the most processed data.
        filtered_path = rebel_fragments_filtered / "rebel_entity_fragments.jsonl"
        if _exists(filtered_path):
            current_fragments_path = filtered_path

    # 5) Build base datasets (linking/clustering)
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

    # 6) Produce splits for all tasks
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

    # MD5 verification/creation
    if verify_md5:
        if not _exists(canonical_md5_file):
            logger.warning(f"Canonical MD5 file not found at {canonical_md5_file}. Creating it from current outputs.")
            canonical_md5_file.parent.mkdir(parents=True, exist_ok=True)
            _create_md5_file(canonical_md5_file, rebel_root)
        else:
            logger.info(f"Verifying outputs in {rebel_root} against canonical MD5 file {canonical_md5_file}")
            _verify_md5_file(canonical_md5_file, rebel_root, logger)


if __name__ == "__main__":
    main()
