# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Download and preprocess the Re-DocRED dataset.

Steps:
1) Optionally fetch Wikidata properties.
2) Download Re-DocRED dataset splits (train/dev/test).
3) Preprocess each split using the Re-DocRED dataset builder.
4) Check MD5 hashes for generated files.

Inputs/outputs are discovered under a single working directory (default: ./data).

Expected input locations under ``<working_dir>``:
- Wikidata properties: ``Wikidata/Properties/wikidata_properties.json``
- Re-DocRED splits: ``Re-DocRED/Original Dataset/{split}/{split}.json``
- Re-DocRED extractions: ``Re-DocRED/Extractions/{split}/``
- MD5 file: ``Re-DocRED/extraction.md5``
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import click
import requests
from kebab.utils import logging_helpers
from kebab.utils.dataset.re_docred.re_docred_dataset_builder import ReDocRedDatasetBuilder
from kebab.utils.dataset.wikidata.wikidata_simple_extractor import (
    WikidataSimpleExtractor,
)
from kebab.utils.io_helpers import md5sum, verify_md5_file


RE_DOCRED_SPLITS = ["train", "dev", "test"]
RE_DOCRED_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/tonytan48/Re-DocRED/refs/heads/main/data/{split}_revised.json"
)
MD5_LINE_PARTS = 2


def _exists(p: Path) -> bool:
    """Check if a path exists."""
    try:
        return p.exists()
    except OSError:
        return False


def _download_file(url: str, dest: Path) -> None:
    """Download a file from a URL to a destination path."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    # Write as text to avoid binary corruption for JSON files
    with open(dest, "w", encoding="utf-8") as f:
        f.write(resp.text)


def _create_md5_file(md5_file: Path, base_dir: Path) -> None:
    """Create an MD5 file from current extraction outputs."""
    extractions_dir = base_dir / "Extractions"
    md5_entries = []

    for split in RE_DOCRED_SPLITS:
        split_dir = extractions_dir / split
        if split_dir.exists():
            for file_path in split_dir.glob("*.jsonl"):
                relative_path = file_path.relative_to(extractions_dir)
                md5_hash = md5sum(file_path)
                md5_entries.append(f"{md5_hash}  {relative_path}")

    md5_entries.sort()  # Sort for consistent output
    with open(md5_file, "w", encoding="utf-8") as f:
        for entry in md5_entries:
            f.write(entry + "\n")


def _verify_md5_file(md5_file: Path, base_dir: Path, logger: logging.Logger) -> None:  # legacy wrapper for consistency
    verify_md5_file(md5_file, base_dir / "Extractions", logger)


@click.command()
@click.option(
    "--working-dir",
    type=Path,
    default=Path.cwd() / "data",
    help="Root data directory to read inputs and write outputs.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Regenerate everything, even if files exist.",
)
def main(
    working_dir: Path,
    force: bool,
) -> None:
    """Run the full Re-DocRED dataset download and preprocessing pipeline."""
    logging_helpers.configure_logging()
    logger = logging.getLogger(Path(__file__).stem)

    wikidata_output_dir = working_dir / "Wikidata" / "Properties"
    redocred_output_dir = working_dir / "Re-DocRED"
    wikidata_properties_file = "wikidata_properties.json"
    wikidata_properties_path = wikidata_output_dir / wikidata_properties_file
    md5_file = redocred_output_dir / "extraction.md5"

    # 1) Fetch Wikidata properties
    if force or not _exists(wikidata_properties_path):
        logger.info("Fetching Wikidata properties...")
        wikidata_output_dir.mkdir(parents=True, exist_ok=True)
        # Call the extractor class directly (properties only)
        extractor = WikidataSimpleExtractor(
            wikidata_json_dump_path=None,
            run_entity_extraction=False,
            run_property_fetch=True,
            output_dir=wikidata_output_dir,
        )
        extractor.run()
    else:
        logger.info(f"Using existing Wikidata properties at {wikidata_properties_path}")

    # 2) Download Re-DocRED dataset splits
    for split in RE_DOCRED_SPLITS:
        split_dir = redocred_output_dir / "Original Dataset" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_filename = split_dir / f"{split}.json"
        if force or not _exists(split_filename):
            logger.info(f"Downloading Re-DocRED {split} split...")
            url = RE_DOCRED_URL_TEMPLATE.format(split=split)
            _download_file(url, split_filename)
        else:
            logger.info(f"File {split_filename} already exists. Skipping download.")

    # 3) Preprocess the Re-DocRED dataset
    for split in RE_DOCRED_SPLITS:
        logger.info(f"Processing split: {split}")
        extractions_path = redocred_output_dir / "Extractions" / split
        if force or not _exists(extractions_path):
            if _exists(extractions_path):
                shutil.rmtree(extractions_path)
            extractions_path.mkdir(parents=True, exist_ok=True)
            builder = ReDocRedDatasetBuilder(
                re_docred_dir=redocred_output_dir / "Original Dataset" / split,
                wikidata_properties_path=wikidata_properties_path,
                output_dir=extractions_path,
            )
            builder.run()
        else:
            logger.info(f"Extraction directory for {split} already exists. Skipping preprocessing.")

    # 4) Check/create MD5 hashes
    if not _exists(md5_file):
        logger.warning(f"MD5 file {md5_file} not found. Creating it from current outputs.")
        _create_md5_file(md5_file, redocred_output_dir)
    else:
        _verify_md5_file(md5_file, redocred_output_dir, logger)


if __name__ == "__main__":
    main()
