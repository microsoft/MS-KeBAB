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

import hashlib
import logging
import shutil
from pathlib import Path

import click
import requests
from kebab.utils import logging_helpers
from kebab.utils.dataset.re_docred.re_docred_dataset_builder import (
    ReDocRedDatasetBuilder,
)
from kebab.utils.dataset.wikidata.wikidata_simple_extractor import (
    WikidataSimpleExtractor,
)


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


def _md5sum(path: Path) -> str:
    """Compute the MD5 hash of a file, ignoring CR characters.

    Note: MD5 is kept to match reference hashes in extraction.md5.
    """
    hash_md5 = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(4096), b""):
            hash_md5.update(buf.replace(b"\r", b""))
    return hash_md5.hexdigest()


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

    # 4) Check MD5 hashes
    if not _exists(md5_file):
        logger.warning(f"MD5 file {md5_file} not found. Skipping MD5 check.")
        return
    with open(md5_file, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != MD5_LINE_PARTS:
                logger.warning(f"Malformed MD5 line: {line}")
                continue
            expected_md5, file_rel = parts
            file_path = redocred_output_dir / "Extractions" / file_rel
            if not _exists(file_path):
                logger.error(f"File {file_path} not found for MD5 check.")
                raise FileNotFoundError(file_path)
            actual_md5 = _md5sum(file_path)
            if actual_md5 != expected_md5:
                logger.error(f"MD5 mismatch for {file_path}: expected {expected_md5}, got {actual_md5}")
                raise ValueError(f"MD5 mismatch for {file_path}")
    logger.info("MD5 check passed. All files are correctly generated.")


if __name__ == "__main__":
    main()
