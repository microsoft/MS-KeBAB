# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the Wikidata names extractor."""

from pathlib import Path

import pytest

from kebab.utils.dataset.wikidata.wikidata_names_dataset_builder import WikidataNamesDatasetBuilder


@pytest.fixture
def wikidata_hierarchy_input_file_path() -> Path:
    """Return the path to the input file with a subset Wikidata type hierarchy."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "wikidata_type_hierarchy.jsonl"


@pytest.fixture
def wikidata_simple_entities_input_file_path() -> Path:
    """Return the path to the input file with a subset of Wikidata simple entities."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "wikidata_simple_entities.jsonl"


def test_extract_names(
    wikidata_hierarchy_input_file_path: Path, wikidata_simple_entities_input_file_path: Path, tmp_path: Path
) -> None:
    """Test the extraction of names from Wikidata."""
    builder = WikidataNamesDatasetBuilder(
        wikidata_simple_entities_path=wikidata_simple_entities_input_file_path,
        wikidata_type_hierarchy_path=wikidata_hierarchy_input_file_path,
        output_dir=tmp_path,
        type_ids=["Q154954"],
    )

    builder.run()

    output_file_path = tmp_path / WikidataNamesDatasetBuilder.WIKIDATA_NAMES_OUTPUT_FILENAME

    assert output_file_path.exists()
    assert output_file_path.is_file()

    with open(output_file_path, encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 1
    assert lines[0].strip().startswith('{"id": "Q23"')
