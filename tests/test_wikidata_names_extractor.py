# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the Wikidata names extractor."""

import json
from pathlib import Path

import pytest
from kebab.utils.dataset.wikidata.wikidata_names_dataset_builder import WikidataNamesDatasetBuilder
from kebab.utils.dataset.wikidata.wikidata_simple_extractor import WikidataSimpleExtractor
from kebab.utils.dataset.wikidata.wikidata_utils import WikidataEntity


@pytest.fixture
def wikidata_dump_sample_input_file_path() -> Path:
    """Return the path to the input file with a subset of the Wikidata dump."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "sample" / "wikidata_dump_sample_records.json"


@pytest.fixture
def wikidata_hierarchy_input_file_path() -> Path:
    """Return the path to the input file with a subset Wikidata type hierarchy."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "wikidata_type_hierarchy.jsonl"


@pytest.fixture
def wikidata_simple_entities_input_file_path() -> Path:
    """Return the path to the input file with a subset of Wikidata simple entities."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "sample" / "wikidata_simple_entities.jsonl"


def test_wikidata_simple_extractor(wikidata_dump_sample_input_file_path: Path, tmp_path: Path) -> None:
    """Test the extraction of simple entities from Wikidata."""
    extractor = WikidataSimpleExtractor(
        wikidata_json_dump_path=wikidata_dump_sample_input_file_path,
        run_entity_extraction=True,
        run_property_scrape=False,
        output_dir=tmp_path,
    )

    extractor.extract_simple_entities()

    output_file_path = tmp_path / WikidataSimpleExtractor.WIKIDATA_SIMPLE_ENTITIES_FILENAME
    assert output_file_path.exists()

    with open(output_file_path, encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 10

    entity_0 = WikidataEntity.from_json(lines[0])
    assert entity_0.entity_id == "Q39293"
    entity_dict = json.loads(lines[0])

    assert entity_dict == {
        "entity_id": "Q39293",
        "properties": {
            "name": ["dissimulation"],
        },
    }


def test_extract_names(
    wikidata_hierarchy_input_file_path: Path, wikidata_simple_entities_input_file_path: Path, tmp_path: Path
) -> None:
    """Test the extraction of names from Wikidata."""
    builder = WikidataNamesDatasetBuilder(
        wikidata_entities_path=wikidata_simple_entities_input_file_path,
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
    assert lines[0].strip().startswith('{"entity_id": "Q23"')
