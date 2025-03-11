# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the REBEL linking and clustering dataset builder."""

from pathlib import Path

import numpy as np
import pytest
from kebab.utils.dataset.rebel.rebel_dataset_builder import RebelDatasetBuilder
from kebab.utils.dataset.wikidata.wikidata_utils import ResolvedWikidataEntity


@pytest.fixture
def rebel_sample_resolved_fragments_file_path() -> Path:
    """Path to the sample resolved REBEL fragments input file."""
    return (
        Path(__file__).parent
        / "data"
        / "datasets"
        / "rebel"
        / "fragments"
        / "resolved"
        / "rebel_entity_fragments.jsonl"
    )


def test_get_confusing_entities_map_and_sample() -> None:
    """Test obtaining the confusing entities map followed by sampling."""
    fragment_dicts = [
        {
            "entity_id": "Entity_1",
            "properties": {"name": ["John", "John Doe"]},
            "metadata": {
                "doc_id": "0",
                "source_text_hash": "0",
                "fragment_id": "1",
            },
        },
        {
            "entity_id": "Entity_1",
            "properties": {"name": ["Doe"]},
            "metadata": {
                "doc_id": "0",
                "source_text_hash": "0",
                "fragment_id": "2",
            },
        },
        {
            "entity_id": "Entity_2",
            "properties": {"name": ["Doe"]},
            "metadata": {
                "doc_id": "0",
                "source_text_hash": "0",
                "fragment_id": "3",
            },
        },
        {
            "entity_id": "Entity_3",
            "properties": {"name": ["John"]},
            "metadata": {
                "doc_id": "0",
                "source_text_hash": "0",
                "fragment_id": "4",
            },
        },
        {
            "entity_id": "Entity_4",
            "properties": {"name": ["Amy"]},
            "metadata": {
                "doc_id": "0",
                "source_text_hash": "0",
                "fragment_id": "5",
            },
        },
    ]

    fragments = [ResolvedWikidataEntity.from_dict(d) for d in fragment_dicts]
    conf_entities = RebelDatasetBuilder.get_confusing_entities_map(fragments)

    assert sorted(conf_entities["Entity_1"]) == ["Entity_1", "Entity_2", "Entity_3"]
    assert sorted(conf_entities["Entity_2"]) == ["Entity_1", "Entity_2"]
    assert sorted(conf_entities["Entity_3"]) == ["Entity_1", "Entity_3"]
    assert sorted(conf_entities["Entity_4"]) == ["Entity_4"]

    # sample pairs
    rng = np.random.default_rng(0)

    pairs = list(
        RebelDatasetBuilder.sample_pairs(
            fragments,
            conf_entities,
            min_acc_rate=1e-6,
            min_fragment_attempt_count=1000,
            max_count=5,
            rng=rng,
        )
    )

    # fragment 1 with 2, 3, 4
    # fragment 2 with 3, 4
    assert len(pairs) == 5

    pairs = [(fragments[left], fragments[right]) for left, right in pairs]
    pairs = sorted(sorted((left.property_values_str(), right.property_values_str())) for left, right in pairs)
    assert pairs == [
        ["Entity_1 | name:Doe", "Entity_1 | name:John,John Doe"],
        ["Entity_1 | name:Doe", "Entity_2 | name:Doe"],
        ["Entity_1 | name:Doe", "Entity_3 | name:John"],
        ["Entity_1 | name:John,John Doe", "Entity_2 | name:Doe"],
        ["Entity_1 | name:John,John Doe", "Entity_3 | name:John"],
    ]


def test_run(rebel_sample_resolved_fragments_file_path: Path, tmp_path: Path) -> None:
    """Test building linking and clustering datasets from the resolved REBEL fragments."""
    builder = RebelDatasetBuilder(fragments_path=rebel_sample_resolved_fragments_file_path, output_dir=tmp_path)
    builder.run()

    output_file = tmp_path / RebelDatasetBuilder.LINKING_DATASET_FILENAME
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert len(lines) > 0

    output_file = tmp_path / RebelDatasetBuilder.LINKING_GROUND_TRUTH_FILENAME
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert len(lines) > 0

    output_file = tmp_path / RebelDatasetBuilder.CLUSTERING_DATASET_FILENAME
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert len(lines) > 0

    output_file = tmp_path / RebelDatasetBuilder.CLUSTERING_GROUND_TRUTH_FILENAME
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert len(lines) > 0
