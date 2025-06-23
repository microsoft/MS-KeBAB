# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the REBEL linking and clustering dataset builder."""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from kebab.utils.dataset.rebel.rebel_dataset_builder import MergeDistributionMode, RebelDatasetBuilder
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


@pytest.fixture
def sample_fragments_for_merge() -> tuple[list[ResolvedWikidataEntity], dict[str, list[int]]]:
    """Create two entities, one with three fragments and one with a single fragment."""
    frag_dicts = [
        {
            "entity_id": "E1",
            "properties": {"name": ["A"]},
            "metadata": {"fragment_id": "1", "type": ["t"]},
        },
        {
            "entity_id": "E1",
            "properties": {"name": ["B"]},
            "metadata": {"fragment_id": "2", "type": ["t"]},
        },
        {
            "entity_id": "E1",
            "properties": {"name": ["C"]},
            "metadata": {"fragment_id": "3", "type": ["t"]},
        },
        {  # single-fragment entity
            "entity_id": "E2",
            "properties": {"name": ["X"]},
            "metadata": {"fragment_id": "4", "type": ["s"]},
        },
    ]
    fragments = [ResolvedWikidataEntity.from_dict(d) for d in frag_dicts]

    mapping: dict[str, list[int]] = defaultdict(list)
    for idx, frag in enumerate(fragments):
        mapping[frag.entity_id].append(idx)

    return fragments, mapping


def test_generate_merged_fragment(sample_fragments_for_merge) -> None:
    """Test generate_merged_fragment method (sanity check)."""
    fragments, entity_idx = sample_fragments_for_merge

    np.random.seed(0)  # noqa: NPY002
    random.seed(0)
    merged = RebelDatasetBuilder.generate_merged_fragment(0, fragments, entity_idx, max_fragments=3)

    # original metadata fields preserved
    assert merged.metadata["fragment_id"] == fragments[0].metadata["fragment_id"]
    assert merged.metadata["type"] == fragments[0].metadata["type"]

    # merge_count present and within expected range
    assert "merge_count" in merged.metadata
    assert 1 <= merged.metadata["merge_count"] <= 3

    # merged fragment names should come from all fragments
    assert set(merged.names).issubset({"A", "B", "C"})

    # when only one fragment exists for the entity, no merge occurs
    single = RebelDatasetBuilder.generate_merged_fragment(3, fragments, entity_idx, max_fragments=3)
    assert single is fragments[3]
    assert "merge_count" not in single.metadata


@pytest.mark.parametrize("dist", [MergeDistributionMode.ZIPF, MergeDistributionMode.TRIANGULAR])
def test_generate_merged_fragment_stats(dist: MergeDistributionMode) -> None:
    """Test generate_merged_fragment - that the empirical distribution is close to the theoretical one."""

    def make_entity(num_frags: int = 10) -> tuple[list[ResolvedWikidataEntity], dict[str, list[int]]]:
        """Create `num_frags` fragments for a single entity."""
        frags = [
            ResolvedWikidataEntity.from_dict(
                {
                    "entity_id": "E",
                    "properties": {"name": [f"n{i}"]},
                    "metadata": {"fragment_id": str(i), "type": ["t"]},
                }
            )
            for i in range(num_frags)
        ]

        mapping: dict[str, list[int]] = defaultdict(list)
        for idx in range(num_frags):
            mapping["E"].append(idx)

        return frags, mapping

    n_frags = 10
    n_samples = 20_000
    tol = 0.025

    np.random.seed(0)  # noqa: NPY002
    random.seed(0)

    fragments, mapping = make_entity(n_frags)
    counts = np.zeros(n_frags, dtype=np.int32)

    for _ in range(n_samples):
        merged = RebelDatasetBuilder.generate_merged_fragment(
            0, fragments, mapping, max_fragments=n_frags, distribution=dist
        )

        counts[merged.metadata["merge_count"] - 1] += 1

    empirical = counts / counts.sum()

    if dist == MergeDistributionMode.ZIPF:
        probs = 1 / np.arange(1, n_frags + 1, dtype=np.float32)
    elif dist == MergeDistributionMode.TRIANGULAR:
        probs = np.arange(n_frags, 0, -1, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported distribution mode: {dist}")

    probs /= probs.sum()

    max_abs_diff = np.abs(empirical - probs).max()
    assert max_abs_diff < tol, f"max deviation {max_abs_diff:.3f} exceeds tolerance {tol}"
