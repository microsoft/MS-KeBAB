# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the REBEL dataset subsampler."""

from pathlib import Path

import pytest
from kebab.utils.dataset.rebel.rebel_dataset_builder import RebelDatasetBuilder
from kebab.utils.dataset.rebel.rebel_dataset_subsampler import (
    ClusteringConfig,
    LinkingConfig,
    RebelDatasetSubsampler,
    SplitBuildConfig,
)


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


def _assert_non_empty(path: Path) -> None:
    """Assert that a file exists and is non-empty."""
    assert path.exists(), f"Expected file to exist: {path}"
    with open(path, encoding="utf-8") as f:
        line = f.readline()
    assert line, f"Expected file to be non-empty: {path}"


def test_run(rebel_sample_resolved_fragments_file_path: Path, tmp_path: Path) -> None:
    """End-to-end: build base datasets, then subsample two small splits and verify outputs."""
    # 1) Build base datasets with the builder
    base_dir = tmp_path / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    builder = RebelDatasetBuilder(
        fragments_path=rebel_sample_resolved_fragments_file_path,
        output_dir=base_dir,
    )
    builder.run()

    # 2) Configure two tiny splits to keep the test fast
    test_split = SplitBuildConfig(
        name="Test",
        linking=LinkingConfig(
            pair_count_limit=50,
            pairs_per_entity_limit=5,
            property_pattern_count_limit=50,
            property_overlap_limits={},
            single_class_ratio_limit=None,
            seed=123,
        ),
        clustering=ClusteringConfig(
            min_fragments_per_entity=1,
            entity_count_limit=0,
            fragments_per_entity_limit=10,
            include_all_linking_fragments_first=True,
        ),
    )

    val_split = SplitBuildConfig(
        name="Validation",
        linking=LinkingConfig(
            pair_count_limit=50,
            pairs_per_entity_limit=5,
            property_pattern_count_limit=50,
            property_overlap_limits={},
            single_class_ratio_limit=None,
            seed=456,
        ),
        clustering=ClusteringConfig(
            min_fragments_per_entity=1,
            entity_count_limit=0,
            fragments_per_entity_limit=10,
            include_all_linking_fragments_first=True,
        ),
    )

    out_dir = tmp_path / "subsampled"
    subsampler = RebelDatasetSubsampler(
        base_dir=base_dir,
        output_dir=out_dir,
        splits=[test_split, val_split],
    )
    subsampler.run()

    # 3) Verify outputs: Test must be non-empty; Validation may be empty depending on tiny sample size
    test_linking_dir = out_dir / "linking" / "Test"
    test_clustering_dir = out_dir / "clustering" / "Test"
    test_entity_gen_dir = out_dir / "entity_generation" / "Test"
    test_fragment_gen_dir = out_dir / "fragment_generation" / "Test"
    _assert_non_empty(test_linking_dir / RebelDatasetBuilder.LINKING_DATASET_FILENAME)
    _assert_non_empty(test_linking_dir / RebelDatasetBuilder.LINKING_GROUND_TRUTH_FILENAME)
    _assert_non_empty(test_linking_dir / "rebel_linking_used_entities.jsonl")
    _assert_non_empty(test_clustering_dir / RebelDatasetBuilder.CLUSTERING_DATASET_FILENAME)
    _assert_non_empty(test_clustering_dir / RebelDatasetBuilder.CLUSTERING_GROUND_TRUTH_FILENAME)
    _assert_non_empty(test_entity_gen_dir / RebelDatasetBuilder.ENTITY_GENERATION_DATASET_FILENAME)
    _assert_non_empty(test_fragment_gen_dir / RebelDatasetBuilder.FRAGMENT_GENERATION_DATASET_FILENAME)

    # Validation files should exist; they may be empty if entities are exhausted
    val_linking_dir = out_dir / "linking" / "Validation"
    val_clustering_dir = out_dir / "clustering" / "Validation"
    val_entity_gen_dir = out_dir / "entity_generation" / "Validation"
    val_fragment_gen_dir = out_dir / "fragment_generation" / "Validation"
    assert (val_linking_dir / RebelDatasetBuilder.LINKING_DATASET_FILENAME).exists()
    assert (val_linking_dir / RebelDatasetBuilder.LINKING_GROUND_TRUTH_FILENAME).exists()
    assert (val_linking_dir / "rebel_linking_used_entities.jsonl").exists()
    assert (val_clustering_dir / RebelDatasetBuilder.CLUSTERING_DATASET_FILENAME).exists()
    assert (val_clustering_dir / RebelDatasetBuilder.CLUSTERING_GROUND_TRUTH_FILENAME).exists()
    assert (val_entity_gen_dir / RebelDatasetBuilder.ENTITY_GENERATION_DATASET_FILENAME).exists()
    assert (val_fragment_gen_dir / RebelDatasetBuilder.FRAGMENT_GENERATION_DATASET_FILENAME).exists()


def test_subsampler_disjoint_entities_between_splits(
    rebel_sample_resolved_fragments_file_path: Path, tmp_path: Path
) -> None:
    """Ensure entity sets used by different splits are disjoint as intended."""
    # Build base
    base_dir = tmp_path / "base"
    builder = RebelDatasetBuilder(fragments_path=rebel_sample_resolved_fragments_file_path, output_dir=base_dir)
    builder.run()

    # Two small splits
    splits = [
        SplitBuildConfig(
            name="A",
            linking=LinkingConfig(
                pair_count_limit=40,
                pairs_per_entity_limit=5,
                property_pattern_count_limit=40,
                property_overlap_limits={},
                single_class_ratio_limit=None,
                seed=1,
            ),
        ),
        SplitBuildConfig(
            name="B",
            linking=LinkingConfig(
                pair_count_limit=40,
                pairs_per_entity_limit=5,
                property_pattern_count_limit=40,
                property_overlap_limits={},
                single_class_ratio_limit=None,
                seed=2,
            ),
        ),
    ]

    out_dir = tmp_path / "out"
    RebelDatasetSubsampler(base_dir=base_dir, output_dir=out_dir, splits=splits).run()

    # Read used entity sets
    def read_entities(p: Path) -> set[str]:
        with open(p, encoding="utf-8") as f:
            return {line.strip().strip('"') for line in f if line.strip()}

    ents_a = read_entities(out_dir / "linking" / "A" / "rebel_linking_used_entities.jsonl")
    ents_b = read_entities(out_dir / "linking" / "B" / "rebel_linking_used_entities.jsonl")

    assert ents_a, "Split A should have entities"
    # Split B can be empty on tiny input; but if not empty, must be disjoint
    assert ents_a.isdisjoint(ents_b), "Entity sets between splits must be disjoint"
