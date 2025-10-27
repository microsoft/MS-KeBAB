# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the REBEL split sampler."""

from pathlib import Path

import pytest
from kebab.utils.dataset.rebel.rebel_pair_sampler import RebelPairSampler
from kebab.utils.dataset.rebel.rebel_split_sampler import (
    ClusteringConfig,
    FragmentSetGenerationConfig,
    LinkingConfig,
    RebelSplitSampler,
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
    # Enable uniform sampling to avoid degenerate case where confusing-entity logic yields zero pairs
    # on very small synthetic test data. This ensures at least one entity enters split A.
    pair_sampler = RebelPairSampler(
        fragments_path=rebel_sample_resolved_fragments_file_path,
        output_dir=base_dir,
        exclude_single_property_value_entities=False,
        uniform_sampling=True,
        max_count=200,
    )
    pair_sampler.run()

    # 2) Configure two tiny splits to keep the test fast
    test_split = SplitBuildConfig(
        name="test",
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
        fragment_set_generation=FragmentSetGenerationConfig(),
    )

    val_split = SplitBuildConfig(
        name="dev",
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
        fragment_set_generation=FragmentSetGenerationConfig(),
    )

    out_dir = tmp_path / "subsampled"
    split_sampler = RebelSplitSampler(
        input_dir=base_dir,
        output_dir=out_dir,
        splits=[test_split, val_split],
    )
    split_sampler.run()

    # 3) Verify outputs: test must be non-empty; dev may be empty depending on tiny sample size
    test_linking_dir = out_dir / "linking" / test_split.name
    test_clustering_dir = out_dir / "clustering" / test_split.name
    test_entity_gen_dir = out_dir / "entity_generation" / test_split.name
    test_fragment_gen_dir = out_dir / "fragment_generation" / test_split.name
    _assert_non_empty(test_linking_dir / RebelPairSampler.LINKING_DATASET_FILENAME)
    _assert_non_empty(test_linking_dir / RebelPairSampler.LINKING_GROUND_TRUTH_FILENAME)
    _assert_non_empty(test_linking_dir / "rebel_linking_used_entities.jsonl")
    _assert_non_empty(test_clustering_dir / RebelPairSampler.CLUSTERING_DATASET_FILENAME)
    _assert_non_empty(test_clustering_dir / RebelPairSampler.CLUSTERING_GROUND_TRUTH_FILENAME)
    _assert_non_empty(test_entity_gen_dir / RebelPairSampler.ENTITY_GENERATION_DATASET_FILENAME)
    _assert_non_empty(test_fragment_gen_dir / RebelPairSampler.FRAGMENT_GENERATION_DATASET_FILENAME)

    from kebab.utils.dataset.rebel.rebel_split_sampler import RebelSplitSampler as _Sampler

    # fragment set generation dataset
    fragset_dir = out_dir / "fragment_set_generation" / test_split.name
    _assert_non_empty(fragset_dir / _Sampler.FRAGMENT_SET_GENERATION_DATASET_FILENAME)

    # dev files should exist; they may be empty if entities are exhausted
    val_linking_dir = out_dir / "linking" / val_split.name
    val_clustering_dir = out_dir / "clustering" / val_split.name
    val_entity_gen_dir = out_dir / "entity_generation" / val_split.name
    val_fragment_gen_dir = out_dir / "fragment_generation" / val_split.name
    assert (val_linking_dir / RebelPairSampler.LINKING_DATASET_FILENAME).exists()
    assert (val_linking_dir / RebelPairSampler.LINKING_GROUND_TRUTH_FILENAME).exists()
    assert (val_linking_dir / "rebel_linking_used_entities.jsonl").exists()
    assert (val_clustering_dir / RebelPairSampler.CLUSTERING_DATASET_FILENAME).exists()
    assert (val_clustering_dir / RebelPairSampler.CLUSTERING_GROUND_TRUTH_FILENAME).exists()
    assert (val_entity_gen_dir / RebelPairSampler.ENTITY_GENERATION_DATASET_FILENAME).exists()
    assert (val_fragment_gen_dir / RebelPairSampler.FRAGMENT_GENERATION_DATASET_FILENAME).exists()
    # fragment set generation dataset
    assert (
        out_dir / "fragment_set_generation" / val_split.name / _Sampler.FRAGMENT_SET_GENERATION_DATASET_FILENAME
    ).exists()


def test_split_sampler_disjoint_entities_between_splits(
    rebel_sample_resolved_fragments_file_path: Path, tmp_path: Path
) -> None:
    """Ensure entity sets used by different splits are disjoint as intended."""
    # Build base
    base_dir = tmp_path / "base"
    pair_sampler = RebelPairSampler(
        fragments_path=rebel_sample_resolved_fragments_file_path,
        output_dir=base_dir,
        exclude_single_property_value_entities=False,
    )
    pair_sampler.run()

    # Two small splits
    splits = [
        SplitBuildConfig(
            name="A",
            linking=LinkingConfig(
                # Lower limits to better fit tiny test data and guarantee at least one entity appears.
                pair_count_limit=20,
                pairs_per_entity_limit=5,
                property_pattern_count_limit=40,
                property_overlap_limits={},
                single_class_ratio_limit=None,
                seed=1,
            ),
            clustering=ClusteringConfig(
                min_fragments_per_entity=1,
                entity_count_limit=0,
                fragments_per_entity_limit=10,
                include_all_linking_fragments_first=True,
            ),
            fragment_set_generation=FragmentSetGenerationConfig(),
        ),
        SplitBuildConfig(
            name="B",
            linking=LinkingConfig(
                pair_count_limit=20,
                pairs_per_entity_limit=5,
                property_pattern_count_limit=40,
                property_overlap_limits={},
                single_class_ratio_limit=None,
                seed=2,
            ),
            clustering=ClusteringConfig(
                min_fragments_per_entity=1,
                entity_count_limit=0,
                fragments_per_entity_limit=10,
                include_all_linking_fragments_first=True,
            ),
            fragment_set_generation=FragmentSetGenerationConfig(),
        ),
    ]

    out_dir = tmp_path / "out"
    RebelSplitSampler(input_dir=base_dir, output_dir=out_dir, splits=splits).run()

    # Read used entity sets
    def read_entities(p: Path) -> set[str]:
        with open(p, encoding="utf-8") as f:
            return {line.strip().strip('"') for line in f if line.strip()}

    ents_a = read_entities(out_dir / "linking" / "A" / "rebel_linking_used_entities.jsonl")
    ents_b = read_entities(out_dir / "linking" / "B" / "rebel_linking_used_entities.jsonl")

    assert ents_a, "Split A should have entities"
    # Split B can be empty on tiny input; but if not empty, must be disjoint
    assert ents_a.isdisjoint(ents_b), "Entity sets between splits must be disjoint"
