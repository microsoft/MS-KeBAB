# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the REBEL linking pair sampler."""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from kebab.utils.dataset.rebel.rebel_pair_sampler import MergeDistributionMode, RebelPairSampler
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
                "fragment_id": "1",
                "title": "Sample Title A",
                "entity_id": "Entity_1",
            },
        },
        {
            "entity_id": "Entity_1",
            "properties": {"name": ["Doe"]},
            "metadata": {
                "fragment_id": "2",
                "title": "Sample Title A",
                "entity_id": "Entity_1",
            },
        },
        {
            "entity_id": "Entity_2",
            "properties": {"name": ["Doe"]},
            "metadata": {
                "fragment_id": "3",
                "title": "Sample Title B",
                "entity_id": "Entity_2",
            },
        },
        {
            "entity_id": "Entity_3",
            "properties": {"name": ["John"]},
            "metadata": {
                "fragment_id": "4",
                "title": "Sample Title C",
                "entity_id": "Entity_3",
            },
        },
        {
            "entity_id": "Entity_4",
            "properties": {"name": ["Amy"]},
            "metadata": {
                "fragment_id": "5",
                "title": "Sample Title D",
                "entity_id": "Entity_4",
            },
        },
    ]

    fragments = [ResolvedWikidataEntity.from_dict(d) for d in fragment_dicts]
    conf_entities = RebelPairSampler.get_confusing_entities_map(fragments)

    assert sorted(conf_entities["Entity_1"]) == ["Entity_1", "Entity_2", "Entity_3"]
    assert sorted(conf_entities["Entity_2"]) == ["Entity_1", "Entity_2"]
    assert sorted(conf_entities["Entity_3"]) == ["Entity_1", "Entity_3"]
    assert sorted(conf_entities["Entity_4"]) == ["Entity_4"]

    # sample pairs
    rng = np.random.default_rng(0)

    pairs = list(
        RebelPairSampler.sample_pairs(
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
    builder = RebelPairSampler(
        fragments_path=rebel_sample_resolved_fragments_file_path,
        output_dir=tmp_path,
        exclude_single_property_value_entities=False,
    )
    builder.run()

    output_file = builder.linking_dataset_output_path
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert len(lines) > 0

    output_file = builder.linking_ground_truth_output_path
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert len(lines) > 0

    output_file = builder.clustering_dataset_output_path
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert len(lines) > 0

    output_file = builder.clustering_ground_truth_output_path
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert len(lines) > 0

    output_file = builder.entity_generation_dataset_output_path
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert len(lines) > 0

    output_file = builder.fragment_generation_dataset_output_path
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
            "metadata": {"fragment_id": "1", "type": ["t"], "entity_id": "E1", "title": "Title A"},
        },
        {
            "entity_id": "E1",
            "properties": {"name": ["B"]},
            "metadata": {"fragment_id": "2", "type": ["t"], "entity_id": "E1", "title": "Title B"},
        },
        {
            "entity_id": "E1",
            "properties": {"name": ["C", "A"]},
            "metadata": {"fragment_id": "3", "type": ["t"], "entity_id": "E1", "title": "Title C"},
        },
        {  # single-fragment entity
            "entity_id": "E2",
            "properties": {"name": ["X"]},
            "metadata": {"fragment_id": "4", "type": ["s"], "entity_id": "E2", "title": "Title X"},
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
    rng = np.random.default_rng(0)
    merged = RebelPairSampler.generate_merged_fragment(0, fragments, entity_idx, max_fragments=3, rng=rng)

    # base fragment included
    assert fragments[0].metadata["fragment_id"] in merged.metadata["fragment_ids"]

    # original metadata fields preserved
    assert merged.metadata["type"] == fragments[0].metadata["type"]

    # merge_count present and within expected range
    assert "merge_count" in merged.metadata
    assert 1 <= merged.metadata["merge_count"] <= 3

    # merged fragment names should come from all fragments
    assert set(merged.names).issubset({"A", "B", "C"})

    # when only one fragment exists for the entity, a merged representation is still produced
    single = RebelPairSampler.generate_merged_fragment(3, fragments, entity_idx, max_fragments=3, rng=rng)
    assert single.metadata["merge_count"] == 1
    assert single.metadata.get("fragment_ids") == [fragments[3].metadata["fragment_id"]]
    # entity_id and title metadata should be present
    assert "entity_id" in single.metadata
    assert single.metadata["entity_id"] == fragments[3].entity_id

    # no deduplication of values
    # create new fragments collection where we take fragment 3, which has "C" and "A" in names, multiple times
    fragments_with_duplicates = [fragments[2], fragments[2], fragments[2]]
    entity_idx_with_duplicates = {"E1": [0, 1, 2]}
    for _ in range(10):
        merged = RebelPairSampler.generate_merged_fragment(
            0,
            fragments_with_duplicates,
            entity_idx_with_duplicates,
            max_fragments=3,
            deduplicate_values=False,
            rng=rng,
        )

        if merged.metadata["merge_count"] == 1:
            continue

        # assert names contain both "C" and "A" more than once
        assert merged.names.count("A") > 1
        assert merged.names.count("C") > 1
        break
    else:
        pytest.fail("Expected to find a merged fragment, but did not.")


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
        merged = RebelPairSampler.generate_merged_fragment(
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


def _make_fragments_for_seed_tests() -> list[ResolvedWikidataEntity]:
    """Create a small synthetic fragment set.
    Entities E1, E2, E3 share overlapping names to yield multiple confusing pairs.
    """
    data = [
        {"entity_id": "E1", "properties": {"name": ["Alice", "Al"]}, "metadata": {"fragment_id": "1"}},
        {"entity_id": "E1", "properties": {"name": ["Alicia"]}, "metadata": {"fragment_id": "2"}},
        {"entity_id": "E1", "properties": {"name": ["A"]}, "metadata": {"fragment_id": "3"}},
        {"entity_id": "E2", "properties": {"name": ["Alice"]}, "metadata": {"fragment_id": "4"}},
        {"entity_id": "E2", "properties": {"name": ["Bob"]}, "metadata": {"fragment_id": "5"}},
        {"entity_id": "E3", "properties": {"name": ["Al", "Bobby"]}, "metadata": {"fragment_id": "6"}},
        {"entity_id": "E3", "properties": {"name": ["Alicia"]}, "metadata": {"fragment_id": "7"}},
        {"entity_id": "E4", "properties": {"name": ["Charlie"]}, "metadata": {"fragment_id": "8"}},
    ]
    return [ResolvedWikidataEntity.from_dict(d) for d in data]


def test_sample_pairs_seed_determinism() -> None:
    """sample_pairs returns the same sequence with the same seed, and differs with a different seed."""
    fragments = _make_fragments_for_seed_tests()
    conf = RebelPairSampler.get_confusing_entities_map(fragments)

    # same seed -> identical results
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    pairs1 = list(
        RebelPairSampler.sample_pairs(
            fragments, conf, max_count=50, min_acc_rate=1e-6, min_fragment_attempt_count=2, rng=rng1
        )
    )
    pairs2 = list(
        RebelPairSampler.sample_pairs(
            fragments, conf, max_count=50, min_acc_rate=1e-6, min_fragment_attempt_count=2, rng=rng2
        )
    )

    assert pairs1 == pairs2
    assert len(pairs1) > 0

    # different seed -> likely different sequence
    rng3 = np.random.default_rng(124)
    pairs3 = list(
        RebelPairSampler.sample_pairs(
            fragments, conf, max_count=50, min_acc_rate=1e-6, min_fragment_attempt_count=2, rng=rng3
        )
    )

    # Assert they differ in at least one position.
    assert any(a != b for a, b in zip(pairs1, pairs3, strict=False)) or len(pairs1) != len(pairs3)


def test_generate_merged_fragment_seed_determinism() -> None:
    """generate_merged_fragment selection is deterministic given the same seed and varies with a different seed."""
    fragments = _make_fragments_for_seed_tests()

    mapping: dict[str, list[int]] = defaultdict(list)
    for idx, f in enumerate(fragments):
        mapping[f.entity_id].append(idx)

    # pick a base fragment that has multiple siblings (E1 has 3)
    base_idx = 0

    rng_a = np.random.default_rng(2024)
    rng_b = np.random.default_rng(2024)
    merged_a = RebelPairSampler.generate_merged_fragment(
        base_idx, fragments, mapping, max_fragments=3, distribution=MergeDistributionMode.ZIPF, rng=rng_a
    )
    merged_b = RebelPairSampler.generate_merged_fragment(
        base_idx, fragments, mapping, max_fragments=3, distribution=MergeDistributionMode.ZIPF, rng=rng_b
    )

    assert merged_a.metadata["fragment_ids"] == merged_b.metadata["fragment_ids"]
    # entity_id metadata should be consistent for deterministic merges
    assert merged_a.metadata.get("entity_id") == merged_b.metadata.get("entity_id")

    rng_c = np.random.default_rng(2025)
    merged_c = RebelPairSampler.generate_merged_fragment(
        base_idx, fragments, mapping, max_fragments=3, distribution=MergeDistributionMode.ZIPF, rng=rng_c
    )

    # Assert they are different
    assert merged_a.metadata["fragment_ids"] != merged_c.metadata["fragment_ids"] or (
        merged_a.metadata["merge_count"] != merged_c.metadata["merge_count"]
    )


def test_end_to_end_seed_determinism(tmp_path: Path) -> None:
    """RebelPairSampler.run produces identical files with the same seed, and different with a different seed."""
    fragments = _make_fragments_for_seed_tests()

    # write a temporary fragments jsonl file
    frag_path = tmp_path / "fragments.jsonl"
    with open(frag_path, "w", encoding="utf-8") as f:
        for frag in fragments:
            f.write(frag.to_json(minimal_repr=True) + "\n")

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    out3 = tmp_path / "out3"

    # same seed runs
    sampler1 = RebelPairSampler(
        fragments_path=frag_path,
        output_dir=out1,
        max_count=100,
        max_merge_fragments=3,
        merge_distribution=MergeDistributionMode.ZIPF,
        seed=777,
        exclude_single_property_value_entities=False,
    )
    sampler2 = RebelPairSampler(
        fragments_path=frag_path,
        output_dir=out2,
        max_count=100,
        max_merge_fragments=3,
        merge_distribution=MergeDistributionMode.ZIPF,
        seed=777,
        exclude_single_property_value_entities=False,
    )

    sampler1.run()
    sampler2.run()

    # different seed run
    sampler3 = RebelPairSampler(
        fragments_path=frag_path,
        output_dir=out3,
        max_count=100,
        max_merge_fragments=3,
        merge_distribution=MergeDistributionMode.ZIPF,
        seed=778,
        exclude_single_property_value_entities=False,
    )
    sampler3.run()

    def _read(path: Path) -> list[str]:
        with open(path, encoding="utf-8") as f:
            return f.read().splitlines()

    # Compare linking datasets and ground truth; they should be identical for the same seed
    ld1 = _read(sampler1.linking_dataset_output_path)
    ld2 = _read(sampler2.linking_dataset_output_path)
    gt1 = _read(sampler1.linking_ground_truth_output_path)
    gt2 = _read(sampler2.linking_ground_truth_output_path)
    assert ld1 == ld2
    assert gt1 == gt2
    assert len(ld1) > 0

    # With a different seed, at least one of dataset or GT should differ
    ld3 = _read(sampler3.linking_dataset_output_path)
    gt3 = _read(sampler3.linking_ground_truth_output_path)
    assert ld1 != ld3 or gt1 != gt3


def test_exclude_singleton_entities(tmp_path: Path) -> None:
    """Test excluding singleton (single-property single-value) entities."""
    data = [
        {
            "entity_id": "E_singleton",
            "properties": {"name": ["Solo"]},
            "metadata": {"fragment_id": "1", "entity_id": "E_singleton"},
        },
        {
            "entity_id": "E_multi",
            "properties": {"name": ["Alpha"]},
            "metadata": {"fragment_id": "2", "entity_id": "E_multi"},
        },
        {
            "entity_id": "E_multi",
            "properties": {"name": ["Beta"]},
            "metadata": {"fragment_id": "3", "entity_id": "E_multi"},
        },
    ]

    fragments = [ResolvedWikidataEntity.from_dict(d) for d in data]
    frag_path = tmp_path / "fragments.jsonl"
    with open(frag_path, "w", encoding="utf-8") as f:
        for frag in fragments:
            f.write(frag.to_json(minimal_repr=True) + "\n")

    sampler = RebelPairSampler(
        fragments_path=frag_path,
        output_dir=tmp_path / "out_exclude",
        max_count=10,
        exclude_single_property_value_entities=True,
    )
    sampler.run()

    # Confusing entities map should only include E_multi (since singleton excluded)
    with open(sampler.confusing_entities_map_output_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    # When only one entity remains after exclusion, it will be confusing only with itself
    assert all("E_singleton" not in line for line in lines)
    assert any("E_multi" in line for line in lines)

    # Now run without exclusion -> both entities present
    sampler2 = RebelPairSampler(
        fragments_path=frag_path,
        output_dir=tmp_path / "out_include",
        max_count=10,
        exclude_single_property_value_entities=False,
    )
    sampler2.run()
    with open(sampler2.confusing_entities_map_output_path, encoding="utf-8") as f:
        lines2 = [line.strip() for line in f if line.strip()]
    assert any("E_singleton" in line for line in lines2)
    assert any("E_multi" in line for line in lines2)
