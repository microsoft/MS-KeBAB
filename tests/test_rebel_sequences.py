import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kebab.utils.dataset.rebel.rebel_split_sampler import (
    ClusteringConfig,
    FragmentSetGenerationConfig,
    IncrementalLinkingConfig,
    LinkingConfig,
    RebelSplitSampler,
    SplitBuildConfig,
)


def _write_jsonl(path: Path, lines: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


def _fragment(entity_id: str, frag_id: str) -> dict:
    return {
        "entity_id": entity_id,
        "properties": {"name": [f"{entity_id}-{frag_id}"]},
        "metadata": {"fragment_id": frag_id, "type": ["T"], "entity_id": entity_id},
    }


def test_sequences_per_entity_sampling(tmp_path):
    # Build minimal base dataset structure
    base_dir = tmp_path

    # Two entities, five fragments each
    e1_frags = [_fragment("E1", f"f1{i}") for i in range(5)]
    e2_frags = [_fragment("E2", f"f2{i}") for i in range(5)]

    clustering_dataset = e1_frags + e2_frags
    clustering_gt = ["E1"] * len(e1_frags) + ["E2"] * len(e2_frags)

    # Base linking dataset: one positive pair per entity (needs fragment_to_entity resolution)
    linking_pairs = [
        [
            dict(e1_frags[0].items()),
            dict(e1_frags[1].items()),
        ],
        [
            dict(e2_frags[0].items()),
            dict(e2_frags[1].items()),
        ],
    ]
    linking_gt = [True, True]

    frag_to_ent_map = [[f["metadata"]["fragment_id"], f["entity_id"]] for f in clustering_dataset]

    # Entity generation + fragment generation reuse clustering fragments (merged vs single not required for test)
    entity_gen_dataset = clustering_dataset
    fragment_gen_dataset = clustering_dataset

    # Write files
    _write_jsonl(base_dir / "linking/base/rebel_linking_dataset.jsonl", linking_pairs)
    _write_jsonl(base_dir / "linking/base/rebel_linking_ground_truth.jsonl", linking_gt)
    _write_jsonl(base_dir / "linking/base/rebel_fragment_to_entity_map.jsonl", frag_to_ent_map)
    _write_jsonl(base_dir / "clustering/base/rebel_clustering_dataset.jsonl", clustering_dataset)
    _write_jsonl(base_dir / "clustering/base/rebel_clustering_ground_truth.jsonl", clustering_gt)
    _write_jsonl(base_dir / "entity_generation/base/rebel_entity_generation_dataset.jsonl", entity_gen_dataset)
    _write_jsonl(base_dir / "fragment_generation/base/rebel_fragment_generation_dataset.jsonl", fragment_gen_dataset)

    split = SplitBuildConfig(
        name="seqtest",
        linking=LinkingConfig(
            pair_count_limit=10,
            pairs_per_entity_limit=10,
            property_pattern_count_limit=10,
            seed=1,
        ),
        clustering=ClusteringConfig(seed=2),
        incremental_linking=IncrementalLinkingConfig(
            max_chain_len=5,
            sequences_per_entity=3,
            negatives_per_positive=1,
            seed=3,
        ),
        incremental_set_linking=IncrementalLinkingConfig(
            max_chain_len=5,
            sequences_per_entity=2,
            negatives_per_positive=1,
            left_as_set=True,
            seed=4,
        ),
        fragment_set_generation=FragmentSetGenerationConfig(
            max_fragments_per_entity=5,
            sequences_per_entity=4,
            seed=5,
        ),
    )

    sampler = RebelSplitSampler(
        input_dir=base_dir,
        output_dir=base_dir / "out",
        splits=[split],
    )
    sampler.run()

    # Assert incremental linking counts
    incr_link_path = base_dir / "out/incremental_linking/seqtest/rebel_linking_dataset.jsonl"
    incr_link_gt_path = base_dir / "out/incremental_linking/seqtest/rebel_linking_ground_truth.jsonl"
    with open(incr_link_path, encoding="utf-8") as f:
        incr_lines = f.readlines()
    with open(incr_link_gt_path, encoding="utf-8") as f:
        incr_gt_lines = f.readlines()
    # 2 entities * 3 sequences * (1 positive + 1 negative)
    assert len(incr_lines) == 2 * 3 * 2
    assert len(incr_gt_lines) == len(incr_lines)
    # Ensure at least one negative present
    assert any(json.loads(x) is False for x in incr_gt_lines)

    # Assert incremental set linking counts
    incr_set_path = base_dir / "out/incremental_set_linking/seqtest/rebel_set_linking_dataset.jsonl"
    incr_set_gt_path = base_dir / "out/incremental_set_linking/seqtest/rebel_set_linking_ground_truth.jsonl"
    with open(incr_set_path, encoding="utf-8") as f:
        incr_set_lines = f.readlines()
    with open(incr_set_gt_path, encoding="utf-8") as f:
        incr_set_gt_lines = f.readlines()
    # 2 entities * 2 sequences * (1 positive + 1 negative)
    assert len(incr_set_lines) == 2 * 2 * 2
    assert len(incr_set_gt_lines) == len(incr_set_lines)

    # Fragment set generation: 2 entities * 4 sequences (one line per sequence)
    fragset_path = base_dir / "out/fragment_set_generation/seqtest/rebel_fragment_set_generation_dataset.jsonl"
    with open(fragset_path, encoding="utf-8") as f:
        fragset_lines = f.readlines()
    assert len(fragset_lines) == 2 * 4

    # Basic structural check: each incremental linking line is a two-element JSON array
    first_obj = json.loads(incr_lines[0])
    assert isinstance(first_obj, list)
    assert len(first_obj) == 2
