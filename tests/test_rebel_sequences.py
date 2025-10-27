import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kebab.utils.dataset.rebel.rebel_split_sampler import (
    ClusteringConfig,
    FragmentSetGenerationConfig,
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
    e1_frags = [_fragment("E1", f"1{i}") for i in range(5)]
    e2_frags = [_fragment("E2", f"2{i}") for i in range(5)]

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

    # Fragment set generation: 2 entities * 4 sequences (one line per sequence)
    fragset_path = base_dir / "out/fragment_set_generation/seqtest/rebel_fragment_set_generation_dataset.jsonl"
    with open(fragset_path, encoding="utf-8") as f:
        fragset_lines = f.readlines()
    assert len(fragset_lines) == 2 * 4
