"""Create split datasets (e.g., Test/Validation/Train) from a base REBEL dataset.

Workflow summary:
1. Load the base linking dataset.
2. Optionally apply type filtering using the Wikidata type hierarchy.
3. Sample a heuristically balanced linking subset (entity, property-pattern, overlap, class balance constraints).
4. Derive auxiliary datasets (clustering, entity generation, fragment generation, fragment set generation).
5. Derive positive/negative linking generation datasets (each record is a raw pair) for sequence generation tasks.

Input layout (produced by RebelPairSampler into input_dir):
* input_dir/linking/base/rebel_linking_dataset.jsonl
* input_dir/linking/base/rebel_linking_ground_truth.jsonl
* input_dir/clustering/base/rebel_clustering_dataset.jsonl
* input_dir/clustering/base/rebel_clustering_ground_truth.jsonl
* input_dir/entity_generation/base/rebel_entity_generation_dataset.jsonl
* input_dir/fragment_generation/base/rebel_fragment_generation_dataset.jsonl

Output layout (within output_dir):
* output_dir/linking/<split>/rebel_linking_dataset.jsonl
* output_dir/linking/<split>/rebel_linking_ground_truth.jsonl
* output_dir/linking/<split>/rebel_linking_used_entities.jsonl
* output_dir/linking/<split>/rebel_linking_generation_positive_dataset.jsonl
* output_dir/linking/<split>/rebel_linking_generation_negative_dataset.jsonl
* output_dir/linking/<split>/rebel_linking_set_dataset.jsonl
* output_dir/linking/<split>/rebel_linking_set_ground_truth.jsonl
* output_dir/clustering/<split>/rebel_clustering_dataset.jsonl
* output_dir/clustering/<split>/rebel_clustering_ground_truth.jsonl
* output_dir/entity_generation/<split>/rebel_entity_generation_dataset.jsonl
* output_dir/fragment_generation/<split>/rebel_fragment_generation_dataset.jsonl
* output_dir/fragment_set_generation/<split>/rebel_fragment_set_generation_dataset.jsonl
* output_dir/fragment_set_generation/<split>/rebel_negative_fragment_set_generation_dataset.jsonl
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from kebab.utils.dataset.rebel.rebel_pair_sampler import RebelPairSampler
from kebab.utils.dataset.wikidata import wikidata_utils
from kebab.utils.dataset.wikidata.wikidata_utils import ResolvedWikidataEntity
from kebab.utils.io_helpers import resolve_path


@dataclass
class LinkingConfig:
    """Sampling configuration for a split of the linking dataset."""

    # size limits
    pair_count_limit: int  # total pairs in the split
    pairs_per_entity_limit: int  # max pairs per entity in the split

    # distribution limits
    property_pattern_count_limit: int  # max pairs per property key pattern in the split
    property_overlap_limits: dict[int, float] = field(
        default_factory=dict
    )  # e.g. only one property in common => no more than 35% of pairs
    single_class_ratio_limit: float | None = None

    seed: int | None = None


@dataclass
class ClusteringConfig:
    """Configuration for building a clustering split from selected entities."""

    min_fragments_per_entity: int = 3
    entity_count_limit: int = 0  # 0 means no limit
    fragments_per_entity_limit: int = 25
    include_all_linking_fragments_first: bool = True
    seed: int | None = None


@dataclass
class FragmentSetGenerationConfig:
    """Configuration for building fragment set generation dataset per entity."""

    max_fragments_per_entity: int = 100
    sequences_per_entity: int | None = 5  # if None use fragment_count cap
    seed: int | None = None


@dataclass
class SplitBuildConfig:
    """Full configuration for producing a single split."""

    name: str
    linking: LinkingConfig
    clustering: ClusteringConfig
    build_set_linking_datasets: bool = False
    build_set_generation_datasets: bool = False
    fragment_set_generation: FragmentSetGenerationConfig | None = None


@dataclass
class TypeFilter:
    """Type filtering settings using the Wikidata type hierarchy graph.

    If required_type_ids is set, only entities whose types intersect the required set are kept.
    If exclude_type_ids is set, any entity whose types intersect the excluded set is skipped.
    """

    type_hierarchy_path: Path
    required_type_ids: list[str] | None = None
    exclude_type_ids: list[str] | None = None
    include_descendants: bool = False


class RebelSplitSampler:
    """Create requested splits from a base REBEL dataset in a single run.

    Adds per-split positive/negative linking generation datasets:
    Each record is the original pair (minimal representation) routed by its label.
    """

    SET_LINKING_DATASET_FILENAME: str = "rebel_linking_set_dataset.jsonl"
    SET_LINKING_GROUND_TRUTH_FILENAME: str = "rebel_linking_set_ground_truth.jsonl"
    FRAGMENT_SET_GENERATION_DATASET_FILENAME: str = "rebel_fragment_set_generation_dataset.jsonl"
    NEGATIVE_FRAGMENT_SET_GENERATION_DATASET_FILENAME: str = "rebel_negative_fragment_set_generation_dataset.jsonl"
    LINKING_GENERATION_POSITIVE_DATASET_FILENAME: str = "rebel_linking_generation_positive_dataset.jsonl"
    LINKING_GENERATION_NEGATIVE_DATASET_FILENAME: str = "rebel_linking_generation_negative_dataset.jsonl"

    # Trim fragments before sampling and reattach info later (saves memory, enable on a <128GB RAM)
    ENABLE_FRAGMENT_TRIMMING: bool = False

    _logger: logging.Logger
    input_dir: Path
    output_dir: Path
    splits: list[SplitBuildConfig]
    type_filter: TypeFilter | None
    base_linking_dataset_path: Path
    base_linking_ground_truth_path: Path
    base_clustering_dataset_path: Path
    base_clustering_ground_truth_path: Path
    base_entity_generation_dataset_path: Path
    base_fragment_generation_dataset_path: Path
    required_entity_types: set[str]
    excluded_entity_types: set[str]

    def __init__(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        splits: list[SplitBuildConfig],
        type_filter: TypeFilter | None = None,
    ) -> None:
        """Initialize the dataset split sampler.

        Args:
            input_dir: Directory containing the base datasets produced by RebelDatasetBuilder.
            output_dir: Directory where the split datasets will be written.
            splits: The list of split configurations to produce (e.g., Test, Validation, Train) in order.
            type_filter: Optional type filtering configuration based on Wikidata type hierarchy.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self.input_dir = resolve_path(input_dir)
        self.output_dir = resolve_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.splits = splits
        self.type_filter = type_filter
        self.base_linking_dataset_path = self.input_dir / "linking" / "base" / RebelPairSampler.LINKING_DATASET_FILENAME
        self.base_linking_ground_truth_path = (
            self.input_dir / "linking" / "base" / RebelPairSampler.LINKING_GROUND_TRUTH_FILENAME
        )
        self.base_clustering_dataset_path = (
            self.input_dir / "clustering" / "base" / RebelPairSampler.CLUSTERING_DATASET_FILENAME
        )
        self.base_clustering_ground_truth_path = (
            self.input_dir / "clustering" / "base" / RebelPairSampler.CLUSTERING_GROUND_TRUTH_FILENAME
        )
        self.base_entity_generation_dataset_path = (
            self.input_dir / "entity_generation" / "base" / RebelPairSampler.ENTITY_GENERATION_DATASET_FILENAME
        )
        self.base_fragment_generation_dataset_path = (
            self.input_dir / "fragment_generation" / "base" / RebelPairSampler.FRAGMENT_GENERATION_DATASET_FILENAME
        )

        # Type filtering setup
        self.required_entity_types = set()
        self.excluded_entity_types = set()
        if self.type_filter and (
            (self.type_filter.required_type_ids and len(self.type_filter.required_type_ids) > 0)
            or (self.type_filter.exclude_type_ids and len(self.type_filter.exclude_type_ids) > 0)
        ):
            self._init_type_filters()

    def _init_type_filters(self) -> None:
        """Initialize type filtering by loading the hierarchy and collecting target/excluded sets."""
        assert self.type_filter
        assert self.type_filter.type_hierarchy_path.exists(), (
            f"Wikidata type hierarchy file not found: {self.type_filter.type_hierarchy_path}"
        )

        graph, type_id_to_node = wikidata_utils.load_type_hierarchy(resolve_path(self.type_filter.type_hierarchy_path))

        # Collect required set from provided required_type_ids
        if self.type_filter.required_type_ids:
            required: set[str] = set()
            for req_id in self.type_filter.required_type_ids:
                types = (
                    wikidata_utils.collect_all_subtypes(graph, type_id_to_node, req_id)
                    if self.type_filter.include_descendants
                    else {req_id}
                )
                required.update(types)
            self.required_entity_types = {type_id_to_node[t]["name"] for t in required}

        if self.type_filter.exclude_type_ids:
            excluded: set[str] = set()
            for t in self.type_filter.exclude_type_ids:
                types = (
                    wikidata_utils.collect_all_subtypes(graph, type_id_to_node, t)
                    if self.type_filter.include_descendants
                    else {t}
                )
                excluded.update(types)
            self.excluded_entity_types = {type_id_to_node[t]["name"] for t in excluded}

        self._logger.info(
            f"Type filters initialized: required={len(self.required_entity_types)}, excluded={len(self.excluded_entity_types)}"
        )

    def run(self) -> None:
        """Build all requested splits in the specified order (Test -> Validation -> Train)."""
        self._logger.info(f"Sampling REBEL split with {len(self.splits)} splits; output_dir={self.output_dir}")

        # Build fragment lookup (used for linking-set expansion; this is based on sampled confusing pairs)
        fragment_lookup = self._build_fragment_lookup()

        # load and optionally type-filter base linking pairs
        pairs, labels = self._load_and_filter_linking_pairs(
            self.base_linking_dataset_path, self.base_linking_ground_truth_path
        )
        self._logger.info(f"Loaded base linking pairs: {len(pairs)}")

        # Track entities used by prior splits to enforce disjointness
        disallowed_entity_ids: set[str] = set()

        for split in self.splits:
            self._logger.info(f"Building split: {split.name}")
            linking_dir = self.output_dir / "linking" / split.name
            linking_dir.mkdir(parents=True, exist_ok=True)
            clustering_dir = self.output_dir / "clustering" / split.name
            clustering_dir.mkdir(parents=True, exist_ok=True)
            entity_gen_dir = self.output_dir / "entity_generation" / split.name
            entity_gen_dir.mkdir(parents=True, exist_ok=True)
            fragment_gen_dir = self.output_dir / "fragment_generation" / split.name
            fragment_gen_dir.mkdir(parents=True, exist_ok=True)
            fragset_gen_dir = self.output_dir / "fragment_set_generation" / split.name
            fragset_gen_dir.mkdir(parents=True, exist_ok=True)

            sampled_pairs, sampled_labels, linking_entity_counter, included_fragment_ids = self._sample_linking_pairs(
                pairs, labels, disallowed_entity_ids, split.linking, self.base_linking_dataset_path
            )

            # write linking outputs
            self._write_linking_outputs(
                linking_dir,
                sampled_pairs,
                sampled_labels,
                linking_entity_counter,
                fragment_lookup,
                build_set_linking_datasets=split.build_set_linking_datasets,
            )

            # Entities used in this split
            entities_to_include = set(linking_entity_counter.keys())

            # derive clustering
            self._derive_clustering(
                clustering_dir,
                entities_to_include,
                included_fragment_ids,
                split.clustering,
            )

            # derive entity generation
            self._derive_entity_generation(
                entity_gen_dir,
                entities_to_include,
            )

            # derive fragment generation
            self._derive_fragment_generation(
                fragment_gen_dir,
                entities_to_include,
            )

            # derive fragment set generation
            if split.build_set_generation_datasets and split.fragment_set_generation is not None:
                self._derive_fragment_set_generation(
                    fragset_gen_dir,
                    entities_to_include,
                    split.fragment_set_generation,
                )

                # negative fragment set generation derived from negative linking pairs
                self._derive_negative_fragment_set_generation(
                    fragset_gen_dir,
                    sampled_pairs,
                    sampled_labels,
                )

            # update disallowed entities for next splits
            disallowed_entity_ids.update(entities_to_include)

    def _load_and_filter_linking_pairs(
        self, ds_path: Path, gt_path: Path
    ) -> tuple[list[tuple[ResolvedWikidataEntity, ResolvedWikidataEntity]], list[bool]]:
        """Load all base linking pairs and ground-truth labels, apply optional type filtering."""
        pairs: list[tuple[ResolvedWikidataEntity, ResolvedWikidataEntity]] = []
        labels: list[bool] = []

        required_types = self.required_entity_types
        excluded_types = self.excluded_entity_types

        id_to_fragment = {}

        with (
            open(resolve_path(ds_path), encoding="utf-8") as f_ds,
            open(resolve_path(gt_path), encoding="utf-8") as f_gt,
        ):
            for i, (line_ds, line_gt) in enumerate(zip(f_ds, f_gt, strict=False)):
                d = json.loads(line_ds)
                left = ResolvedWikidataEntity.from_dict(d[0])
                right = ResolvedWikidataEntity.from_dict(d[1])

                if required_types and (
                    not set(left.wikidata_type or []).intersection(required_types)
                    or not set(right.wikidata_type or []).intersection(required_types)
                ):
                    continue

                if excluded_types and (
                    set(left.wikidata_type or []).intersection(excluded_types)
                    or set(right.wikidata_type or []).intersection(excluded_types)
                ):
                    continue

                # if any type contains word wikimedia - exclude it
                if (left.wikidata_type and any("wikimedia" in t.lower() for t in left.wikidata_type)) or (
                    right.wikidata_type and any("wikimedia" in t.lower() for t in right.wikidata_type)
                ):
                    continue

                if self.ENABLE_FRAGMENT_TRIMMING:

                    def _trim_fragment(fragment: ResolvedWikidataEntity) -> ResolvedWikidataEntity:
                        """Trim fragment to only keep property names (no values) and the fragment id.

                        Also deduplicates fragments by id to save memory.
                        """
                        fragment_id = self._get_fragment_id(fragment)

                        if fragment_id not in id_to_fragment:
                            fragment.metadata = {"id": fragment_id}
                            for p in fragment.properties:
                                fragment.properties[p] = None  # type: ignore

                            id_to_fragment[fragment_id] = fragment

                        return id_to_fragment[fragment_id]

                    # Drop all property values and metadata so that the collection of all fragments fits in memory
                    # more easily. We'll re-attach them later for the sampled pairs only.
                    left = _trim_fragment(left)
                    right = _trim_fragment(right)

                pairs.append((left, right))
                labels.append(bool(json.loads(line_gt)))

                if i > 0 and i % 100_000 == 0:
                    self._logger.info(f"Read {i + 1:,d} pairs, kept {len(pairs):,d}")

        assert len(pairs) == len(labels)
        self._logger.info(
            f"After type filtering: {len(pairs)} pairs kept; positives = {sum(labels)} ({(100.0 * sum(labels) / len(labels)) if pairs else 0.0:.2f}%)"
        )
        return pairs, labels

    @staticmethod
    def _pair_property_overlap(left: ResolvedWikidataEntity, right: ResolvedWikidataEntity) -> int:
        """Count of overlapping property keys between two entities."""
        left_props = set((left.properties or {}).keys())
        right_props = set((right.properties or {}).keys())
        return len(left_props.intersection(right_props))

    def _sample_linking_pairs(
        self,
        pairs: list[tuple[ResolvedWikidataEntity, ResolvedWikidataEntity]],
        labels: list[bool],
        disallowed_entity_ids: set[str],
        cfg: LinkingConfig,
        ds_path: Path,
    ) -> tuple[
        list[tuple[ResolvedWikidataEntity, ResolvedWikidataEntity]],
        list[bool],
        Counter[str],
        set[str],
    ]:
        """Heuristic sampling to produce a more interesting subset."""
        rng = random.Random(cfg.seed)  # noqa: S311 - pseudo-random is fine for dataset sampling

        indices = list(range(len(pairs)))
        rng.shuffle(indices)

        sampled_pairs: list[tuple[ResolvedWikidataEntity, ResolvedWikidataEntity]] = []
        sampled_labels: list[bool] = []

        linking_entity_counter: Counter[str] = Counter()
        property_pattern_counter: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
        property_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()
        overlap_counter: Counter[int] = Counter()
        class_counter: Counter[bool] = Counter()

        included_fragment_ids: set[str] = set()

        id_to_fragment = {}

        iterations = 0
        for i in indices:
            iterations += 1
            if len(sampled_pairs) >= cfg.pair_count_limit:
                break

            left, right = pairs[i]
            label = labels[i]

            if left.entity_id in disallowed_entity_ids or right.entity_id in disallowed_entity_ids:
                continue

            if (
                linking_entity_counter[left.entity_id] >= cfg.pairs_per_entity_limit
                or linking_entity_counter[right.entity_id] >= cfg.pairs_per_entity_limit
            ):
                continue

            left_keys = tuple(sorted((left.properties or {}).keys()))
            right_keys = tuple(sorted((right.properties or {}).keys()))
            prop_pattern: tuple[tuple[str, ...], tuple[str, ...]] = (
                (left_keys, right_keys) if left_keys <= right_keys else (right_keys, left_keys)
            )
            if property_pattern_counter[prop_pattern] >= cfg.property_pattern_count_limit:
                continue

            # property overlap distribution shaping
            overlap_num = self._pair_property_overlap(left, right)
            if cfg.property_overlap_limits:
                allowed_ratio = cfg.property_overlap_limits.get(overlap_num)
                if allowed_ratio is not None and overlap_counter[overlap_num] >= allowed_ratio * cfg.pair_count_limit:
                    continue

            # class balance shaping (limit single class)
            if (
                cfg.single_class_ratio_limit is not None
                and class_counter[label] > cfg.single_class_ratio_limit * cfg.pair_count_limit
            ):
                continue

            # accept the pair
            sampled_pairs.append((left, right))
            sampled_labels.append(label)

            class_counter[label] += 1
            overlap_counter[overlap_num] += 1

            linking_entity_counter[left.entity_id] += 1
            if left.entity_id != right.entity_id:
                linking_entity_counter[right.entity_id] += 1

            property_pattern_counter[prop_pattern] += 1

            for t in set(left.wikidata_type or []).union(right.wikidata_type or []):
                type_counter[t] += 1

            for prop_name in set((left.properties or {}).keys()).union((right.properties or {}).keys()):
                property_counter[prop_name] += 1

            id_to_fragment[left.metadata["id"]] = left
            id_to_fragment[right.metadata["id"]] = right

        # Re-attach the full metadata and properties if trimming was enabled.
        if self.ENABLE_FRAGMENT_TRIMMING and id_to_fragment:
            with (
                open(resolve_path(ds_path), encoding="utf-8") as f_ds,
            ):
                for i, line_ds in enumerate(f_ds):
                    d = json.loads(line_ds)
                    left = ResolvedWikidataEntity.from_dict(d[0])
                    right = ResolvedWikidataEntity.from_dict(d[1])

                    def _try_reattach(fragment: ResolvedWikidataEntity) -> None:
                        """Re-attach full metadata and properties to the fragment if it was sampled."""
                        f_id = self._get_fragment_id(fragment)
                        if f_id in id_to_fragment:
                            f = id_to_fragment[f_id]
                            f.metadata = fragment.metadata
                            f.properties = fragment.properties

                    _try_reattach(left)
                    _try_reattach(right)

                    if i > 0 and i % 100_000 == 0:
                        self._logger.info(f"Reattaching metadata and properties: read {i:,d} pairs")

        for left, right in sampled_pairs:
            assert "fragment_id" in left.metadata or "fragment_ids" in left.metadata
            assert "fragment_id" in right.metadata or "fragment_ids" in right.metadata

            if "fragment_id" in left.metadata:
                included_fragment_ids.add(left.metadata["fragment_id"])
            else:
                included_fragment_ids.update(left.metadata["fragment_ids"])

            if "fragment_id" in right.metadata:
                included_fragment_ids.add(right.metadata["fragment_id"])
            else:
                included_fragment_ids.update(right.metadata["fragment_ids"])

        self._logger.info(
            f"Sampled {len(sampled_pairs)}/{cfg.pair_count_limit} pairs; positives: {sum(sampled_labels)}"
            f" ({(100.0 * sum(sampled_labels) / len(sampled_labels)) if sampled_labels else 0.0:.2f}%);"
            f" iterations: {iterations};"
            f" acceptance: {(100.0 * len(sampled_pairs) / iterations) if iterations else 0.0:.2f}%; entities: {len(linking_entity_counter)}"
        )

        self._logger.info(f"Included fragment ids: {len(included_fragment_ids):,d}")

        self._logger.info("Top 10 types: " + ", ".join(f"{t}:{c}" for t, c in type_counter.most_common(10)))
        self._logger.info("Top 10 properties: " + ", ".join(f"{p}:{c}" for p, c in property_counter.most_common(10)))

        return sampled_pairs, sampled_labels, linking_entity_counter, included_fragment_ids

    def _write_linking_outputs(
        self,
        split_dir: Path,
        sampled_pairs: list[tuple[ResolvedWikidataEntity, ResolvedWikidataEntity]],
        sampled_labels: list[bool],
        linking_entity_counter: Counter[str],
        fragment_lookup: dict[str, dict],
        *,
        build_set_linking_datasets: bool = False,
    ) -> None:
        """Write merged linking dataset plus a set (unmerged) variant."""
        split_dir.mkdir(parents=True, exist_ok=True)
        ds_out = split_dir / RebelPairSampler.LINKING_DATASET_FILENAME
        gt_out = split_dir / RebelPairSampler.LINKING_GROUND_TRUTH_FILENAME
        set_ds_out = split_dir / self.SET_LINKING_DATASET_FILENAME
        set_gt_out = split_dir / self.SET_LINKING_GROUND_TRUTH_FILENAME
        used_ents_out = split_dir / "rebel_linking_used_entities.jsonl"
        pos_gen_out = split_dir / self.LINKING_GENERATION_POSITIVE_DATASET_FILENAME
        neg_gen_out = split_dir / self.LINKING_GENERATION_NEGATIVE_DATASET_FILENAME

        def expand(entity: ResolvedWikidataEntity) -> list[dict]:
            raw_ids = entity.metadata.get("fragment_ids") or [entity.metadata.get("fragment_id")]
            frag_ids = cast(list[str], list(raw_ids))
            expanded: list[dict] = []
            for fid in frag_ids:
                fragment = fragment_lookup[fid]
                expanded.append(fragment)
            return expanded

        with (
            open(ds_out, "w", encoding="utf-8") as f_ds,
            open(pos_gen_out, "w", encoding="utf-8") as f_pos_gen,
            open(neg_gen_out, "w", encoding="utf-8") as f_neg_gen,
        ):
            if build_set_linking_datasets:
                with open(set_ds_out, "w", encoding="utf-8") as f_set_ds:
                    for (left, right), label in zip(sampled_pairs, sampled_labels, strict=False):
                        merged_pair = [left.to_dict(minimal_repr=True), right.to_dict(minimal_repr=True)]
                        # Vanilla linking dataset
                        f_ds.write(json.dumps(merged_pair) + "\n")
                        # Set-variant expansion
                        f_set_ds.write(json.dumps([expand(left), expand(right)]) + "\n")
                        # Generation datasets (positive / negative separation)
                        if label:
                            f_pos_gen.write(json.dumps(merged_pair) + "\n")
                        else:
                            f_neg_gen.write(json.dumps(merged_pair) + "\n")
            else:
                for (left, right), label in zip(sampled_pairs, sampled_labels, strict=False):
                    merged_pair = [left.to_dict(minimal_repr=True), right.to_dict(minimal_repr=True)]
                    # Vanilla linking dataset
                    f_ds.write(json.dumps(merged_pair) + "\n")
                    # Generation datasets (positive / negative separation)
                    if label:
                        f_pos_gen.write(json.dumps(merged_pair) + "\n")
                    else:
                        f_neg_gen.write(json.dumps(merged_pair) + "\n")

        with open(gt_out, "w", encoding="utf-8") as f_gt:
            for label in sampled_labels:
                line = json.dumps(bool(label)) + "\n"
                f_gt.write(line)

        if build_set_linking_datasets:
            with open(set_gt_out, "w", encoding="utf-8") as f_set_gt:
                for label in sampled_labels:
                    f_set_gt.write(json.dumps(bool(label)) + "\n")

        with open(used_ents_out, "w", encoding="utf-8") as f_ents:
            for ent_id in linking_entity_counter:
                f_ents.write(json.dumps(ent_id) + "\n")

        if build_set_linking_datasets:
            self._logger.info(
                "Wrote linking outputs: %s, %s (set variant: %s, %s) generation(+/-): %s, %s entities=%d",
                ds_out,
                gt_out,
                set_ds_out,
                set_gt_out,
                pos_gen_out,
                neg_gen_out,
                len(linking_entity_counter),
            )
        else:
            self._logger.info(
                "Wrote linking outputs: %s, %s generation(+/-): %s, %s entities=%d (set variant disabled)",
                ds_out,
                gt_out,
                pos_gen_out,
                neg_gen_out,
                len(linking_entity_counter),
            )

    def _build_fragment_lookup(self) -> dict[str, dict]:
        """Build lookup of fragment_id -> fragment from base clustering dataset."""
        lookup: dict[str, dict] = {}
        try:
            with open(self.base_clustering_dataset_path, encoding="utf-8") as f_ds:
                for line in f_ds:
                    frag = ResolvedWikidataEntity.from_dict(json.loads(line))
                    fid = frag.metadata.get("fragment_id")
                    if fid and fid not in lookup:
                        lookup[fid] = frag.without_metadata().to_dict(minimal_repr=True)
            self._logger.info(f"Fragment lookup size: {len(lookup)}")
        except FileNotFoundError:
            self._logger.warning("Clustering dataset not found; linking-set dataset will use merged fragments only")
        return lookup

    def _derive_clustering(
        self,
        out_dir: Path,
        entities_to_include: set[str],
        included_fragment_ids: set[str],
        cfg: ClusteringConfig,
    ) -> None:
        """Derive clustering dataset for the split by subsetting the base clustering dataset."""
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_out = out_dir / RebelPairSampler.CLUSTERING_DATASET_FILENAME
        gt_out = out_dir / RebelPairSampler.CLUSTERING_GROUND_TRUTH_FILENAME

        clustering_entities: dict[str, list[ResolvedWikidataEntity]] = defaultdict(list)
        with (
            open(self.base_clustering_dataset_path, encoding="utf-8") as f_ds,
            open(self.base_clustering_ground_truth_path, encoding="utf-8") as f_gt,
        ):
            for line_ds, line_gt in zip(f_ds, f_gt, strict=False):
                fragment = ResolvedWikidataEntity.from_dict(json.loads(line_ds))
                entity_id = json.loads(line_gt)
                if entity_id in entities_to_include:
                    clustering_entities[entity_id].append(fragment)

        records: list[tuple[dict, str]] = []
        entities_included = 0
        for entity_id, fragments in clustering_entities.items():
            if len(fragments) < cfg.min_fragments_per_entity:
                continue

            if 0 < cfg.entity_count_limit <= entities_included:
                break

            # include linking fragments first (if requested)
            if cfg.include_all_linking_fragments_first:
                sorted_fragments = sorted(
                    fragments,
                    key=lambda f: f.metadata.get("fragment_id")
                    or f.metadata.get("fragment_ids", [])[0] not in included_fragment_ids,
                )
            else:
                sorted_fragments = fragments

            selected = sorted_fragments[: cfg.fragments_per_entity_limit]
            records.extend(
                (
                    fragment.to_dict(minimal_repr=True),
                    entity_id,
                )
                for fragment in selected
            )

            entities_included += 1

        # Shuffle the combined list
        rng = random.Random(cfg.seed)  # noqa: S311 - pseudo-random is fine for dataset sampling
        rng.shuffle(records)
        with open(ds_out, "w", encoding="utf-8") as f_ds_out, open(gt_out, "w", encoding="utf-8") as f_gt_out:
            for frag_obj, entity_id in records:
                f_ds_out.write(json.dumps(frag_obj) + "\n")
                f_gt_out.write(json.dumps(entity_id) + "\n")

    def _derive_entity_generation(
        self,
        out_dir: Path,
        entities_to_include: set[str],
    ) -> None:
        """Derive entity generation dataset for the split by subsetting the base entity generation dataset."""
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_out = out_dir / RebelPairSampler.ENTITY_GENERATION_DATASET_FILENAME
        count = 0

        with (
            open(self.base_entity_generation_dataset_path, encoding="utf-8") as f_ds,
            open(ds_out, "w", encoding="utf-8") as f_out,
        ):
            for line in f_ds:
                fragment = ResolvedWikidataEntity.from_dict(json.loads(line))
                entity_id = fragment.entity_id
                if entity_id not in entities_to_include:
                    continue
                f_out.write(json.dumps(fragment.to_dict(minimal_repr=True)) + "\n")
                count += 1

        self._logger.info(f"Entity generation: wrote {count} records to {ds_out}")

    def _derive_fragment_generation(
        self,
        out_dir: Path,
        entities_to_include: set[str],
    ) -> None:
        """Derive fragment generation dataset for the split by subsetting the base fragment generation dataset."""
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_out = out_dir / RebelPairSampler.FRAGMENT_GENERATION_DATASET_FILENAME
        count = 0

        with (
            open(self.base_fragment_generation_dataset_path, encoding="utf-8") as f_ds,
            open(ds_out, "w", encoding="utf-8") as f_out,
        ):
            for line in f_ds:
                fragment = ResolvedWikidataEntity.from_dict(json.loads(line))
                entity_id = fragment.entity_id
                if entity_id not in entities_to_include:
                    continue
                f_out.write(json.dumps(fragment.to_dict(minimal_repr=True)) + "\n")
                count += 1

        self._logger.info(f"Fragment generation: wrote {count} records to {ds_out}")

    def _derive_fragment_set_generation(
        self,
        out_dir: Path,
        entities_to_include: set[str],
        cfg: FragmentSetGenerationConfig,
    ) -> None:
        """Derive fragment set generation dataset using random subset sampling (no cumulative prefixes)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_out = out_dir / self.FRAGMENT_SET_GENERATION_DATASET_FILENAME

        ent_to_fragments: dict[str, list[ResolvedWikidataEntity]] = defaultdict(list)
        with (
            open(self.base_clustering_dataset_path, encoding="utf-8") as f_ds,
            open(self.base_clustering_ground_truth_path, encoding="utf-8") as f_gt,
        ):
            for line_ds, line_gt in zip(f_ds, f_gt, strict=False):
                fragment = ResolvedWikidataEntity.from_dict(json.loads(line_ds))
                entity_id = json.loads(line_gt)
                if entity_id in entities_to_include:
                    ent_to_fragments[entity_id].append(fragment)

        rng = random.Random(cfg.seed)  # noqa: S311 - pseudo-random is fine for dataset sampling
        total_records = 0

        with open(ds_out, "w", encoding="utf-8") as f_out:
            for fragments in ent_to_fragments.values():
                fragment_count = len(fragments)
                max_len = min(cfg.max_fragments_per_entity, fragment_count)
                if max_len == 0:
                    continue

                seq_cap = cfg.sequences_per_entity or fragment_count
                seq_target = min(seq_cap, fragment_count)

                for _ in range(seq_target):
                    seq_len = rng.randint(1, max_len)
                    sampled = rng.sample(fragments, seq_len)
                    rng.shuffle(sampled)
                    frag_list = [f.to_dict(minimal_repr=True) for f in sampled]
                    f_out.write(json.dumps(frag_list) + "\n")
                    total_records += 1

        self._logger.info(
            f"Fragment set generation: wrote {total_records} data points to {ds_out} for {len(ent_to_fragments)} entities"
        )

    def _derive_negative_fragment_set_generation(
        self,
        out_dir: Path,
        sampled_pairs: list[tuple[ResolvedWikidataEntity, ResolvedWikidataEntity]],
        sampled_labels: list[bool],
    ) -> None:
        """Derive negative fragment set generation dataset from negative-labeled linking pairs."""
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_out = out_dir / self.NEGATIVE_FRAGMENT_SET_GENERATION_DATASET_FILENAME

        # Build fragment lookup from clustering dataset
        fragment_id_lookup: dict[str, dict] = {}
        with open(self.base_clustering_dataset_path, encoding="utf-8") as f_ds:
            for line in f_ds:
                frag = ResolvedWikidataEntity.from_dict(json.loads(line))
                fid = frag.metadata.get("fragment_id")
                assert fid not in fragment_id_lookup
                fragment_id_lookup[str(fid)] = frag.without_metadata().to_dict(minimal_repr=True)

        # rng = random.Random(cfg.seed)
        written = 0
        negative_pairs_total = 0

        def collect_fragment_dicts(entity: ResolvedWikidataEntity) -> list[dict]:
            raw_ids = entity.metadata.get("fragment_ids") or [entity.metadata.get("fragment_id")]
            frag_ids = cast(list[str], list(raw_ids))
            frags: list[dict] = []

            for fid in frag_ids:
                fid_str = str(fid)
                frag_obj = fragment_id_lookup.get(fid_str)
                assert frag_obj is not None
                frags.append(frag_obj)

            return frags

        with open(ds_out, "w", encoding="utf-8") as f_out:
            for (left, right), label in zip(sampled_pairs, sampled_labels, strict=False):
                if label:  # only negatives
                    continue

                negative_pairs_total += 1
                left_frags = collect_fragment_dicts(left)
                right_frags = collect_fragment_dicts(right)

                merged = left_frags + right_frags
                # rng.shuffle(merged)
                f_out.write(json.dumps(merged) + "\n")
                written += 1

        self._logger.info(
            "Negative fragment set generation: wrote %d records to %s (from %d sampled negative pairs; lookup size=%d)",
            written,
            ds_out,
            negative_pairs_total,
            len(fragment_id_lookup),
        )

    @classmethod
    def _get_fragment_id(cls, fragment: ResolvedWikidataEntity) -> tuple[int, ...]:
        t = (
            fragment.metadata["fragment_ids"]
            if "fragment_ids" in fragment.metadata
            else [fragment.metadata["fragment_id"]]
        )
        t = tuple(sorted(int(v) for v in t))
        return t


def default_splits(seed: int | None = None) -> list[SplitBuildConfig]:
    """Provide default configs for Test, Validation, Train splits."""
    seed_increment = 0

    def get_next_seed() -> int:
        nonlocal seed_increment
        seed_increment += 1
        return (seed if seed is not None else 0) + seed_increment

    test = SplitBuildConfig(
        name="test",
        linking=LinkingConfig(
            pair_count_limit=5000,
            pairs_per_entity_limit=50,
            property_pattern_count_limit=100,
            property_overlap_limits={1: 0.07},
            single_class_ratio_limit=0.99,  # naturally ~50/50
            seed=get_next_seed(),
        ),
        clustering=ClusteringConfig(
            min_fragments_per_entity=2,
            fragments_per_entity_limit=10,
            seed=get_next_seed(),
        ),
        fragment_set_generation=FragmentSetGenerationConfig(
            seed=get_next_seed(),
        ),
    )

    dev = SplitBuildConfig(
        name="dev",
        linking=LinkingConfig(
            pair_count_limit=5000,
            pairs_per_entity_limit=50,
            property_pattern_count_limit=100,
            property_overlap_limits={1: 0.07},
            single_class_ratio_limit=0.99,  # naturally ~50/50
            seed=get_next_seed(),
        ),
        clustering=ClusteringConfig(
            seed=get_next_seed(),
        ),
        fragment_set_generation=FragmentSetGenerationConfig(
            seed=get_next_seed(),
        ),
    )

    train = SplitBuildConfig(
        name="train",
        linking=LinkingConfig(
            pair_count_limit=5_000_000,
            pairs_per_entity_limit=1_000,
            property_pattern_count_limit=10_000,
            property_overlap_limits={1: 0.55},
            single_class_ratio_limit=0.8,
            seed=get_next_seed(),
        ),
        clustering=ClusteringConfig(
            fragments_per_entity_limit=1,
            seed=get_next_seed(),
        ),
        fragment_set_generation=FragmentSetGenerationConfig(
            max_fragments_per_entity=100,
            seed=get_next_seed(),
        ),
    )

    return [test, dev, train]
