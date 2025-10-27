"""
Create Test/Validation/Train datasets from a base REBEL dataset, focusing on more diverse samples.

Workflow:
- Load the base linking dataset.
- Optionally filter by entity types using the Wikidata type hierarchy.
- Sample a more "interesting" Test split using heuristic limits (per-entity, property-pattern,
  property-overlap distribution, and class balance), then a Validation split that excludes Test entities,
  and finally a Train split that excludes entities from both Test and Validation.
- Derive Clustering and Generation datasets for each split based on the entities used in the split by
  subsetting the corresponding base datasets.

Input layout (produced by RebelPairSampler into input_dir):
- input_dir/linking/base/rebel_linking_dataset.jsonl
- input_dir/linking/base/rebel_linking_ground_truth.jsonl
- input_dir/clustering/base/rebel_clustering_dataset.jsonl
- input_dir/clustering/base/rebel_clustering_ground_truth.jsonl
- input_dir/entity_generation/base/rebel_entity_generation_dataset.jsonl
- input_dir/fragment_generation/base/rebel_fragment_generation_dataset.jsonl

Output layout (within output_dir):
- output_dir/linking/<split>/rebel_linking_dataset.jsonl
- output_dir/linking/<split>/rebel_linking_ground_truth.jsonl
- output_dir/linking/<split>/rebel_linking_used_entities.jsonl
- output_dir/clustering/<split>/rebel_clustering_dataset.jsonl
- output_dir/clustering/<split>/rebel_clustering_ground_truth.jsonl
- output_dir/entity_generation/<split>/rebel_entity_generation_dataset.jsonl
- output_dir/fragment_generation/<split>/rebel_fragment_generation_dataset.jsonl
- output_dir/fragment_set_generation/<split>/rebel_fragment_set_generation_dataset.jsonl
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
    fragment_set_generation: FragmentSetGenerationConfig


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
    """Create Test/Validation/Train datasets from a base REBEL dataset in a single run."""

    SET_LINKING_DATASET_FILENAME: str = "rebel_linking_set_dataset.jsonl"
    SET_LINKING_GROUND_TRUTH_FILENAME: str = "rebel_linking_set_ground_truth.jsonl"
    FRAGMENT_SET_GENERATION_DATASET_FILENAME: str = "rebel_fragment_set_generation_dataset.jsonl"

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
            if split.fragment_set_generation is not None:
                self._derive_fragment_set_generation(
                    fragset_gen_dir,
                    entities_to_include,
                    split.fragment_set_generation,
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

                # attach entity ids (these were removed in base linking dataset)
                try:
                    left.entity_id = left.metadata["entity_id"]
                    right.entity_id = right.metadata["entity_id"]
                except KeyError as ex:
                    self._logger.debug(f"Missing fragment->entity mapping at line {i}: {ex}")
                    continue

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

                f_id = self._get_fragment_id(left)
                if f_id not in id_to_fragment:
                    left.metadata = {"id": f_id}
                    for p in left.properties:
                        left.properties[p] = None  # type: ignore

                    id_to_fragment[f_id] = left

                left = id_to_fragment[f_id]

                f_id = self._get_fragment_id(right)
                if f_id not in id_to_fragment:
                    right.metadata = {"id": f_id}
                    for p in right.properties:
                        right.properties[p] = None  # type: ignore

                    id_to_fragment[f_id] = right

                right = id_to_fragment[f_id]

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

        # Re-attach the full metadata and properties
        with (
            open(resolve_path(ds_path), encoding="utf-8") as f_ds,
        ):
            for i, line_ds in enumerate(f_ds):
                d = json.loads(line_ds)
                left = ResolvedWikidataEntity.from_dict(d[0])
                right = ResolvedWikidataEntity.from_dict(d[1])

                left_id = self._get_fragment_id(left)
                if left_id in id_to_fragment:
                    fragment = id_to_fragment[left_id]
                    fragment.metadata = left.metadata
                    fragment.properties = left.properties

                right_id = self._get_fragment_id(right)
                if right_id in id_to_fragment:
                    fragment = id_to_fragment[right_id]
                    fragment.metadata = right.metadata
                    fragment.properties = right.properties

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
    ) -> None:
        """Write merged linking dataset plus a set (unmerged) variant."""
        split_dir.mkdir(parents=True, exist_ok=True)
        ds_out = split_dir / RebelPairSampler.LINKING_DATASET_FILENAME
        gt_out = split_dir / RebelPairSampler.LINKING_GROUND_TRUTH_FILENAME
        set_ds_out = split_dir / self.SET_LINKING_DATASET_FILENAME
        set_gt_out = split_dir / self.SET_LINKING_GROUND_TRUTH_FILENAME
        used_ents_out = split_dir / "rebel_linking_used_entities.jsonl"

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
            open(set_ds_out, "w", encoding="utf-8") as f_set_ds,
        ):
            for left, right in sampled_pairs:
                merged_pair = [
                    left.without_entity_id().to_dict(minimal_repr=True),
                    right.without_entity_id().to_dict(minimal_repr=True),
                ]
                f_ds.write(json.dumps(merged_pair) + "\n")
                f_set_ds.write(json.dumps([expand(left), expand(right)]) + "\n")

        with (
            open(gt_out, "w", encoding="utf-8") as f_gt,
            open(set_gt_out, "w", encoding="utf-8") as f_set_gt,
        ):
            for label in sampled_labels:
                line = json.dumps(bool(label)) + "\n"
                f_gt.write(line)
                f_set_gt.write(line)

        with open(used_ents_out, "w", encoding="utf-8") as f_ents:
            for ent_id in linking_entity_counter:
                f_ents.write(json.dumps(ent_id) + "\n")

        self._logger.info(
            "Wrote linking outputs: %s, %s (set variant: %s, %s), entities=%d",
            ds_out,
            gt_out,
            set_ds_out,
            set_gt_out,
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
                        lookup[fid] = frag.without_entity_id().without_metadata().to_dict(minimal_repr=True)
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
                    fragment.without_entity_id().to_dict(minimal_repr=True),
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
                f_out.write(json.dumps(fragment.without_entity_id().to_dict(minimal_repr=True)) + "\n")
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
                f_out.write(json.dumps(fragment.without_entity_id().to_dict(minimal_repr=True)) + "\n")
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
                    frag_list = [f.without_entity_id().to_dict(minimal_repr=True) for f in sampled]
                    f_out.write(json.dumps(frag_list) + "\n")
                    total_records += 1

        self._logger.info(
            f"Fragment set generation: wrote {total_records} data points to {ds_out} for {len(ent_to_fragments)} entities"
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
            property_overlap_limits={1: 0.35},
            single_class_ratio_limit=0.76,
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
            property_overlap_limits={1: 0.35},
            single_class_ratio_limit=0.76,
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
            single_class_ratio_limit=0.76,
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
