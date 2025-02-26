"""
Build datasets based on REBEL and Wikidata:
- Pairwise linking dataset that contains pairs of entity fragments from REBEL.
- Clustering dataset that contains merged fragments from REBEL.

Required inputs:
- REBEL entity fragments
- Wikidata simple entities
- Wikidata properties
- Wikidata type hierarchy

Workflow:
- Filter fragments based on the number of properties they have, remove names not found on Wikidata, etc.
- Collect "confusing" entities - entities that share at least one name through their fragments.
- Sample pairs of fragments such that each pair comes from entities that are confusing to each other.
- Given all the entity IDs used in the pairwise dataset, take all fragments for these entities and merge to form
  the clustering dataset.
- Write the datasets.

Example output:
TBD.
"""

from __future__ import annotations

import json
import logging
import pathlib
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from kebab.utils.dataset.wikidata.wikidata_utils import ResolvedWikidataEntity


class RebelDatasetBuilder:
    """Build linking and clustering fragment datasets based on REBEL and Wikidata."""

    LINKING_DATASET_FILENAME: str = "rebel_linking_pairs_dataset.jsonl"
    CLUSTERING_DATASET_FILENAME: str = "rebel_clustering_entities_dataset.jsonl"

    def __init__(
        self,
        *,
        fragments_path: pathlib.Path,
        output_dir: pathlib.Path,
        max_count: int | None = None,
    ):
        """Initialize the dataset creator."""
        self._logger: logging.Logger = logging.getLogger(__name__)

        self.fragments_path: pathlib.Path = fragments_path

        self.output_dir: pathlib.Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_count: int | None = max_count

        self.clustering_dataset_output_path = self.output_dir / self.CLUSTERING_DATASET_FILENAME
        self.linking_dataset_output_path = self.output_dir / self.LINKING_DATASET_FILENAME

    def run(self) -> None:
        """Run the dataset building pipeline."""
        # load the input fragments
        fragments = self.load_fragments()

        # get the confusing entities map
        conf_entities_map = self.get_confusing_entities_map(fragments)

        # sample pairs
        pairs = self.sample_pairs(fragments, conf_entities_map, max_count=self.max_count)

        # execute and write the dataset
        self.write_datasets(fragments, pairs)

    def load_fragments(self, fragments_path: pathlib.Path | None = None) -> list[ResolvedWikidataEntity]:
        """Load REBEL entity fragments from the file."""
        fragments_path = fragments_path or self.fragments_path
        fragments = []

        with open(fragments_path, encoding="utf-8") as f:
            for line in f:
                fragment = ResolvedWikidataEntity.from_json(line)
                fragments.append(fragment)

        self._logger.info(f"Loaded {len(fragments):,} fragments")

        return fragments

    @classmethod
    def get_confusing_entities_map(
        cls,
        fragments: list[ResolvedWikidataEntity],
        use_lowercase: bool = False,
    ) -> dict[str, list[str]]:
        """
        Get a dictionary that maps each entity id to all entity ids that are confusing to it.

        Two entities are considered confusing if they share at least one common name within their fragments.
        This relationship is symmetric but not transitive.

        A entity is always confusing to itself.
        """
        logging.info("Building confusing entities map")

        name_to_entity_ids = defaultdict(set)

        # build the dictionary that maps every name occurring in the fragments to the set of entity ids
        # that have that name in their fragments
        for fragment in fragments:
            for name in fragment.names:
                name_to_entity_ids[name.lower() if use_lowercase else name].add(fragment.entity_id)

        # build the entity_id to confusing entity_ids map
        ent_id_to_conf_ent_ids = defaultdict(set)
        for entity_ids in name_to_entity_ids.values():
            for entity_id in entity_ids:
                ent_id_to_conf_ent_ids[entity_id].update(entity_ids)

        ent_id_to_conf_ent_ids = {entity_id: list(values) for entity_id, values in ent_id_to_conf_ent_ids.items()}

        logging.info(f"Built the confusing entities map, keys = {len(ent_id_to_conf_ent_ids):,}")

        return ent_id_to_conf_ent_ids

    @classmethod
    def sample_pairs(
        cls,
        fragments: list[ResolvedWikidataEntity],
        confusing_entities_map: dict[str, list[str]],
        max_count: int | None = None,
        min_acc_rate: float = 0.1,
        min_fragment_acc_rate: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> Iterable[tuple[int, int]]:
        """
        Sample pairs of fragments such that each pair comes from entities that are confusing to each other.

        - A fragment cannot form a pair with itself.
        - Returned pairs are unique, the order within each pair is irrelevant.
        - The set of all possible pairs is not materialised. This is a random sampling that rejects seen pairs
          and uses acceptance rate thresholds to maintain efficiency.
        """
        logging.info("Sampling pairs")

        # we'll operate on numpy arrays
        entity_ids = list(confusing_entities_map.keys())
        entity_id_to_idx = {entity_id: i for i, entity_id in enumerate(entity_ids)}
        fragment_idx_to_entity_idx = {i: entity_id_to_idx[fragment.entity_id] for i, fragment in enumerate(fragments)}

        entity_idx_to_confusing_idx = {
            entity_id_to_idx[entity_id]: np.array([entity_id_to_idx[eid] for eid in entity_ids])
            for entity_id, entity_ids in confusing_entities_map.items()
        }

        logging.info(
            f"Received {len(fragments):,} fragments, confusing entities map contains {len(confusing_entities_map):,} keys"
        )

        entity_idx_to_fragment_idx = defaultdict(list)
        for idx, fragment in enumerate(fragments):
            entity_idx_to_fragment_idx[entity_id_to_idx[fragment.entity_id]].append(idx)

        entity_idx_to_fragment_idx = {idx: np.array(indices) for idx, indices in entity_idx_to_fragment_idx.items()}

        entity_sizes = np.array(
            [len(entity_idx_to_fragment_idx[idx]) for idx in range(len(entity_ids))], dtype=np.float32
        )

        logging.info(f"Largest entity contains {entity_sizes.max():,.0f} fragments")

        # maintain a set of available fragment indices that are (still) likely to form a valid pair
        rng = rng or np.random.default_rng()

        available_fragment_idx = _RandomSet(rng)
        for i in range(len(fragments)):
            available_fragment_idx.add(i)

        acc_rate = _EmaThreshold(decay=0.99, min_value=min_acc_rate, min_count=1000)

        fragment_acc_rates = {
            idx: _EmaThreshold(decay=0.9, min_value=min_fragment_acc_rate, min_count=10)
            for idx in range(len(fragments))
        }

        seen_pairs = set()
        iterations = 0

        while available_fragment_idx.items and not acc_rate.below_threshold():
            left_f_idx = available_fragment_idx.sample()
            entity_idx = fragment_idx_to_entity_idx[left_f_idx]
            ent_indices = entity_idx_to_confusing_idx[entity_idx]
            probs = entity_sizes[ent_indices]
            probs /= probs.sum()

            fragment_ar = fragment_acc_rates[left_f_idx]
            accepted = False

            while not fragment_ar.below_threshold():
                ent_idx = rng.choice(ent_indices, p=probs)
                right_f_idx = rng.choice(entity_idx_to_fragment_idx[ent_idx])
                pair = (left_f_idx, right_f_idx) if left_f_idx < right_f_idx else (right_f_idx, left_f_idx)
                if pair not in seen_pairs and left_f_idx != right_f_idx:
                    yield left_f_idx, right_f_idx
                    accepted = True
                    seen_pairs.add(pair)
                    break

                fragment_ar.update(0)

            if accepted:
                acc_rate.update(1)
                fragment_ar.update(1)
                if max_count and len(seen_pairs) >= max_count:
                    break
            else:
                acc_rate.update(0)
                available_fragment_idx.remove(left_f_idx)

            iterations += 1
            if iterations % 100 == 0:
                logging.info(
                    f"Run for {iterations:,} iterations, acceptance rate = {acc_rate.avg:.2f}, "
                    f"available fragments = {len(available_fragment_idx.items):,}"
                )

        if max_count and len(seen_pairs) >= max_count:
            logging.info(f"Sampling finished - reached {len(seen_pairs):,} pairs")
        elif acc_rate.below_threshold():
            logging.info(f"Sampling finished - acceptance rate {acc_rate.avg:.2f} < {min_acc_rate:.2f}")
        else:
            logging.info(f"Sampling finished - all candidate fragments acceptance rate < {min_fragment_acc_rate:.2f}")

        logging.info(f"Produced {len(seen_pairs):,} pairs of fragments")
        logging.info(f"Final acceptance rate: {acc_rate.avg:.2f}")

    def write_datasets(self, fragments: list[ResolvedWikidataEntity], pairs: Iterable[tuple[int, int]]) -> None:
        """Write the linking and clustering datasets."""
        indices = set()

        # write the linking dataset
        count = 0
        with open(self.linking_dataset_output_path, mode="w", encoding="utf-8") as f:
            for pair in pairs:
                d = {
                    "left": fragments[pair[0]].to_dict(minimal_repr=True),
                    "right": fragments[pair[1]].to_dict(minimal_repr=True),
                }
                f.write(json.dumps(d) + "\n")

                indices.add(pair[0])
                indices.add(pair[1])

                count += 1

        logging.info(f"Wrote {count:,} pairs to {self.linking_dataset_output_path}")

        # write the clustering dataset
        with open(self.clustering_dataset_output_path, mode="w", encoding="utf-8") as f:
            for index in indices:
                f.write(fragments[index].to_json(minimal_repr=True) + "\n")

        logging.info(f"Wrote {len(indices):,} fragments to {self.clustering_dataset_output_path}")


class _RandomSet:
    """A set that supports sampling."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.items: list[int] = []
        self.value_to_idx: dict[int, int] = {}

        self.rng = rng or np.random.default_rng()

    def add(self, val: int) -> bool:
        """Add an element to the set."""
        if val in self.value_to_idx:
            return False

        self.value_to_idx[val] = len(self.items)
        self.items.append(val)
        return True

    def remove(self, val: int) -> bool:
        """Remove an element from the set."""
        if val not in self.value_to_idx:
            return False

        # get the index of the element to delete
        index = self.value_to_idx[val]
        last_element = self.items[-1]

        # swap with last element (if not already last)
        self.items[index] = last_element
        self.value_to_idx[last_element] = index

        # remove last element
        self.items.pop()
        del self.value_to_idx[val]

        return True

    def sample(self) -> int:
        """Return a random sample from the set."""
        if not self.items:
            raise IndexError("Cannot sample from an empty set")

        return self.rng.choice(self.items)


class _EmaThreshold:
    """
    Exponential moving average based threshold that triggers if after a certain number of updates the average
    drops below a certain value.
    """

    def __init__(self, decay: float = 0.999, min_value: float = 0.1, min_count: int = 10):
        self.decay: float = decay
        self.min_value: float = min_value
        self.min_count: int = min_count
        self.n: int = 0
        self.avg: float = 0

    def update(self, x: float) -> None:
        """Update the moving average."""
        self.n += 1
        self.avg = (1 - self.decay) * x + self.decay * self.avg

    def below_threshold(self) -> bool:
        """Check if the threshold is met."""
        return self.n >= self.min_count and self.avg <= self.min_value
