"""
Build datasets based on REBEL and Wikidata:
- Pairwise linking dataset that contains pairs of entity fragments from REBEL.
- Clustering dataset that contains merged fragments from REBEL.

Required inputs:
- REBEL fragments

Workflow:
- Load all fragment into memory.
- Collect "confusing" entities - entities that share at least one name through their fragments.
- Sample pairs of fragments such that each pair comes from entities that are confusing to each other.
- Given all the entity IDs used in the pairwise dataset, use all fragments of these entities to produce the clustering dataset.
- Write the datasets.

Example output (linking):
{"left": {"entity_id": "Q3490384", "properties": {"name": ["Song for a Blue Guitar"], "performer": ["Red House Painters"]}, "metadata": {"fragment_id": "9623392", "type": ["album"]}}, "right": {"entity_id": "Q3490384", "properties": {"name": ["Songs for a Blue Guitar"]}, "metadata": {"fragment_id": "8314516", "type": ["album"]}}}
{"left": {"entity_id": "Q5520487", "properties": {"name": ["Gananoque River"], "located in the administrative territorial entity": ["Leeds and the Thousand Islands"], "country": ["Canada"]}, "metadata": {"fragment_id": "15937038", "type": ["river"]}}, "right": {"entity_id": "Q5520487", "properties": {"name": ["Gananoque River"]}, "metadata": {"fragment_id": "4126717", "type": ["river"]}}}

Example output (clustering):
{"entity_id": "Q2038835", "properties": {"name": ["Trinity Peninsula"], "part of": ["Graham Land"], "continent": ["Antarctica"]}, "metadata": {"fragment_id": "1", "type": ["peninsula"]}}
{"entity_id": "Q618370", "properties": {"name": ["Graham Land"], "continent": ["Antarctica"]}, "metadata": {"fragment_id": "2", "type": ["geographic region"]}}
"""

from __future__ import annotations

import json
import logging
import pathlib
from collections import defaultdict
from collections.abc import Iterable
from time import perf_counter as pc

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

        self._logger.info("Loading REBEL entity fragments from %s", fragments_path)

        with open(fragments_path, encoding="utf-8") as f:
            for line in f:
                fragment = ResolvedWikidataEntity.from_json(line)

                # discard unused data
                if "doc_id" in fragment.metadata:
                    del fragment.metadata["doc_id"]

                if "source_text_hash" in fragment.metadata:
                    del fragment.metadata["source_text_hash"]

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
        min_acc_rate: float = 0.5,
        min_fragment_attempt_count: int = 3,
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

        logging.info(
            f"Received {len(fragments):,} fragments, confusing entities map contains {len(confusing_entities_map):,} keys"
        )

        rng = rng or np.random.default_rng()

        # we'll keep base collections (fragments, entity_ids) as lists, and maintain index->array(indices) dictionaries
        entity_ids = list(confusing_entities_map.keys())

        # maps entity id to entity index in the flat list
        entity_id_to_ent_index = {entity_id: i for i, entity_id in enumerate(entity_ids)}

        # maps fragment index to its entity index
        fragment_entities = [entity_id_to_ent_index[fragment.entity_id] for i, fragment in enumerate(fragments)]

        # reverse: maps entity index to fragment indices that belong to this entity
        entity_i_to_frag_ixs = defaultdict(list)
        for idx, fragment in enumerate(fragments):
            entity_i_to_frag_ixs[entity_id_to_ent_index[fragment.entity_id]].append(idx)

        entity_fragments = [np.array(entity_i_to_frag_ixs[i]) for i in range(len(entity_ids))]
        del entity_i_to_frag_ixs

        # the list of entity sizes (measured in the number of fragments)
        entity_sizes = np.array([len(entity_fragments[idx]) for idx in range(len(entity_ids))], dtype=np.float32)

        logging.info(f"Largest entity contains {entity_sizes.max():,.0f} fragments")

        # maps entity index to indices of entities that are confusing with it
        confusing_entity_indices = [
            np.array([entity_id_to_ent_index[eid] for eid in confusing_entities_map[entity_id]])
            for entity_id in entity_ids
        ]

        # maps entity index to a numpy array of sizes of its confusing entities
        conf_entity_sizes = [
            entity_sizes[confusing_entity_indices[entity_idx]] for entity_idx in range(len(entity_ids))
        ]

        # maintain a set of available fragment indices that are (still) likely to form an unseen pair
        available_fragment_indices = _RandomSet(rng)
        for i in range(len(fragments)):
            # a fragment from a single-entity that is confusing only with itself can not form a pair
            if conf_entity_sizes[fragment_entities[i]].sum() > 1:
                available_fragment_indices.add(i)

        logging.info(f"Available fragment indices {len(available_fragment_indices.items):,}")

        conf_entity_probs = [arr / arr.sum() for arr in conf_entity_sizes]
        del conf_entity_sizes

        # track overall acceptance rate for diagnostics
        acc_rate = _EmaThreshold(decay=0.99, min_value=0, min_count=0)

        # track each fragment's acceptance rate and exclude it if the AR drops below threshold
        fragment_acc_rates = {
            idx: _EmaThreshold(decay=0.7, min_value=min_acc_rate, min_count=min_fragment_attempt_count)
            for idx in range(len(fragments))
        }

        seen_pairs = set()
        iterations = 0
        start = pc()

        while available_fragment_indices.items:
            # sample left fragment uniformly
            left_f_idx = available_fragment_indices.sample()
            entity_idx = fragment_entities[left_f_idx]
            ent_indices = confusing_entity_indices[entity_idx]

            probs = conf_entity_probs[entity_idx]
            fragment_ar = fragment_acc_rates[left_f_idx]
            accepted = False
            while not fragment_ar.below_threshold():
                # sample the candidate right entity according to their sizes
                ent_idx = rng.choice(ent_indices, p=probs)

                # sample the candidate right fragment uniformly from the selected right entity
                right_f_idx = rng.choice(entity_fragments[ent_idx])

                # accept the pair if unseen
                pair = (left_f_idx, right_f_idx) if left_f_idx < right_f_idx else (right_f_idx, left_f_idx)
                if left_f_idx != right_f_idx and pair not in seen_pairs:
                    accepted = True
                    seen_pairs.add(pair)
                    fragment_ar.update(1)
                    acc_rate.update(1)
                    yield left_f_idx, right_f_idx
                    break

                # mark reject and retry
                fragment_ar.update(0)
                acc_rate.update(0)

            if accepted:
                # stop early if the required number of pairs was reached
                if max_count and len(seen_pairs) >= max_count:
                    break
            else:
                available_fragment_indices.remove(left_f_idx)

            iterations += 1
            if iterations % 10_000 == 0:
                logging.info(
                    f"Run for {iterations:,} iterations"
                    f", returned {len(seen_pairs):,} pairs"
                    f", acceptance rate = {acc_rate.avg:.2f}"
                    f", available fragments = {len(available_fragment_indices.items):,}"
                )

        if max_count and len(seen_pairs) >= max_count:
            logging.info(f"Sampling finished - reached {len(seen_pairs):,} pairs")
        elif acc_rate.below_threshold():
            logging.info(f"Sampling finished - acceptance rate {acc_rate.avg:.2f} < {min_acc_rate:.2f}")
        else:
            logging.info(f"Sampling finished - all candidate fragments acceptance rate < {min_acc_rate:.2f}")

        logging.info(f"Produced {len(seen_pairs):,} pairs of fragments")
        logging.info(
            f"Final acceptance rate: {acc_rate.avg:.2f}, speed = {len(seen_pairs) / (pc() - start):,.2f} pairs/second"
        )

    def write_datasets(self, fragments: list[ResolvedWikidataEntity], pairs: Iterable[tuple[int, int]]) -> None:
        """Write the linking and clustering datasets."""
        entity_ids = set()

        # write the linking dataset
        count = 0
        with open(self.linking_dataset_output_path, mode="w", encoding="utf-8") as f:
            for pair in pairs:
                d = {
                    "left": fragments[pair[0]].to_dict(minimal_repr=True),
                    "right": fragments[pair[1]].to_dict(minimal_repr=True),
                }
                f.write(json.dumps(d) + "\n")

                entity_ids.add(fragments[pair[0]].entity_id)
                entity_ids.add(fragments[pair[1]].entity_id)

                count += 1

        logging.info(f"Wrote {count:,} pairs to {self.linking_dataset_output_path}")

        # write the clustering dataset
        count = 0
        with open(self.clustering_dataset_output_path, mode="w", encoding="utf-8") as f:
            for fragment in fragments:
                if fragment.entity_id not in entity_ids:
                    continue

                f.write(fragment.to_json(minimal_repr=True) + "\n")
                count += 1

        logging.info(
            f"Wrote {count:,} fragments from {len(entity_ids):,} entities to {self.clustering_dataset_output_path}"
        )


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

        idx = self.rng.integers(0, len(self.items))
        return self.items[idx]


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
        self.avg = ((1 - self.decay) * x + self.decay * self.avg) if self.n > 1 else x

    def below_threshold(self) -> bool:
        """Check if the threshold is met."""
        return self.n >= self.min_count and self.avg <= self.min_value
