# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Filter entity fragments coming from REBEL dataset to remove entries (values or fragments) that tend to be degenerate.

- Process all fragments sequentially and apply a set of filters.

Example output:
{"entity_id": "Q2038835", "properties": {"name": ["Trinity Peninsula"], "P361": ["Graham Land"], "P30": ["Antarctica"]}, "metadata": {"fragment_id": "1", "title": "Coburg Peak", "entity_id": "Q5139027"}}
{"entity_id": "Q618370", "properties": {"name": ["Graham Land"], "P30": ["Antarctica"]}, "metadata": {"fragment_id": "2", "title": "Coburg Peak", "entity_id": "Q5139027"}}
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from kebab.utils.dataset.wikidata.wikidata_utils import ResolvedWikidataEntity
from kebab.utils.io_helpers import resolve_path


class RebelFragmentFilter:
    """Filter entity fragments that come from REBEL dataset."""

    ENTITY_FRAGMENTS_FILENAME: str = "rebel_entity_fragments.jsonl"

    _logger: logging.Logger
    fragments_path: Path
    output_dir: Path
    drop_fragments_without_type: bool

    def __init__(
        self,
        *,
        fragments_path: Path,
        output_dir: Path,
        drop_fragments_without_type: bool = True,
    ):
        """
        Initialize the fragment filter.

        Args:
            fragments_path: Path to the input REBEL fragments file.
            output_dir: Directory where output datasets will be written.
            drop_fragments_without_type: If True, drop fragments that have no WikiData type in metadata.
                These tend to be incomplete or poorly defined entities.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self.fragments_path = resolve_path(fragments_path)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.drop_fragments_without_type = drop_fragments_without_type

    def run(self) -> None:
        """Run the dataset filter pipeline."""
        fragments_output_path = self.output_dir / self.fragments_path.name
        counter = Counter()

        self._logger.info(f"Filtering fragments from {self.fragments_path}")

        with (
            open(fragments_output_path, mode="w", encoding="utf-8") as f_out,
            open(self.fragments_path, encoding="utf-8") as f_in,
        ):
            for line in f_in:
                fragment = ResolvedWikidataEntity.from_json(line)
                counter["total"] += 1

                if counter["total"] % 100_000 == 0:
                    self._logger.info(f"Processed {counter['total']:,} fragments")

                for values in fragment.properties.values():
                    for value in values:
                        if not value.strip():
                            counter["empty_values"] += 1

                # Remove values that are empty strings and remove properties that have empty values
                fragment.properties = {
                    prop: [value for value in values if value.strip()]
                    for prop, values in fragment.properties.items()
                    if any(value.strip() for value in values)
                }

                if not fragment.properties:
                    counter["empty_properties"] += 1
                    continue

                # Optionally remove fragments that have no WikiData type in the metadata
                if self.drop_fragments_without_type and not fragment.wikidata_type:
                    counter["no_wikidata_types"] += 1
                    continue

                counter["valid"] += 1
                f_out.write(fragment.to_json(minimal_repr=True) + "\n")

        for key, value in counter.items():
            self._logger.info(f"{key.replace('_', ' ').title()}: {value:,}")
