# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Extract the text paragraph and corresponding to it entities from Re-DocRED dataset.
The dataset can be found here: https://github.com/tonytan48/Re-DocRED/tree/main.

Example output:
"""


from __future__ import annotations

import logging
import typing
from pathlib import Path


class ReDocRedEntityExtractor:
    """Extract text paragraphs and entities from Re-DocRED dataset."""

    ENTITY_PREFIXES: typing.ClassVar[set[str]] = {"http://www.wikidata.org/entity/"}
    PROPERTY_PREFIXES: typing.ClassVar[set[str]] = {"http://www.wikidata.org/prop/direct/"}

    DATETIME_TYPE_SUFFIX: typing.ClassVar[str] = "^^http://www.w3.org/2001/XMLSchema#dateTime"

    def __init__(
        self,
        *,
        dataset_dir: Path,
        path_to_wikidata_properties: Path,
        output_dir: Path,
    ) -> None:
        """
        Initialize the dataset extractor.

        Args:
            dataset_dir: Path to the Re-DocRED data directory.
            path_to_wikidata_properties: Path to the Wikidata properties file.
            output_dir: Path to the output directory ("all_entities_fragments.jsonl" will be created there).
        """
        self._logger: logging.Logger = logging.getLogger(__name__)

        self.data_dir: Path = dataset_dir
        self.path_to_wikidata_properties: Path = path_to_wikidata_properties
        self.output_dir: Path = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_ent_fragments_output_path = self.output_dir / "all_entities_fragments.jsonl"
        assert not self.all_ent_fragments_output_path.exists(), "Output file already exists."

        self.wikidata_properties: dict[str, dict] = {}

    def run(self) -> None:
        """Run the fragment extraction process."""
        # extract entity fragments

