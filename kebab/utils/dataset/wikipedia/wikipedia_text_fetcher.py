# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Download Wikipedia page texts for the given Wikidata entities in a batched-streaming manner via API.
Skip entities that already have Wikipedia pages in the output directory.

Required inputs:
- Wikidata Entities (in WikidataEntity json format)

The workflow is as follows:
1. See if there are any existing Wikipedia pages in the output directory.
2. Query Wikipedia for all entity_ids found in the input entities but not in the existing Wikipedia pages.
3. Save the Wikipedia pages to a JSONL file, incrementally, querying Wikipedia in batches.
"""
# ruff: noqa: S101

from __future__ import annotations

import json
import logging
import pathlib
import random

from tqdm import tqdm

from kebab.utils.dataset.wikidata import wikidata_utils


class WikipediaTextFetcher:
    """Extract Wikipedia pages for the given Wikidata entities."""

    WIKIPEDIA_PAGES_FILENAME = "wikipedia_pages.jsonl"

    def __init__(
        self,
        *,
        entities_path: pathlib.Path | None = None,
        fetch_full_intros: bool = False,
        output_dir: pathlib.Path | None = None,
    ):
        """Initialize the extractor."""
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.entities_path: pathlib.Path | None = entities_path
        self.fetch_full_intros: bool = fetch_full_intros
        self.output_dir: pathlib.Path = output_dir or pathlib.Path.cwd()

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_wikipedia_texts(
        self,
        entities_path: pathlib.Path | None = None,
        fetch_full_intros: bool = False,
    ) -> pathlib.Path:
        """Fetch Wikipedia texts for the given Wikidata entities."""
        entities_path = entities_path or self.entities_path
        if entities_path is None:
            raise ValueError("Path to the entities file is required.")

        fetch_full_intros = fetch_full_intros or self.fetch_full_intros

        output_path = self.output_dir / self.WIKIPEDIA_PAGES_FILENAME

        existing_entities = set()

        if output_path.exists():
            self._logger.info(f"Loading existing Wikipedia pages from {output_path}...")
            with open(output_path, encoding="utf-8") as output_file:
                for line in output_file:
                    entity = json.loads(line)
                    entity_id = entity["entity_id"]
                    existing_entities.add(entity_id)

            self._logger.info(f"Found {len(existing_entities)} existing Wikipedia pages.")

        self._logger.info(f"Loading Wikidata entity IDs from {entities_path}...")
        entity_ids = []
        with open(entities_path, encoding="utf-8") as entities_file:
            for line in entities_file:
                entity = wikidata_utils.WikidataEntity.from_json(line)

                if entity.entity_id in existing_entities:
                    continue

                entity_ids.append(entity.entity_id)

        # shuffle entities to reduce the load on the Wikipedia servers
        random.shuffle(entity_ids)

        self._logger.info(f"Fetching Wikipedia pages for {len(entity_ids)} entities...")

        fetcher = wikidata_utils.get_wikipedia_intros if fetch_full_intros else wikidata_utils.get_wikipedia_titles

        with open(output_path, "a", encoding="utf-8", newline="\n") as output_file:
            for entity_id, text in tqdm(fetcher(entity_ids), total=len(entity_ids)):
                output_file.write(json.dumps({"entity_id": entity_id, "text": text}) + "\n")

        return output_path
