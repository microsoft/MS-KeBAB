# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pathlib import Path

import pytest
from kebab.utils.dataset.wikidata import wikidata_utils
from kebab.utils.dataset.wikidata.wikidata_resolver import WikidataResolver
from kebab.utils.dataset.wikidata.wikidata_utils import ResolvedWikidataEntity, WikidataEntity
from pytest_mock import MockerFixture


@pytest.fixture
def rebel_sample_fragments_file_path() -> Path:
    """Path to the sample REBEL fragments input file."""
    return Path(__file__).parent / "data" / "datasets" / "rebel" / "fragments" / "rebel_entity_fragments.jsonl"


@pytest.fixture
def rebel_sample_fragments_queried_entities_file_path() -> Path:
    """Path to the sample REBEL fragments queried entities input file."""
    return Path(__file__).parent / "data" / "datasets" / "rebel" / "fragments" / "queried_resolved_entities.jsonl"


def test_resolve_rebel_fragments(
    rebel_sample_fragments_file_path: Path,
    wikidata_sample_simple_entities_file_path: Path,
    wikidata_full_type_hierarchy_file_path: Path,
    wikidata_full_properties_file_path: Path,
    mocker: MockerFixture,
    rebel_sample_fragments_queried_entities_file_path: Path,
    tmp_path: Path,
) -> None:
    """Test the Wikidata resolver on the REBEL fragments."""
    with open(rebel_sample_fragments_queried_entities_file_path, encoding="utf-8") as f:
        queried_entities = [WikidataEntity.from_json(line) for line in f.readlines()]

    queried_entities = {e.entity_id: e for e in queried_entities}

    query_func = wikidata_utils.collect_wikidata_entities
    mocker.patch(f"{query_func.__module__}.{query_func.__qualname__}", return_value=queried_entities)

    resolver = WikidataResolver(
        entities_path=rebel_sample_fragments_file_path,
        wikidata_simple_entities_path=wikidata_sample_simple_entities_file_path,
        wikidata_properties_path=wikidata_full_properties_file_path,
        wikidata_type_hierarchy_path=wikidata_full_type_hierarchy_file_path,
        resolve_attached_types=True,
        output_dir=tmp_path,
    )

    resolver.run()

    output_file_path = tmp_path / rebel_sample_fragments_file_path.name
    assert output_file_path.exists()

    with open(output_file_path, encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 2
    entity = ResolvedWikidataEntity.from_json(lines[0])
    assert entity.entity_id == "Q223419"
    assert entity.properties["position held"] == ["President of the Philippines"]
    assert entity.wikidata_type == ["human"]
