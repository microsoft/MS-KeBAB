# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
from pathlib import Path
from unittest.mock import Mock

from kebab.utils.dataset.wikidata.wikidata_utils import WikidataEntity
from kebab.utils.dataset.wikipedia.wikipedia_text_fetcher import WikipediaTextFetcher
from pytest_mock import MockerFixture


def test_fetch_wikipedia_pages(wikidata_entity: WikidataEntity, mocker: MockerFixture, tmp_path: Path) -> None:
    """Test the fetch_wikipedia_pages method."""
    mock_intro_response = Mock()
    mock_intro_response.json.return_value = {
        "query": {"pages": {"12345": {"pageid": 31, "title": "Test title", "extract": "Test intro text"}}}
    }

    mock_title_response = Mock()
    mock_title_response.json.return_value = {"entities": {"Q31": {"sitelinks": {"enwiki": {"title": "Test title"}}}}}

    def mock_get(_, params, **__):
        if params.get("action") == "wbgetentities":
            return mock_title_response

        if params.get("action") == "query":
            return mock_intro_response

        raise ValueError("Unexpected params")

    mocker.patch("requests.get", side_effect=mock_get)

    test_entities_path = tmp_path / "entities.jsonl"
    test_entities_path.write_text(wikidata_entity.to_json() + "\n", encoding="utf-8")

    # test titles
    output_dir = tmp_path / "output_1"
    fetcher = WikipediaTextFetcher(entities_path=test_entities_path, output_dir=output_dir)
    output_file = fetcher.fetch_wikipedia_texts()

    results = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert len(results) == 1
    assert results[0]["entity_id"] == "Q31"
    assert "Test title" in results[0]["text"]

    # test intros
    output_dir = tmp_path / "output_2"
    fetcher = WikipediaTextFetcher(entities_path=test_entities_path, output_dir=output_dir)
    output_file = fetcher.fetch_wikipedia_texts(fetch_full_intros=True)

    results = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert len(results) == 1
    assert results[0]["entity_id"] == "Q31"
    assert "Test intro text" in results[0]["text"]
