# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the wikidata_hierarchy_WikidataTypeHierarchyExtractor module."""

import json
from pathlib import Path

import pytest
from kebab.utils.dataset.wikidata import wikidata_utils
from kebab.utils.dataset.wikidata.wikidata_type_hierarchy_extractor import WikidataTypeHierarchyExtractor
from pytest_mock import MockerFixture


@pytest.fixture
def wikidata_dump_sample_1_record_file_path() -> Path:
    """Return the path to the input file with 1 sample record from the Wikidata dump."""
    return Path(__file__).parent / "data" / "datasets" / "wikidata" / "sample" / "wikidata_dump_1_record.json"


@pytest.fixture
def wikidata_type_hierarchy_queried_entities_file_path() -> Path:
    """Return the path to the input file with the queried entities for type hierarchy of Wikidata."""
    return (
        Path(__file__).parent
        / "data"
        / "datasets"
        / "wikidata"
        / "sample"
        / "wikidata_type_hierarchy_queried_entities.json"
    )


def test_wikidata_type_hierarchy_extractor(
    wikidata_dump_sample_1_record_file_path: Path,
    wikidata_type_hierarchy_queried_entities_file_path: Path,
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """Test the extraction of simple entities from Wikidata."""
    with open(wikidata_type_hierarchy_queried_entities_file_path, encoding="utf-8") as f:
        queried_entities = json.loads(f.read())

    query_func = wikidata_utils._query_entities_via_api  # noqa: SLF001
    mocker.patch(f"{query_func.__module__}.{query_func.__qualname__}", return_value=queried_entities)

    extractor = WikidataTypeHierarchyExtractor(
        wikidata_json_dump_path=wikidata_dump_sample_1_record_file_path,
        output_dir=tmp_path,
    )

    extractor.run()

    output_file_path = tmp_path / WikidataTypeHierarchyExtractor.WIKIDATA_HIERARCHY_FILENAME
    assert output_file_path.exists()

    with open(output_file_path, encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 3


def test_sort_topologically():
    """Test topological sorting of a graph."""
    # empty graph
    result = WikidataTypeHierarchyExtractor.sort_topologically({})
    assert result == []

    # single node
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    result = WikidataTypeHierarchyExtractor.sort_topologically(graph)
    assert result == ["Q1"]

    # two nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": set()},
    }
    result = WikidataTypeHierarchyExtractor.sort_topologically(graph)
    assert result == ["Q2", "Q1"]

    # cycle - shouldn't fail
    graph = {
        "Q1": {"entity_id": "Q1", "parents": {"Q2"}, "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q1"}},
    }

    result = WikidataTypeHierarchyExtractor.sort_topologically(graph)
    assert result in (["Q1", "Q2"], ["Q2", "Q1"])


def test_collect_cycles():
    """Test cycle detection in a graph."""
    # empty graph
    result = WikidataTypeHierarchyExtractor.collect_cycles({})
    assert result == []

    # graph with a single node with no cycles
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    result = WikidataTypeHierarchyExtractor.collect_cycles(graph)
    assert result == []

    # graph with multiple nodes with no cycles
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": set()},
    }
    result = WikidataTypeHierarchyExtractor.collect_cycles(graph)
    assert result == []

    # graph with a single self cycle
    graph = {
        "Q1": {"entity_id": "Q1", "parents": {"Q1"}, "children": {"Q1"}},
    }
    result = WikidataTypeHierarchyExtractor.collect_cycles(graph)
    assert result == [("Q1", "Q1")]

    # graph with a single cycle
    graph = {
        "Q1": {"entity_id": "Q1", "parents": {"Q2"}, "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q1"}},
    }
    result = WikidataTypeHierarchyExtractor.collect_cycles(graph)
    assert result in ([("Q1", "Q2", "Q1")], [("Q2", "Q1", "Q2")])

    # graph with a root and a single cycle
    graph = {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1"}},
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q1"}},
    }
    result = WikidataTypeHierarchyExtractor.collect_cycles(graph)
    assert result == [("Q1", "Q2", "Q1")]

    # graph with multiple cycles
    graph = {
        "Q1": {"entity_id": "Q1", "parents": {"Q2"}, "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1", "Q3"}, "children": {"Q1", "Q3"}},
        "Q3": {"entity_id": "Q3", "parents": {"Q2"}, "children": {"Q2"}},
    }
    result = WikidataTypeHierarchyExtractor.collect_cycles(graph)
    assert len(result) == 2
    assert ("Q1", "Q2", "Q1") in result or ("Q2", "Q1", "Q2") in result
    assert ("Q2", "Q3", "Q2") in result or ("Q3", "Q2", "Q3") in result


def test_merge_entities():
    """Test merging of entities in a graph."""
    # graph with a single node
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    WikidataTypeHierarchyExtractor.merge_entities(graph, ["Q1"])
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set(), "merged_ids": {"Q1"}}}

    # graph with two separate nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": set()},
        "Q2": {"entity_id": "Q2", "parents": set(), "children": set()},
    }
    WikidataTypeHierarchyExtractor.merge_entities(graph, ["Q1", "Q2"])
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set(), "merged_ids": {"Q1", "Q2"}}}

    # graph with two connected nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": set()},
    }
    WikidataTypeHierarchyExtractor.merge_entities(graph, ["Q1", "Q2"])
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set(), "merged_ids": {"Q1", "Q2"}}}

    # graph with three connected nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q3"}},
        "Q3": {"entity_id": "Q3", "parents": {"Q2"}, "children": set()},
    }
    WikidataTypeHierarchyExtractor.merge_entities(graph, ["Q1", "Q2", "Q3"])
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set(), "merged_ids": {"Q1", "Q2", "Q3"}}}

    # graph with a two node cycle
    graph = {
        "Q1": {"entity_id": "Q1", "parents": {"Q2"}, "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q1"}},
    }
    WikidataTypeHierarchyExtractor.merge_entities(graph, ["Q1", "Q2"])
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set(), "merged_ids": {"Q1", "Q2"}}}

    # graph with a root and then a two node cycle
    graph = {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1"}},
        "Q1": {"entity_id": "Q1", "parents": {"Q0", "Q2"}, "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q1"}},
    }
    WikidataTypeHierarchyExtractor.merge_entities(graph, ["Q1", "Q2"])
    assert graph == {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1"}, "merged_ids": {"Q0"}},
        "Q1": {"entity_id": "Q1", "parents": {"Q0"}, "children": set(), "merged_ids": {"Q1", "Q2"}},
    }

    # graph with a root and a three node clique
    graph = {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1", "Q2", "Q3"}},
        "Q1": {"entity_id": "Q1", "parents": {"Q0", "Q2", "Q3"}, "children": {"Q2", "Q3"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q0", "Q1", "Q3"}, "children": {"Q1", "Q3"}},
        "Q3": {"entity_id": "Q3", "parents": {"Q0", "Q1", "Q2"}, "children": {"Q1", "Q2"}},
    }
    WikidataTypeHierarchyExtractor.merge_entities(graph, ["Q1", "Q2", "Q3"])
    assert graph == {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1"}, "merged_ids": {"Q0"}},
        "Q1": {"entity_id": "Q1", "parents": {"Q0"}, "children": set(), "merged_ids": {"Q1", "Q2", "Q3"}},
    }


def test_merge_cycles():
    """Test merging of cycles in a graph."""
    # graph with no nodes
    result = WikidataTypeHierarchyExtractor.merge_cycles({})
    assert result == []

    # graph with a single node
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    result = WikidataTypeHierarchyExtractor.merge_cycles(graph)
    assert result == []
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}

    # graph with two separate nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": set()},
        "Q2": {"entity_id": "Q2", "parents": set(), "children": set()},
    }
    result = WikidataTypeHierarchyExtractor.merge_cycles(graph)
    assert result == []
    assert graph == {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": set()},
        "Q2": {"entity_id": "Q2", "parents": set(), "children": set()},
    }

    # a graph with a single cycle
    graph = {
        "Q1": {"entity_id": "Q1", "parents": {"Q2"}, "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q1"}},
    }
    result = WikidataTypeHierarchyExtractor.merge_cycles(graph)
    assert len(result) == 1
    assert ("Q1", "Q2", "Q1") in result or ("Q2", "Q1", "Q2") in result
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set(), "merged_ids": {"Q1", "Q2"}}}

    # a graph with a root and two non-overlapping cycles
    graph = {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1", "Q2"}},
        "Q1": {"entity_id": "Q1", "parents": {"Q0", "Q1b"}, "children": {"Q1a"}},
        "Q1a": {"entity_id": "Q1a", "parents": {"Q1"}, "children": {"Q1b"}},
        "Q1b": {"entity_id": "Q1b", "parents": {"Q1a"}, "children": {"Q1"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q0", "Q2x"}, "children": {"Q2x"}},
        "Q2x": {"entity_id": "Q2x", "parents": {"Q2"}, "children": {"Q2"}},
    }
    result = WikidataTypeHierarchyExtractor.merge_cycles(graph)
    assert len(result) == 2
    assert ("Q1", "Q1a", "Q1b", "Q1") in result or ("Q1", "Q1b", "Q1a", "Q1") in result
    assert ("Q2", "Q2x", "Q2") in result or ("Q2", "Q2", "Q2x") in result
    assert graph == {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1", "Q2"}, "merged_ids": {"Q0"}},
        "Q1": {"entity_id": "Q1", "parents": {"Q0"}, "children": set(), "merged_ids": {"Q1", "Q1a", "Q1b"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q0"}, "children": set(), "merged_ids": {"Q2", "Q2x"}},
    }

    # a graph with a root and two overlapping cycles
    graph = {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1"}},
        "Q1": {"entity_id": "Q1", "parents": {"Q0", "Q1b", "Q1y"}, "children": {"Q1a", "Q1x"}},
        "Q1a": {"entity_id": "Q1a", "parents": {"Q1"}, "children": {"Q1b"}},
        "Q1b": {"entity_id": "Q1b", "parents": {"Q1a"}, "children": {"Q1"}},
        "Q1x": {"entity_id": "Q1x", "parents": {"Q1"}, "children": {"Q1y"}},
        "Q1y": {"entity_id": "Q1y", "parents": {"Q1x"}, "children": {"Q1"}},
    }
    result = WikidataTypeHierarchyExtractor.merge_cycles(graph)
    assert len(result) == 2
    assert ("Q1", "Q1a", "Q1b", "Q1") in result or ("Q1", "Q1b", "Q1a", "Q1") in result
    assert ("Q1", "Q1x", "Q1y", "Q1") in result or ("Q1", "Q1y", "Q1x", "Q1") in result
    assert graph == {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1"}, "merged_ids": {"Q0"}},
        "Q1": {
            "entity_id": "Q1",
            "parents": {"Q0"},
            "children": set(),
            "merged_ids": {"Q1", "Q1a", "Q1b", "Q1x", "Q1y"},
        },
    }


def test_remove_grandparent_links():
    """Test removal of grandparent links in a graph."""
    # empty graph
    graph = {}
    result = WikidataTypeHierarchyExtractor.remove_grandparent_links(graph)
    assert result == []
    assert graph == {}

    # single node
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    result = WikidataTypeHierarchyExtractor.remove_grandparent_links(graph)
    assert result == []
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}

    # two nodes without grandparent link
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": set()},
    }
    result = WikidataTypeHierarchyExtractor.remove_grandparent_links(graph)
    assert result == []
    assert graph == {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": set()},
    }

    # grandparent link: (root -> children) Q1 -> Q2 -> Q3, Q1 -> Q3
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2", "Q3"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q3"}},
        "Q3": {"entity_id": "Q3", "parents": {"Q1", "Q2"}, "children": set()},
    }
    result = WikidataTypeHierarchyExtractor.remove_grandparent_links(graph)
    assert result == [("Q3", "Q1")]
    assert graph == {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q3"}},
        "Q3": {"entity_id": "Q3", "parents": {"Q2"}, "children": set()},
    }


def test_add_subgraph_sizes():
    """Test addition of sub-graph sizes and heights to a graph."""
    # a graph with no nodes
    graph = {}
    WikidataTypeHierarchyExtractor.add_subgraph_sizes(graph)
    assert graph == {}

    # a graph with a single node
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    WikidataTypeHierarchyExtractor.add_subgraph_sizes(graph)
    assert graph["Q1"]["subgraph_size"] == 1
    assert graph["Q1"]["subgraph_height"] == 0

    # a graph with two separate nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": set()},
        "Q2": {"entity_id": "Q2", "parents": set(), "children": set()},
    }
    WikidataTypeHierarchyExtractor.add_subgraph_sizes(graph)
    assert graph["Q1"]["subgraph_size"] == 1
    assert graph["Q1"]["subgraph_height"] == 0
    assert graph["Q2"]["subgraph_size"] == 1
    assert graph["Q2"]["subgraph_height"] == 0

    # a graph with two connected nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": set()},
    }
    WikidataTypeHierarchyExtractor.add_subgraph_sizes(graph)
    assert graph["Q1"]["subgraph_size"] == 2
    assert graph["Q1"]["subgraph_height"] == 1
    assert graph["Q2"]["subgraph_size"] == 1
    assert graph["Q2"]["subgraph_height"] == 0

    # a graph with three connected nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q3"}},
        "Q3": {"entity_id": "Q3", "parents": {"Q2"}, "children": set()},
    }
    WikidataTypeHierarchyExtractor.add_subgraph_sizes(graph)
    assert graph["Q1"]["subgraph_size"] == 3
    assert graph["Q1"]["subgraph_height"] == 2
    assert graph["Q2"]["subgraph_size"] == 2
    assert graph["Q2"]["subgraph_height"] == 1
    assert graph["Q3"]["subgraph_size"] == 1
    assert graph["Q3"]["subgraph_height"] == 0

    # a graph with a 4 node diamond
    graph = {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1", "Q2"}},
        "Q1": {"entity_id": "Q1", "parents": {"Q0"}, "children": {"Q3"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q0"}, "children": {"Q3"}},
        "Q3": {"entity_id": "Q3", "parents": {"Q1", "Q2"}, "children": set()},
    }
    WikidataTypeHierarchyExtractor.add_subgraph_sizes(graph)
    assert graph["Q0"]["subgraph_size"] == 4
    assert graph["Q0"]["subgraph_height"] == 2
    assert graph["Q1"]["subgraph_size"] == 2
    assert graph["Q1"]["subgraph_height"] == 1
    assert graph["Q2"]["subgraph_size"] == 2
    assert graph["Q2"]["subgraph_height"] == 1
    assert graph["Q3"]["subgraph_size"] == 1
    assert graph["Q3"]["subgraph_height"] == 0
