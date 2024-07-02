# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the wikidata_hierarchy_WikidataHierarchyExtractor module."""

from kebab.dataset.wikidata.wikidata_hierarchy_extractor import WikidataHierarchyExtractor


def test_sort_topologically():
    """Test topological sorting of a graph."""
    # empty graph
    result = WikidataHierarchyExtractor.sort_topologically({})
    assert result == []

    # single node
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    result = WikidataHierarchyExtractor.sort_topologically(graph)
    assert result == ["Q1"]

    # two nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": set()},
    }
    result = WikidataHierarchyExtractor.sort_topologically(graph)
    assert result == ["Q2", "Q1"]

    # cycle - shouldn't fail
    graph = {
        "Q1": {"entity_id": "Q1", "parents": {"Q2"}, "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q1"}},
    }

    result = WikidataHierarchyExtractor.sort_topologically(graph)
    assert result in (["Q1", "Q2"], ["Q2", "Q1"])


def test_collect_cycles():
    """Test cycle detection in a graph."""
    # empty graph
    result = WikidataHierarchyExtractor.collect_cycles({})
    assert result == []

    # graph with a single node with no cycles
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    result = WikidataHierarchyExtractor.collect_cycles(graph)
    assert result == []

    # graph with multiple nodes with no cycles
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": set()},
    }
    result = WikidataHierarchyExtractor.collect_cycles(graph)
    assert result == []

    # graph with a single self cycle
    graph = {
        "Q1": {"entity_id": "Q1", "parents": {"Q1"}, "children": {"Q1"}},
    }
    result = WikidataHierarchyExtractor.collect_cycles(graph)
    assert result == [("Q1", "Q1")]

    # graph with a single cycle
    graph = {
        "Q1": {"entity_id": "Q1", "parents": {"Q2"}, "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q1"}},
    }
    result = WikidataHierarchyExtractor.collect_cycles(graph)
    assert result in ([("Q1", "Q2", "Q1")], [("Q2", "Q1", "Q2")])

    # graph with a root and a single cycle
    graph = {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1"}},
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q1"}},
    }
    result = WikidataHierarchyExtractor.collect_cycles(graph)
    assert result == [("Q1", "Q2", "Q1")]

    # graph with multiple cycles
    graph = {
        "Q1": {"entity_id": "Q1", "parents": {"Q2"}, "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1", "Q3"}, "children": {"Q1", "Q3"}},
        "Q3": {"entity_id": "Q3", "parents": {"Q2"}, "children": {"Q2"}},
    }
    result = WikidataHierarchyExtractor.collect_cycles(graph)
    assert len(result) == 2
    assert ("Q1", "Q2", "Q1") in result or ("Q2", "Q1", "Q2") in result
    assert ("Q2", "Q3", "Q2") in result or ("Q3", "Q2", "Q3") in result


def test_merge_entities():
    """Test merging of entities in a graph."""
    # graph with a single node
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    WikidataHierarchyExtractor.merge_entities(graph, ["Q1"])
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set(), "merged_ids": {"Q1"}}}

    # graph with two separate nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": set()},
        "Q2": {"entity_id": "Q2", "parents": set(), "children": set()},
    }
    WikidataHierarchyExtractor.merge_entities(graph, ["Q1", "Q2"])
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set(), "merged_ids": {"Q1", "Q2"}}}

    # graph with two connected nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": set()},
    }
    WikidataHierarchyExtractor.merge_entities(graph, ["Q1", "Q2"])
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set(), "merged_ids": {"Q1", "Q2"}}}

    # graph with three connected nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q3"}},
        "Q3": {"entity_id": "Q3", "parents": {"Q2"}, "children": set()},
    }
    WikidataHierarchyExtractor.merge_entities(graph, ["Q1", "Q2", "Q3"])
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set(), "merged_ids": {"Q1", "Q2", "Q3"}}}

    # graph with a two node cycle
    graph = {
        "Q1": {"entity_id": "Q1", "parents": {"Q2"}, "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q1"}},
    }
    WikidataHierarchyExtractor.merge_entities(graph, ["Q1", "Q2"])
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set(), "merged_ids": {"Q1", "Q2"}}}

    # graph with a root and then a two node cycle
    graph = {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1"}},
        "Q1": {"entity_id": "Q1", "parents": {"Q0", "Q2"}, "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": {"Q1"}},
    }
    WikidataHierarchyExtractor.merge_entities(graph, ["Q1", "Q2"])
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
    WikidataHierarchyExtractor.merge_entities(graph, ["Q1", "Q2", "Q3"])
    assert graph == {
        "Q0": {"entity_id": "Q0", "parents": set(), "children": {"Q1"}, "merged_ids": {"Q0"}},
        "Q1": {"entity_id": "Q1", "parents": {"Q0"}, "children": set(), "merged_ids": {"Q1", "Q2", "Q3"}},
    }


def test_merge_cycles():
    """Test merging of cycles in a graph."""
    # graph with no nodes
    result = WikidataHierarchyExtractor.merge_cycles({})
    assert result == []

    # graph with a single node
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    result = WikidataHierarchyExtractor.merge_cycles(graph)
    assert result == []
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}

    # graph with two separate nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": set()},
        "Q2": {"entity_id": "Q2", "parents": set(), "children": set()},
    }
    result = WikidataHierarchyExtractor.merge_cycles(graph)
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
    result = WikidataHierarchyExtractor.merge_cycles(graph)
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
    result = WikidataHierarchyExtractor.merge_cycles(graph)
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
    result = WikidataHierarchyExtractor.merge_cycles(graph)
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
    result = WikidataHierarchyExtractor.remove_grandparent_links(graph)
    assert result == []
    assert graph == {}

    # single node
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    result = WikidataHierarchyExtractor.remove_grandparent_links(graph)
    assert result == []
    assert graph == {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}

    # two nodes without grandparent link
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": set()},
    }
    result = WikidataHierarchyExtractor.remove_grandparent_links(graph)
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
    result = WikidataHierarchyExtractor.remove_grandparent_links(graph)
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
    WikidataHierarchyExtractor.add_subgraph_sizes(graph)
    assert graph == {}

    # a graph with a single node
    graph = {"Q1": {"entity_id": "Q1", "parents": set(), "children": set()}}
    WikidataHierarchyExtractor.add_subgraph_sizes(graph)
    assert graph["Q1"]["subgraph_size"] == 1
    assert graph["Q1"]["subgraph_height"] == 0

    # a graph with two separate nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": set()},
        "Q2": {"entity_id": "Q2", "parents": set(), "children": set()},
    }
    WikidataHierarchyExtractor.add_subgraph_sizes(graph)
    assert graph["Q1"]["subgraph_size"] == 1
    assert graph["Q1"]["subgraph_height"] == 0
    assert graph["Q2"]["subgraph_size"] == 1
    assert graph["Q2"]["subgraph_height"] == 0

    # a graph with two connected nodes
    graph = {
        "Q1": {"entity_id": "Q1", "parents": set(), "children": {"Q2"}},
        "Q2": {"entity_id": "Q2", "parents": {"Q1"}, "children": set()},
    }
    WikidataHierarchyExtractor.add_subgraph_sizes(graph)
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
    WikidataHierarchyExtractor.add_subgraph_sizes(graph)
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
    WikidataHierarchyExtractor.add_subgraph_sizes(graph)
    assert graph["Q0"]["subgraph_size"] == 4
    assert graph["Q0"]["subgraph_height"] == 2
    assert graph["Q1"]["subgraph_size"] == 2
    assert graph["Q1"]["subgraph_height"] == 1
    assert graph["Q2"]["subgraph_size"] == 2
    assert graph["Q2"]["subgraph_height"] == 1
    assert graph["Q3"]["subgraph_size"] == 1
    assert graph["Q3"]["subgraph_height"] == 0
