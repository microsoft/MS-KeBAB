import bisect
import itertools
import json
import time
from datetime import timedelta
from itertools import islice
from pathlib import Path
from typing import cast

import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.connectivity import (
    build_auxiliary_node_connectivity,
    minimum_st_node_cut,
)
from networkx.algorithms.flow import build_residual_network

from kebab import mskebab
from kebab.contracts.entity import Entity
from kebab.tasks.linking import LinkingTask


def main() -> None:
    """Compute a new train-test split of linking pairs."""
    dataset_folder = Path(r"C:\Users\minka\Microsoft\Project Alexandria - Documents\Benchmark\Datasets")
    benchmark = mskebab.Benchmark(root_for_relative_paths=dataset_folder)
    task_instance = cast(LinkingTask, benchmark.tasks_by_name["Linking-REBEL-Test"])
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_list(e: Entity | list[Entity]) -> list[Entity]:
        """Ensure we have a list of entities."""
        return e if isinstance(e, list) else [e]

    def get_values(e: Entity | list[Entity]) -> set[str]:
        """Get all values of all properties from an entity or list of entities."""
        return {value for ent in make_list(e) for values in ent.properties.values() for value in values}

    def get_properties(e: Entity | list[Entity]) -> set[str]:
        """Get all property names from an entity or list of entities."""
        return {prop for ent in make_list(e) for prop in ent.properties}

    val_instance = benchmark.tasks_by_name["Linking-REBEL-Validation"]
    training_instance = benchmark.tasks_by_name["Linking-REBEL-Train"]
    print("Reading training items...(this takes about 1 minute)")
    training_instances = list(itertools.islice(training_instance.read_items(), 10000000))
    print(f"Read {len(training_instances)} training values.")
    test_instances = list(itertools.islice(task_instance.read_items(), 10000))
    print(f"Read {len(test_instances)} testing values.")
    val_instances = list(itertools.islice(val_instance.read_items(), 10000))
    print(f"Read {len(val_instances)} validation values.")

    class Stats:
        def __init__(
            self,
            true_count: int,
            false_count: int,
            single_overlap_count: int,
            multiple_overlap_count: int,
            multiple_overlap_false_count: int,
        ):
            self.true_count = true_count
            self.false_count = false_count
            self.single_overlap_count = single_overlap_count
            self.multiple_overlap_count = multiple_overlap_count
            self.multiple_overlap_false_count = multiple_overlap_false_count

    def get_data_stats(
        instances: list[tuple[tuple[Entity | list[Entity], Entity | list[Entity]], bool | None]], name: str
    ) -> Stats:
        """Print out statistics about the given set of instances."""
        true_count = sum(1 for _, label in instances if label)
        false_count = len(instances) - true_count
        print(f"{name} set has {true_count} positive examples and {false_count} negative examples.")
        single_overlap_count = sum(
            1 for pair, _ in instances if len(get_properties(pair[0]).intersection(get_properties(pair[1]))) == 1
        )
        multiple_overlap_count = len(instances) - single_overlap_count
        print(
            f"{name} set has {single_overlap_count} single-overlap examples and {multiple_overlap_count} multiple-overlap examples."
        )
        multiple_overlap_false_count = sum(
            1
            for pair, label in instances
            if not label and len(get_properties(pair[0]).intersection(get_properties(pair[1]))) > 1
        )
        print(f"{name} set has {multiple_overlap_false_count} multiple-overlap negative examples.")
        return Stats(
            true_count, false_count, single_overlap_count, multiple_overlap_count, multiple_overlap_false_count
        )

    test_stats = get_data_stats(test_instances, "Original test")
    val_stats = get_data_stats(val_instances, "Original validation")

    instances = training_instances + test_instances + val_instances

    # Construct a bipartite graph: instances connected to their values
    graph = nx.Graph()

    # Add all instance nodes
    instance_nodes = {str(idx) for idx in range(len(instances))}
    graph.add_nodes_from(instance_nodes, bipartite=0)

    true_nodes = set()
    single_overlap_nodes = set()
    multi_overlap_false_nodes = set()
    for idx, (pair, label) in enumerate(instances):
        if idx > 0 and idx % 100000 == 0:
            print(f"Processing instance {idx} / {len(instances)}")
        values = get_values(pair[0]).union(get_values(pair[1]))
        shared_properties = get_properties(pair[0]).intersection(get_properties(pair[1]))
        instance_node = str(idx)
        if label:
            true_nodes.add(instance_node)
        if len(shared_properties) == 1:
            single_overlap_nodes.add(instance_node)
        if not label and len(shared_properties) > 1:
            multi_overlap_false_nodes.add(instance_node)

        # Connect instance to each of its values
        for value in values:
            value_node = f":{value}"
            graph.add_node(value_node, bipartite=1)
            graph.add_edge(instance_node, value_node)
    print(f"Constructed bipartite graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    # Verify it's actually bipartite
    if not bipartite.is_bipartite(graph):
        raise ValueError("Graph is not bipartite - this indicates a bug in construction")

    def split(graph: nx.Graph, stats: Stats) -> tuple[set, set]:
        predicates = [
            Predicate("Label", true_nodes, stats.true_count, stats.false_count),
            Predicate("SingleOverlap", single_overlap_nodes, stats.single_overlap_count, stats.multiple_overlap_count),
            Predicate("MultipleOverlapFalse", multi_overlap_false_nodes, stats.multiple_overlap_false_count, 0),
        ]
        return train_test_split(graph, instance_nodes, predicates)

    training_nodes, test_nodes = split(graph, test_stats)

    test_indices = {int(node) for node in test_nodes}
    # Write out the new test set
    new_test = [instances[i] for i in sorted(test_indices)]
    get_data_stats(new_test, "New test")
    write_instances(new_test, output_dir / "test")
    print(f"Wrote test set with {len(test_indices)} instances.")

    # Further split the new training set to make a new validation set
    training_graph = graph.subgraph(training_nodes).copy()
    training_nodes, validation_nodes = split(training_graph, val_stats)
    training_indices = {int(node) for node in training_nodes if node in instance_nodes}
    validation_indices = {int(node) for node in validation_nodes}
    new_train = [instances[i] for i in sorted(training_indices)]
    write_instances(new_train, output_dir / "train")
    new_validation = [instances[i] for i in sorted(validation_indices)]
    write_instances(new_validation, output_dir / "dev")
    print(
        f"Wrote training set with {len(training_indices)} instances, validation set with {len(validation_indices)} instances."
    )


def write_instances(instances: list[tuple[tuple[Entity, Entity], bool]], output_dir: Path) -> None:
    """Write out linking instances to jsonl files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        open(output_dir / "rebel_linking_dataset.jsonl", "w", encoding="utf-8") as pair_file,
        open(output_dir / "rebel_linking_ground_truth.jsonl", "w", encoding="utf-8") as label_file,
    ):
        for pair, label in instances:
            merged_pair = [
                pair[0].to_dict(minimal_repr=True),
                pair[1].to_dict(minimal_repr=True),
            ]
            pair_file.write(json.dumps(merged_pair) + "\n")
            label_file.write(json.dumps(bool(label)) + "\n")


def test_train_test_split():
    """Test the train_test_split function with a sample graph."""
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
            ("d", "e"),
            ("c", "e"),
            ("e", "f"),
            ("d", "f"),
            ("f", "g"),
            ("g", "h"),
            ("g", "i"),
            ("g", "j"),
        ],
    )
    nodes = set(graph.nodes())
    train, test = train_test_split(graph, nodes, [Predicate("SamplePredicate", {"a", "c", "e", "g", "i"}, 1, 1)])
    print("Training set:", train)
    print("Test set:", test)
    removed = nodes - train - test
    print("Removed nodes:", removed)


class Predicate:
    """Represents a predicate that we want to maintain a certain number of true and false instances for in the test set."""

    name: str
    true_nodes: set[str]
    min_true_count: int
    min_false_count: int

    def __init__(self, name: str, true_nodes: set[str], min_true_count: int, min_false_count: int) -> None:
        """true_nodes can be a superset of the graph nodes.  min_true_count and min_false_count are the minimum number of true and false instances that must be in the test set."""
        self.name = name
        self.true_nodes = true_nodes
        self.min_true_count = min_true_count
        self.min_false_count = min_false_count

    def __repr__(self) -> str:
        """Return a string representation of the predicate."""
        return self.name

    def get_counts(self, nodes: set[str]) -> tuple[int, int]:
        """Get the number of true and false instances in the given set of instance nodes."""
        true_count = sum(1 for node in nodes if node in self.true_nodes)
        false_count = len(nodes) - true_count
        return true_count, false_count


def train_test_split(
    graph: nx.Graph,
    instance_nodes: set[str],
    predicates: list[Predicate],
) -> tuple[set, set]:
    """Partition the graph into training and test set using minimum node cut.  Graph is modified.  Returns training set and test set.
    instance_nodes is the set of nodes corresponding to instances that we want to split, and predicates is the list of predicates we want to maintain a certain number of true and false instances for in the test set.
    instance_nodes can be a superset of the graph nodes.
    """
    # The largest connected component is always in the training set.
    # If there are too few remaining nodes, we cut the largest connected component and try again.
    components = nx.connected_components(graph)
    component_graphs_with_sizes = sorted(
        [
            (
                graph.subgraph(c),
                len(nodes := {node for node in c if node in instance_nodes}),
                {predicate: predicate.get_counts(nodes) for predicate in predicates},
            )
            for c in components
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    remaining_count = sum(size[1] for size in component_graphs_with_sizes[1:])
    remaining_true_count = {
        predicate: sum(size[2][predicate][0] for size in component_graphs_with_sizes[1:]) for predicate in predicates
    }
    remaining_false_count = {predicate: remaining_count - remaining_true_count[predicate] for predicate in predicates}
    print("Initial remaining counts:", remaining_count, remaining_true_count, remaining_false_count)
    while any(
        remaining_true_count[predicate] < predicate.min_true_count
        or remaining_false_count[predicate] < predicate.min_false_count
        for predicate in predicates
    ):
        print(
            f"Largest connected component has size {(component_graphs_with_sizes[0][1], component_graphs_with_sizes[0][2])}. Computing minimum node cut..."
        )
        largest_cc_graph = component_graphs_with_sizes[0][0]
        # removed_from_largest = minimum_node_cut(largest_cc_graph, flow_func=shortest_augmenting_path)
        reduced_predicates = [
            Predicate(
                predicate.name,
                predicate.true_nodes,
                max(0, predicate.min_true_count - remaining_true_count[predicate]),
                max(0, predicate.min_false_count - remaining_false_count[predicate]),
            )
            for predicate in predicates
        ]
        removed_from_largest = approximate_minimum_node_cut(largest_cc_graph, instance_nodes, reduced_predicates)
        if len(removed_from_largest) == 0:
            raise ValueError("Cannot further split the largest connected component.")
        print(f"Removing {len(removed_from_largest)} nodes from largest connected component")
        graph.remove_nodes_from(removed_from_largest)
        largest_cc_graph = graph.subgraph(largest_cc_graph.nodes() - removed_from_largest)
        # Remove components[0]
        component_graphs_with_sizes = component_graphs_with_sizes[1:]
        # Add new components from largest_cc_graph
        new_components = nx.connected_components(largest_cc_graph)
        new_component_graphs_with_sizes = [
            (
                largest_cc_graph.subgraph(c),
                len(nodes := {node for node in c if node in instance_nodes}),
                {predicate: predicate.get_counts(nodes) for predicate in predicates},
            )
            for c in new_components
        ]
        print(
            f"Split largest connected component into {list(islice((size[1] for size in new_component_graphs_with_sizes if size[1] > 0), 100))}."
        )
        component_graphs_with_sizes = sorted(
            new_component_graphs_with_sizes + component_graphs_with_sizes, key=lambda x: x[1], reverse=True
        )
        remaining_count = sum(size[1] for size in component_graphs_with_sizes[1:])
        remaining_true_count = {
            predicate: sum(size[2][predicate][0] for size in component_graphs_with_sizes[1:])
            for predicate in predicates
        }
        remaining_false_count = {
            predicate: remaining_count - remaining_true_count[predicate] for predicate in predicates
        }
        print("Updated remaining counts:", remaining_count, remaining_true_count, remaining_false_count)
    # Allocate non-maximal components greedily, starting with the largest since this makes the greedy heuristic more effective
    max_training_true_count = {
        predicate: max(0, remaining_true_count[predicate] - predicate.min_true_count) for predicate in predicates
    }
    max_training_false_count = {
        predicate: max(0, remaining_false_count[predicate] - predicate.min_false_count) for predicate in predicates
    }
    training_components = [0]
    testing_components = []
    # print("component sizes:", [(size[1], size[2]) for size in component_graphs_with_sizes if size[1] > 0])
    for i in range(1, len(component_graphs_with_sizes)):
        new_max_training_true_count = {
            predicate: max_training_true_count[predicate] - component_graphs_with_sizes[i][2][predicate][0]
            for predicate in predicates
        }
        new_max_training_false_count = {
            predicate: max_training_false_count[predicate] - component_graphs_with_sizes[i][2][predicate][1]
            for predicate in predicates
        }
        if all(
            new_max_training_true_count[predicate] >= 0 and new_max_training_false_count[predicate] >= 0
            for predicate in predicates
        ):
            training_components.append(i)
            max_training_true_count = new_max_training_true_count
            max_training_false_count = new_max_training_false_count
        else:
            testing_components.append(i)
    training_set = {
        node
        for i in training_components
        for node in component_graphs_with_sizes[i][0].nodes()
        # if node in instance_nodes
    }
    test_set = {
        node for i in testing_components for node in component_graphs_with_sizes[i][0].nodes() if node in instance_nodes
    }
    return training_set, test_set


def minimum_node_cut(G, s=None, t=None, flow_func=None, approximate=False) -> set:
    r"""Returns a set of nodes of minimum cardinality that disconnects G.

    If source and target nodes are provided, this function returns the
    set of nodes of minimum cardinality that, if removed, would destroy
    all paths among source and target in G. If not, it returns a set
    of nodes of minimum cardinality that disconnects G.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node. Optional. Default value: None.

    t : node
        Target node. Optional. Default value: None.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The
        choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    Returns:
    -------
    cutset : set
        Set of nodes that, if removed, would disconnect G. If source
        and target nodes are provided, the set contains the nodes that
        if removed, would destroy all paths between source and target.

    Examples:
    --------
    >>> # Platonic icosahedral graph has node connectivity 5
    >>> G = nx.icosahedral_graph()
    >>> node_cut = nx.minimum_node_cut(G)
    >>> len(node_cut)
    5

    You can use alternative flow algorithms for the underlying maximum
    flow computation. In dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better
    than the default :meth:`edmonds_karp`, which is faster for
    sparse networks with highly skewed degree distributions. Alternative
    flow functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> node_cut == nx.minimum_node_cut(G, flow_func=shortest_augmenting_path)
    True

    If you specify a pair of nodes (source and target) as parameters,
    this function returns a local st node cut.

    >>> len(nx.minimum_node_cut(G, 3, 7))
    5

    If you need to perform several local st cuts among different
    pairs of nodes on the same graph, it is recommended that you reuse
    the data structures used in the maximum flow computations. See
    :meth:`minimum_st_node_cut` for details.

    Notes:
    -----
    This is a flow based implementation of minimum node cut. The algorithm
    is based in solving a number of maximum flow computations to determine
    the capacity of the minimum cut on an auxiliary directed network that
    corresponds to the minimum node cut of G. It handles both directed
    and undirected graphs. This implementation is based on algorithm 11
    in [1]_.

    See Also:
    --------
    :meth:`minimum_st_node_cut`
    :meth:`minimum_cut`
    :meth:`minimum_edge_cut`
    :meth:`stoer_wagner`
    :meth:`node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References:
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    if (s is not None and t is None) or (s is None and t is not None):
        raise nx.NetworkXError("Both source and target must be specified.")

    # Local minimum node cut.
    if s is not None and t is not None:
        if s not in G:
            raise nx.NetworkXError(f"node {s} not in graph")
        if t not in G:
            raise nx.NetworkXError(f"node {t} not in graph")
        return minimum_st_node_cut(G, s, t, flow_func=flow_func)

    # Global minimum node cut.
    # Analog to the algorithm 11 for global node connectivity in [1].
    if G.is_directed():
        if not nx.is_weakly_connected(G):
            raise nx.NetworkXError("Input graph is not connected")
        iter_func = itertools.permutations
        smallest_possible_cut_size = 1

        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors(v), G.successors(v)])

    else:
        if not nx.is_connected(G):
            raise nx.NetworkXError("Input graph is not connected")
        iter_func = itertools.combinations
        neighbors = G.neighbors
        # If the min cut has size 1, it is an articulation point.
        for min_cut in nx.articulation_points(G):
            return {min_cut}
        # If we reach this line, we know the min cut has size at least 2.
        smallest_possible_cut_size = 2

    # Reuse the auxiliary digraph and the residual network.
    H = build_auxiliary_node_connectivity(G)
    R = build_residual_network(H, "capacity")
    kwargs = {"flow_func": flow_func, "auxiliary": H, "residual": R}

    # Choose a node with minimum degree.
    v = min(G, key=G.degree)
    # Initial node cutset is all neighbors of the node with minimum degree.
    min_cut = set(G[v])
    if len(min_cut) == smallest_possible_cut_size:
        return min_cut
    # Compute st node cuts between v and all its non-neighbors nodes in G.
    for w in set(G) - set(neighbors(v)) - {v}:
        this_cut = minimum_st_node_cut(G, v, w, **kwargs)
        if len(min_cut) >= len(this_cut):
            min_cut = this_cut
            if len(min_cut) == smallest_possible_cut_size or approximate:
                return min_cut
    # Also for non adjacent pairs of neighbors of v.
    for x, y in iter_func(neighbors(v), 2):
        if y in G[x]:
            continue
        this_cut = minimum_st_node_cut(G, x, y, **kwargs)
        if len(min_cut) >= len(this_cut):
            min_cut = this_cut
            if len(min_cut) == smallest_possible_cut_size or approximate:
                return min_cut

    return min_cut


def approximate_minimum_node_cut(
    graph: nx.Graph,
    instance_nodes: set[str],
    predicates: list[Predicate],
) -> set:
    """Returns an approximate minimum node cut that splits G into two groups of at least min_group_size. instance_nodes can be a superset of the graph nodes."""
    # The algorithm is inspired by Tarjan's DFS algorithm for finding articulation points.
    # We use depth-first search to convert the graph into a tree with back edges.
    # Each node can be considered the root of a subtree, and all back edges that go out of the subtree (excluding the node itself) are called subtree back edges.
    # If the node is not the DFS root and there are no subtree back edges, then the node is an articulation point and its removal cuts the graph into two or more groups.
    # Otherwise if the node is not the DFS root, removing the node and the source or target of each subtree back edge creates a cut.
    # We only consider cuts consisting of instance nodes.
    # Given a cut, we consider the largest connected component that results from removing the cut, and let all remaining nodes form the "smaller group".
    # Out of every cut that can be constructed in this way, we find the smallest one whose smaller group satisfies min_true_count and min_false_count.
    # If there is no such cut, we find the one that maximizes the smaller group size.

    # Some facts about minimal cuts (cut that cannot be made smaller by removing nodes from the cut):
    # 1. One cut node must be an ancestor of all other cut nodes in the DFS tree.  Otherwise, the cut nodes would form two groups where each group can reach the DFS root without passing through the other group, so removing one group from the cut would still disconnect the graph.
    # 2. If the cut nodes form an ancestor chain in the DFS tree, then they cut this chain into blocks.  Adjacent blocks must be disconnected by the cut, otherwise the cut node between them would be redundant.  For the same reason, the blocks must form exactly two connected components after the cut.

    # Make a copy of the predicates since they are modified below
    predicates = [
        Predicate(predicate.name, predicate.true_nodes, predicate.min_true_count, predicate.min_false_count)
        for predicate in predicates
    ]

    # Single-pass DFS with efficient incremental back edge counting
    # Start at a non-instance node if possible
    root = max(
        (node for node in graph if node not in instance_nodes), key=lambda node: graph.degree[node], default=None
    )
    if root is None:
        root = max(graph.nodes(), key=lambda node: graph.degree[node])
    print("Starting DFS at root node:", root)
    discovery = {root: 0}  # time of first discovery of node during search
    discovered_nodes = [root]  # nodes in order of discovery
    finish = {}  # time of finishing exploration of node
    back_edge_target = {root: root}  # earliest target of the back edges out of node (only updated for instance nodes)
    back_edge_source = {}  # latest source of the back edges into node
    # Number of back edges out of subtree rooted at node (excluding node itself)
    # To make this reflect the size of the cut:
    # when a back edge goes from an instance node to a non-instance node, we only count the back edge with the earliest non-instance node.
    # when a back edge goes from a non-instance node to an instance node, we only count the back edge with the latest non-instance node.
    subtree_back_edge_count = {root: 0}
    subtree_back_edge_from_instance_count = {
        root: 0
    }  # number of back edges out of subtree rooted at node from instance nodes (excluding node itself)
    subtree_back_edge_from_predicate_count = {
        root: dict.fromkeys(predicates, 0)
    }  # number of back edges out of subtree rooted at node from instance nodes that are true for predicate (excluding node itself)
    subtree_size = {
        root: 1 if root in instance_nodes else 0
    }  # number of instances in subtree rooted at node (including node itself)
    subtree_size_by_predicate = {
        root: {predicate: 1 if root in predicate.true_nodes else 0 for predicate in predicates}
    }  # number of instances in subtree rooted at node that are true for predicate (including node itself)
    stack = [(root, root, iter(graph[root]))]
    time_counter = 1
    start_time = time.perf_counter()
    while stack:
        grandparent, parent, children = stack[-1]
        try:
            child = next(children)
            if grandparent == child:
                continue
            if child in discovery:
                if discovery[child] < discovery[back_edge_target[parent]]:  # Back edge
                    if parent in instance_nodes:
                        back_edge_target[parent] = child
                    else:  # child is an instance node
                        if child in back_edge_source:
                            # Find common ancestor of back_edge_source[child] and parent.  It must be on the stack.
                            previous_discovery = discovery[back_edge_source[child]]
                            if True:
                                # first index where discovery >= previous_discovery, which may be len(stack)
                                stack_position = bisect.bisect_left(
                                    stack,
                                    previous_discovery,
                                    key=lambda x: discovery[x[1]],
                                )
                                # last index where discovery < previous_discovery
                                stack_position -= 1
                                common_ancestor = stack[stack_position][1]
                            else:  # linear search
                                common_ancestor = parent
                                stack_position = -1
                                while discovery[common_ancestor] > previous_discovery:
                                    stack_position -= 1
                                    common_ancestor = stack[stack_position][1]
                        else:
                            common_ancestor = child
                        back_edge_source[child] = parent
                        # This back edge will be cut for all ancestors between grandparent and child.
                        # But the ancestors between common_ancestor and child have already counted another equivalent back edge,
                        # so we only need to update from grandparent to common_ancestor.
                        subtree_back_edge_count[grandparent] += 1
                        subtree_back_edge_count[common_ancestor] -= 1
            else:
                discovery[child] = time_counter
                time_counter += 1
                time_block = 100000
                if time_counter % time_block == 0:
                    current_time = time.perf_counter()
                    elapsed_per_counter = (current_time - start_time) / time_block
                    start_time = current_time
                    estimated_seconds_remaining = int(elapsed_per_counter * (graph.number_of_nodes() - time_counter))
                    timedelta_seconds = timedelta(seconds=estimated_seconds_remaining)
                    print(f"DFS visited {time_counter} nodes. {timedelta_seconds} remaining")
                back_edge_target[child] = child
                subtree_back_edge_count[child] = 0
                subtree_back_edge_from_instance_count[child] = 0
                subtree_back_edge_from_predicate_count[child] = dict.fromkeys(predicates, 0)
                subtree_size[child] = 1 if child in instance_nodes else 0
                subtree_size_by_predicate[child] = {
                    predicate: 1 if child in predicate.true_nodes else 0 for predicate in predicates
                }
                stack.append((parent, child, iter(graph[child])))
                discovered_nodes.append(child)
        except StopIteration:
            stack.pop()
            finish[parent] = time_counter
            if grandparent != parent:
                child = back_edge_target[parent]
                if child != parent:  # note child cannot be grandparent here
                    # parent must be an instance node, and child is not
                    subtree_back_edge_count[grandparent] += 1
                    subtree_back_edge_count[child] -= 1
                    subtree_back_edge_from_instance_count[grandparent] += 1
                    subtree_back_edge_from_instance_count[child] -= 1
                    for predicate in predicates:
                        if parent in predicate.true_nodes:
                            subtree_back_edge_from_predicate_count[grandparent][predicate] += 1
                            subtree_back_edge_from_predicate_count[child][predicate] -= 1
                subtree_size[grandparent] += subtree_size[parent]
                subtree_back_edge_count[grandparent] += subtree_back_edge_count[parent]
                subtree_back_edge_from_instance_count[grandparent] += subtree_back_edge_from_instance_count[parent]
                for predicate in predicates:
                    subtree_size_by_predicate[grandparent][predicate] += subtree_size_by_predicate[parent][predicate]
                    subtree_back_edge_from_predicate_count[grandparent][predicate] += (
                        subtree_back_edge_from_predicate_count[parent][predicate]
                    )

    cut_size = {node: 1 + subtree_back_edge_count[node] for node in subtree_back_edge_count if node in instance_nodes}
    instance_node_count = sum(1 for node in graph if node in instance_nodes)
    # node_count_by_predicate = {
    #     predicate: sum(1 for node in graph if node in predicate.true_nodes) for predicate in predicates
    # }

    min_group_size = max(predicate.min_true_count + predicate.min_false_count for predicate in predicates)

    def safe_divide(numerator: int, denominator: int) -> float:
        """Divide numerator by denominator, returning 0 if denominator is 0."""
        if numerator == 0:
            return 0.0
        if denominator == 0:
            return float("inf")
        return numerator / denominator

    # A valid node_cost function must have the property that if A is an ancestor of B in the DFS tree and
    # node_cost(A) > node_cost(B), then cutting at B should not decrease node_cost(A).
    # Also any cuts at nodes that are not ancestors or descendants of A should not decrease node_cost(A).
    def get_gain_by_predicate(node: str) -> dict[Predicate, tuple[int, int]]:
        """Compute the gain by predicate of cutting at this node."""
        subtree_cut_size = 1 + subtree_back_edge_from_instance_count[node]
        subtree_size_after_cut = subtree_size[node] - subtree_cut_size
        subtree_size_after_cut_by_predicate = {
            predicate: subtree_size_by_predicate[node][predicate]
            - subtree_back_edge_from_predicate_count[node][predicate]
            - (1 if node in predicate.true_nodes else 0)
            for predicate in predicates
        }
        remaining_cut_size = subtree_back_edge_count[node] - subtree_back_edge_from_instance_count[node]
        remaining_size_after_cut = instance_node_count - subtree_size[node] - remaining_cut_size
        if subtree_size_after_cut < remaining_size_after_cut:
            gain_by_predicate = {
                predicate: (
                    min(subtree_size_after_cut_by_predicate[predicate], predicate.min_true_count),
                    min(
                        subtree_size_after_cut - subtree_size_after_cut_by_predicate[predicate],
                        predicate.min_false_count,
                    ),
                )
                for predicate in predicates
            }
        else:
            gain_by_predicate: dict[Predicate, tuple[int, int]] = dict.fromkeys(predicates, (0, 0))
        return gain_by_predicate
        # if subtree_size_after_cut < remaining_size_after_cut and any(
        #     gain_by_predicate[predicate] > 0
        #     for predicate in predicates
        # ):
        #     # An estimate of how many cuts we will need to compute, based on how many predicate nodes are added by this cut and how many predicate nodes we need.
        #     future_cut_count = max(
        #         safe_divide(
        #             predicate.min_true_count,
        #             min(predicate.min_true_count, subtree_size_after_cut_by_predicate[predicate]),
        #         )
        #         for predicate in predicates
        #     )
        #     # An estimate of the number of predicate nodes that will need to be cut to satisfy the predicate requirements, based on how many are in this cut and how many future cuts are required.
        #     future_cut_by_predicate = {
        #         predicate: safe_divide(
        #             predicate.min_true_count * subtree_back_edge_from_predicate_count[node][predicate],
        #             min(predicate.min_true_count, subtree_size_after_cut_by_predicate[predicate]),
        #         )
        #         for predicate in predicates
        #     }
        #     if any(
        #         future_cut_by_predicate[predicate] > node_count_by_predicate[predicate] - predicate.min_true_count
        #         for predicate in predicates
        #     ):
        #         return float(
        #             "inf"
        #         )  # this cut would make it impossible to satisfy the predicate requirements in the future
        #     total_future_cut = sum(future_cut_by_predicate.values())
        #     total_gain = sum(gain_by_predicate.values())
        #     if verbose:
        #         print(
        #             f"Node {node}: subtree size after cut {subtree_size_after_cut}, by predicate {subtree_size_after_cut_by_predicate}, future cut by predicate {future_cut_by_predicate}, total future cut {total_future_cut}, future cut count {future_cut_count}, total gain {total_gain}"
        #         )
        #     # This is valid because if B has larger gain per cut than A, then cutting at B (so that B is no longer part of A) cannot increase A's gain per cut.
        #     return -total_gain / cut_size[node]
        #     # If we don't penalize by future_cut_count, we can get a very good solution but only after computing thousands of cuts.
        #     return (
        #         future_cut_count * cut_size[node] + total_future_cut
        #     )  # prefer cuts that are likely to satisfy the predicate requirements with fewer future cuts and fewer nodes cut.
        # return float("inf")
        # smaller_group_size = min(subtree_size_after_cut, remaining_size_after_cut)
        # if smaller_group_size >= min_group_size:
        #     return cut_size[node] - instance_node_count  # prefer smaller cuts that satisfy min_group_size
        # return -smaller_group_size  # maximize smaller group size if min_group_size not satisfied

    # best_node = min(cut_size, key=node_cost)
    # 0.01 is too large
    # 0.001 is too large
    # 0.0001 is too small
    min_gain_per_cut = 2 * min_group_size / instance_node_count
    print(
        f"Min group size: {min_group_size}, Instance node count: {instance_node_count}, Minimum gain per cut: {min_gain_per_cut}"
    )
    cut = set()
    # We use discovery_threshold to ensure that if we cut at a node, we won't consider cutting at any of its descendants.
    discovery_threshold = 0
    for best_node in discovered_nodes:
        if discovery[best_node] < discovery_threshold:
            continue
        if best_node not in instance_nodes:
            continue
        gain_by_predicate = get_gain_by_predicate(best_node)
        total_gain = sum(t + f for t, f in gain_by_predicate.values())
        gain_per_cut = total_gain / cut_size[best_node]
        if gain_per_cut < min_gain_per_cut:
            continue
        print(
            f"Chose cut at node {best_node} with cut size {cut_size[best_node]} and gain {gain_by_predicate}, gain per cut {gain_per_cut}"
        )
        # Exclude descendants of best_node from consideration by updating discovery_threshold.
        discovery_threshold = finish[best_node]
        for predicate in predicates:
            predicate.min_true_count = max(0, predicate.min_true_count - gain_by_predicate[predicate][0])
            predicate.min_false_count = max(0, predicate.min_false_count - gain_by_predicate[predicate][1])
        # Collect subtree back edges for best node
        best_subtree_back_edges = [
            (node, child)
            for node in discovered_nodes[discovery[best_node] + 1 : finish[best_node] - 1]
            for child in graph[node]
            if discovery[child] < discovery[best_node]
        ]
        cut.update(source if source in instance_nodes else target for source, target in best_subtree_back_edges)
        cut.add(best_node)
    return cut


if __name__ == "__main__":
    # test_train_test_split()
    main()
