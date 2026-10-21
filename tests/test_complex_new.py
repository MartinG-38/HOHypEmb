import networkx as nx
import numpy as np
import pytest

from hohypemb.complex import SimplicialComplex


def _toy():
    """A tetrahedron 0123 with a nested triangle 012, a triangle 345, and edge 2-4."""
    G = nx.Graph()
    for u, v in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5), (2, 4)]:
        G.add_edge(u, v)
    return SimplicialComplex.from_graph(G, [(0, 1, 2, 3), (0, 1, 2), (3, 4, 5)], name="toy")


def test_rejects_out_of_range_ids():
    with pytest.raises(ValueError, match=r"outside 0\.\.2.*from_graph"):
        SimplicialComplex(simplices=[(1, 2, 3)], n_nodes=3, name="one-based")


def test_layer2_views():
    sc = _toy()
    assert sc.higher_order_simplices() == [[0, 1, 2], [0, 1, 2, 3], [3, 4, 5]]
    assert sc.maximal_simplices() == [[0, 1, 2, 3], [3, 4, 5]]
    assert sc.faces(max_dim=2) == [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (3, 4, 5)]
    assert sc.maximal_simplices(max_dim=2) == [[3, 4, 5]]
    with pytest.raises(ValueError, match="max_dim"):
        sc.faces(max_dim=None)


def test_filled_and_empty_triangles():
    sc = _toy()
    assert sc.is_filled((0, 1, 3)) and sc.is_filled((3, 4, 5))
    assert not sc.is_filled((2, 3, 4)) and not sc.is_filled((0, 1))
    assert sc.filled_triangles() == [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (3, 4, 5)]
    assert sc.empty_triangles() == [(2, 3, 4)]


def test_from_graph_relabels_and_validates():
    G = nx.Graph([("b", "a"), ("a", "c"), ("b", "c"), ("c", "d")])
    sc = SimplicialComplex.from_graph(G, [("a", "b", "c")], node_labels={"a": "x", "d": "y"})
    assert sc.node_names == ["a", "b", "c", "d"]
    assert sc.simplices == [(0, 1, 2)]
    assert list(sc.node_labels) == ["x", None, None, "y"]
    assert sc.indices_of(["d", "zz", "a"]) == [3, 0]
    with pytest.raises(ValueError, match="not a clique"):
        SimplicialComplex.from_graph(G, [("a", "b", "d")])
    with pytest.raises(ValueError, match="not in G"):
        SimplicialComplex.from_graph(G, [("a", "b", "q")])


def test_from_graph_unsortable_ids():
    G = nx.Graph([(1, "a"), ("a", 2), (2, 1)])
    sc = SimplicialComplex.from_graph(G, [(1, "a", 2)])
    assert sc.node_names == [1, "a", 2]
    assert sc.simplices == [(0, 1, 2)]
    assert sc.indices_of(["a"]) == [1]


def test_from_graph_keeps_edge_weights():
    G = nx.Graph()
    G.add_edge("u", "v", weight=3.0)
    G.add_edge("v", "w")
    sc = SimplicialComplex.from_graph(G, [])
    np.testing.assert_array_equal(sc.adjacency, [[0, 3, 0], [3, 0, 1], [0, 1, 0]])


def test_triangles_do_not_enumerate_large_cliques():
    # enumerating every clique of K30 (2**30 subsets) would never finish
    sc = SimplicialComplex(simplices=[], n_nodes=30, base_graph=nx.complete_graph(30))
    assert len(sc.triangles_in_skeleton()) == 4060


def test_giant_component_keeps_names_and_labels():
    G = nx.Graph([("a", "b"), ("b", "c"), ("a", "c"), ("x", "y")])
    sc = SimplicialComplex.from_graph(G, [("a", "b", "c")], node_labels={"a": 1, "b": 2, "c": 3})
    giant = sc.giant_component()
    assert giant.node_names == ["a", "b", "c"]
    assert list(giant.node_labels) == [1, 2, 3]
    assert giant.simplices == [(0, 1, 2)]
