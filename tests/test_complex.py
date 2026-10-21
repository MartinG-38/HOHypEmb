import unittest

import networkx as nx
import numpy as np

from hohypemb.complex import SimplicialComplex


class TestSimplicialComplexBasics(unittest.TestCase):
    def setUp(self):
        self.sc = SimplicialComplex(
            simplices=[(0, 1, 2), (2, 3), (3, 4, 5, 6)], n_nodes=7, name="toy"
        )

    def test_one_skeleton_is_clique_expansion(self):
        G = self.sc.one_skeleton
        for u, v in [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (5, 6)]:
            self.assertTrue(G.has_edge(u, v), f"missing edge ({u}, {v})")
        self.assertFalse(G.has_edge(0, 4))

    def test_edge_weight_counts_distinct_facets(self):
        sc = SimplicialComplex(simplices=[(0, 1, 2), (0, 1, 3)], n_nodes=4, name="t")
        self.assertEqual(sc.one_skeleton[0][1]["weight"], 2)
        self.assertEqual(sc.one_skeleton[0][2]["weight"], 1)

    def test_adjacency_shape_and_symmetry(self):
        A = self.sc.adjacency
        self.assertEqual(A.shape, (7, 7))
        np.testing.assert_allclose(A, A.T)

    def test_higher_order_facets_filters_by_dimension(self):
        self.assertEqual(self.sc.higher_order_simplices(min_dim=2), [[0, 1, 2], [3, 4, 5, 6]])
        self.assertEqual(self.sc.higher_order_simplices(min_dim=2, max_dim=2), [[0, 1, 2]])

    def test_is_facet(self):
        self.assertTrue(self.sc.is_input_simplex((2, 1, 0)))
        self.assertFalse(self.sc.is_input_simplex((0, 1, 3)))

    def test_facets_are_deduplicated_and_sorted(self):
        sc = SimplicialComplex(simplices=[(2, 1, 0), (0, 1, 2), (3, 4)], n_nodes=5, name="d")
        self.assertEqual(sc.simplices, [(0, 1, 2), (3, 4)])


class TestBaseGraphOverride(unittest.TestCase):
    def test_base_graph_overrides_derived_skeleton(self):
        G = nx.Graph()
        G.add_nodes_from(range(4))
        G.add_edge(0, 3, weight=1.0)
        sc = SimplicialComplex(simplices=[(0, 1, 2)], n_nodes=4, name="o", base_graph=G)
        self.assertTrue(sc.one_skeleton.has_edge(0, 3))
        self.assertFalse(sc.one_skeleton.has_edge(1, 2))

    def test_two_complexes_sharing_a_base_graph_have_identical_adjacency(self):
        G = nx.powerlaw_cluster_graph(40, 2, 0.4, seed=1)
        a = SimplicialComplex(simplices=[(0, 1, 2)], n_nodes=40, name="a", base_graph=G)
        b = SimplicialComplex(simplices=[(3, 4, 5), (6, 7, 8)], n_nodes=40, name="b", base_graph=G)
        np.testing.assert_array_equal(a.adjacency, b.adjacency)

    def test_one_skeleton_is_shared_by_reference_not_copied(self):
        """Sharing is deliberate (the identical-skeleton control relies on it); this pins it so it cannot drift."""
        G = nx.powerlaw_cluster_graph(30, 2, 0.4, seed=2)
        first = SimplicialComplex(simplices=[(0, 1, 2)], n_nodes=30, name="a", base_graph=G)
        second = SimplicialComplex(simplices=[(3, 4, 5)], n_nodes=30, name="b", base_graph=G)
        self.assertIs(first.one_skeleton, second.one_skeleton)


class TestEqualityAndHashing(unittest.TestCase):
    def test_complexes_with_labels_compare_without_raising(self):
        """Auto-generated dataclass __eq__ raised ValueError on the ndarray label field."""
        labels = np.array(["a", "b", "c"])
        first = SimplicialComplex(simplices=[(0, 1, 2)], n_nodes=3, name="x", node_labels=labels)
        second = SimplicialComplex(simplices=[(0, 1, 2)], n_nodes=3, name="x", node_labels=labels)
        self.assertNotEqual(first, second)   # identity equality: distinct objects
        self.assertEqual(first, first)

    def test_complexes_are_hashable(self):
        labels = np.array(["a", "b", "c"])
        sc = SimplicialComplex(simplices=[(0, 1, 2)], n_nodes=3, name="x", node_labels=labels)
        self.assertEqual(len({sc, sc}), 1)


class TestDerivedStructures(unittest.TestCase):
    def setUp(self):
        self.sc = SimplicialComplex(
            simplices=[(0, 1, 2), (1, 2, 3), (4, 5)], n_nodes=6, name="t"
        )

    def test_incidence_matrix_shape_and_content(self):
        M = self.sc.incidence_matrix().toarray()
        self.assertEqual(M.shape, (3, 6))
        np.testing.assert_array_equal(M[0], [1, 1, 1, 0, 0, 0])

    def test_facet_shared_count_matrix_counts_cofacets(self):
        S = self.sc.shared_count_matrix()
        self.assertEqual(S.shape, (6, 6))
        self.assertEqual(S[1, 2], 2)   # nodes 1 and 2 share two facets
        self.assertEqual(S[0, 3], 0)
        self.assertEqual(S[0, 0], 0)   # diagonal zeroed

    def test_triangles_in_skeleton(self):
        tris = self.sc.triangles_in_skeleton()
        self.assertIn((0, 1, 2), tris)
        self.assertIn((1, 2, 3), tris)

    def test_giant_component_reindexes_contiguously(self):
        sc = SimplicialComplex(
            simplices=[(0, 1, 2), (10, 11)], n_nodes=12, name="frag"
        )
        giant = sc.giant_component()
        self.assertEqual(giant.n_nodes, 3)
        self.assertEqual(sorted({v for f in giant.simplices for v in f}), [0, 1, 2])

    def test_giant_component_carries_labels_through(self):
        labels = np.array(list("aaabbbccccdd"))
        sc = SimplicialComplex(
            simplices=[(0, 1, 2), (10, 11)], n_nodes=12, name="frag", node_labels=labels
        )
        np.testing.assert_array_equal(sc.giant_component().node_labels, ["a", "a", "a"])

    def test_giant_component_preserves_shared_base_graph_identity(self):
        """Two complexes with unrelated facets over one base_graph must still agree
        edge-for-edge after restriction — the invariant the identical-skeleton control depends on."""
        G = nx.powerlaw_cluster_graph(40, 2, 0.4, seed=3)
        first = SimplicialComplex(simplices=[(0, 1, 2)], n_nodes=40, name="a", base_graph=G)
        second = SimplicialComplex(simplices=[(5, 6, 7), (8, 9, 10)], n_nodes=40, name="b", base_graph=G)
        np.testing.assert_array_equal(
            first.giant_component().adjacency, second.giant_component().adjacency
        )


if __name__ == "__main__":
    unittest.main()
