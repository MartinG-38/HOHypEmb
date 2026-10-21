import unittest

import networkx as nx
import numpy as np

from hohypemb.datasets.synthetic import identical_skeleton_pair, planted_hypergraph_sbm


class TestIdenticalSkeletonPair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.b, cls.labels = identical_skeleton_pair(
            n_nodes=200, K=3, n_planted_triangles=300, seed=0
        )

    def test_adjacency_matrices_are_bit_identical(self):
        np.testing.assert_array_equal(self.a.adjacency, self.b.adjacency)

    def test_base_graph_is_literally_the_same_object(self):
        self.assertIs(self.a.base_graph, self.b.base_graph)

    def test_equal_number_of_facets(self):
        self.assertEqual(len(self.a.simplices), len(self.b.simplices))

    def test_aligned_facets_are_community_pure(self):
        for f in self.a.simplices:
            self.assertEqual(len(set(self.labels[list(f)])), 1)

    def test_random_facets_are_not_all_pure(self):
        impure = sum(1 for f in self.b.simplices if len(set(self.labels[list(f)])) > 1)
        self.assertGreater(impure, 0, "control condition must contain mixed facets")

    def test_every_facet_is_a_triangle_of_the_shared_graph(self):
        G = self.a.base_graph
        for sc in (self.a, self.b):
            for f in sc.simplices:
                self.assertEqual(len(f), 3)
                for u, v in ((f[0], f[1]), (f[0], f[2]), (f[1], f[2])):
                    self.assertTrue(G.has_edge(u, v))

    def test_labels_length_matches_node_count(self):
        self.assertEqual(len(self.labels), self.a.n_nodes)

    def test_graph_is_connected(self):
        self.assertEqual(nx.number_connected_components(self.a.base_graph), 1)

    def test_reproducible_under_same_seed(self):
        a2, b2, l2 = identical_skeleton_pair(
            n_nodes=200, K=3, n_planted_triangles=300, seed=0
        )
        self.assertEqual(self.a.simplices, a2.simplices)
        self.assertEqual(self.b.simplices, b2.simplices)
        np.testing.assert_array_equal(self.labels, l2)

    def test_different_seeds_give_different_complexes(self):
        a3, _, _ = identical_skeleton_pair(
            n_nodes=200, K=3, n_planted_triangles=300, seed=1
        )
        self.assertNotEqual(self.a.simplices, a3.simplices)

    def test_n_fill_caps_the_number_of_facets(self):
        a, b, _ = identical_skeleton_pair(
            n_nodes=200, K=3, n_planted_triangles=300, n_fill=25, seed=0
        )
        self.assertEqual(len(a.simplices), 25)
        self.assertEqual(len(b.simplices), 25)


class TestPlantedHypergraphSBM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = planted_hypergraph_sbm(
            n_nodes=150, K=3, n_facets=200, facet_sizes=(3, 4), p_in=0.9, seed=0
        )

    def test_has_labels_and_facets(self):
        self.assertEqual(len(self.sc.node_labels), self.sc.n_nodes)
        self.assertGreater(len(self.sc.simplices), 0)

    def test_most_facets_are_community_pure(self):
        pure = sum(
            1 for f in self.sc.simplices if len(set(self.sc.node_labels[list(f)])) == 1
        )
        self.assertGreater(pure / len(self.sc.simplices), 0.5)

    def test_facet_sizes_respected(self):
        sizes = {len(f) for f in self.sc.simplices}
        self.assertTrue(sizes <= {3, 4}, f"unexpected facet sizes {sizes}")

    def test_giant_component_is_usable(self):
        giant = self.sc.giant_component()
        self.assertGreater(giant.n_nodes, 50)
        self.assertEqual(nx.number_connected_components(giant.one_skeleton), 1)

    def test_reproducible_under_same_seed(self):
        other = planted_hypergraph_sbm(
            n_nodes=150, K=3, n_facets=200, facet_sizes=(3, 4), p_in=0.9, seed=0
        )
        self.assertEqual(self.sc.simplices, other.simplices)


if __name__ == "__main__":
    unittest.main()
