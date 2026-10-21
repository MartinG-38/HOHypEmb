import unittest

from hohypemb.datasets.cora import load_cora_coauthorship
from tests._data import CORA_PATH, require

import networkx as nx


class TestCoraLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = load_cora_coauthorship(require(CORA_PATH), max_dim=5).giant_component()

    def test_reproduces_reference_counts(self):
        self.assertEqual(self.sc.n_nodes, 723)
        self.assertEqual(self.sc.one_skeleton.number_of_edges(), 2243)

    def test_higher_order_facet_counts_match_reference(self):
        ho = self.sc.higher_order_simplices(min_dim=2, max_dim=5)
        self.assertEqual(len(ho), 281)
        dims = {}
        for f in ho:
            dims[len(f) - 1] = dims.get(len(f) - 1, 0) + 1
        self.assertEqual(dims, {2: 179, 3: 61, 4: 28, 5: 13})

    def test_seven_categories_present(self):
        self.assertEqual(len(set(self.sc.node_labels.tolist())), 7)

    def test_giant_component_is_connected(self):
        self.assertEqual(nx.number_connected_components(self.sc.one_skeleton), 1)

    def test_facets_are_deduplicated(self):
        self.assertEqual(len(self.sc.simplices), len(set(self.sc.simplices)))


if __name__ == "__main__":
    unittest.main()
