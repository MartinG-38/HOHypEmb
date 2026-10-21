import unittest

import numpy as np
import pytest

from hohypemb import embedders as emb
from hohypemb import presets
from hohypemb.datasets.cora import load_cora_coauthorship
from hohypemb.datasets.synthetic import identical_skeleton_pair
from hohypemb.embedders import baselines

from tests._data import CORA_PATH, require


def _cora():
    return load_cora_coauthorship(require(CORA_PATH), max_dim=5).giant_component()


class TestEmbeddingResult(unittest.TestCase):
    def test_poincare_distance_matrix_is_hyperbolic(self):
        r = emb.EmbeddingResult(
            coords=np.array([[0.5, 0.0], [-0.5, 0.0]]),
            space="poincare", layer2=None, name="t", meta={},
        )
        D = r.distance_matrix()
        self.assertGreater(D[0, 1], 1.0)  # hyperbolic distance exceeds the euclidean 1.0

    def test_euclidean_distance_matrix_is_euclidean(self):
        r = emb.EmbeddingResult(
            coords=np.array([[0.5, 0.0], [-0.5, 0.0]]),
            space="euclidean", layer2=None, name="t", meta={},
        )
        np.testing.assert_allclose(r.distance_matrix()[0, 1], 1.0)

    def test_check_non_degenerate_warns_on_collapsed_radii(self):
        # Adapted (not a straight port): HOHypEmb's check_non_degenerate warns rather than
        # raises for identical radii (task-6 decision, spec-driven -- see
        # test_equal_radii_warn_instead_of_failing in the task-6 brief); collapsed radii can
        # legitimately occur for a degree-regular graph, unlike non-finite/out-of-disk coordinates.
        r = emb.EmbeddingResult(
            coords=np.full((10, 2), 1e-14), space="poincare",
            layer2=None, name="t", meta={},
        )
        with pytest.warns(UserWarning, match="identical"):
            r.check_non_degenerate()

    def test_check_non_degenerate_rejects_non_finite(self):
        coords = np.array([[0.1, 0.2], [np.nan, 0.0]])
        r = emb.EmbeddingResult(coords=coords, space="poincare", layer2=None, name="t", meta={})
        with self.assertRaises(ValueError):
            r.check_non_degenerate()

    def test_results_with_array_fields_compare_without_raising(self):
        """The generated dataclass __eq__ raised ValueError on the ndarray fields."""
        coords = np.array([[0.1, 0.2], [0.3, 0.4]])
        first = emb.EmbeddingResult(coords=coords, space="poincare", layer2=None, name="t", meta={})
        second = emb.EmbeddingResult(coords=coords, space="poincare", layer2=None, name="t", meta={})
        self.assertNotEqual(first, second)   # identity equality: distinct objects
        self.assertEqual(first, first)

    def test_results_are_hashable(self):
        coords = np.array([[0.1, 0.2]])
        result = emb.EmbeddingResult(coords=coords, space="poincare", layer2=None, name="t", meta={})
        self.assertEqual(len({result, result}), 1)


class TestClassicalAndHHO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = _cora()
        cls.classical = emb.ClassicalEmbedder(**presets.EMBEDDING_PARAMS["cora"])
        cls.hho = emb.HHOEmbedder(mu=5.0, gamma=1.0, layer2="input", max_dim=5, **presets.EMBEDDING_PARAMS["cora"])
        cls.r_cl = cls.classical.fit(cls.sc, seed=0)
        cls.r_hho = cls.hho.fit(cls.sc, seed=0)

    def test_shapes(self):
        self.assertEqual(self.r_cl.coords.shape, (723, 2))
        self.assertEqual(self.r_hho.coords.shape, (723, 2))

    def test_coordinates_are_inside_the_disk(self):
        for r in (self.r_cl, self.r_hho):
            self.assertTrue(np.all(np.linalg.norm(r.coords, axis=1) < 1.0))
            self.assertTrue(np.all(np.isfinite(r.coords)))

    def test_hho_exposes_a_layer2_embedding(self):
        self.assertIsNotNone(self.r_hho.layer2)
        self.assertEqual(self.r_hho.layer2.shape, (281, 2))

    def test_classical_has_no_layer2(self):
        self.assertIsNone(self.r_cl.layer2)

    def test_radii_are_identical_between_classical_and_hho(self):
        """Structural property of the method: only angles change."""
        r1 = np.linalg.norm(self.r_cl.coords, axis=1)
        r2 = np.linalg.norm(self.r_hho.coords, axis=1)
        self.assertLess(float(np.abs(r1 - r2).max()), 1e-12)

    def test_angles_do_change(self):
        a1 = np.arctan2(self.r_cl.coords[:, 1], self.r_cl.coords[:, 0])
        a2 = np.arctan2(self.r_hho.coords[:, 1], self.r_hho.coords[:, 0])
        delta = np.abs(np.angle(np.exp(1j * (a2 - a1))))
        self.assertGreater(float(delta.mean()), 0.1)

    def test_reproducible_under_same_seed(self):
        again = self.hho.fit(self.sc, seed=0)
        np.testing.assert_allclose(again.coords, self.r_hho.coords, atol=1e-12)

    def test_structure_and_space_tags(self):
        self.assertEqual(self.classical.structure, "pairwise")
        self.assertEqual(self.classical.space, "poincare")
        self.assertEqual(self.hho.structure, "higher_order")
        self.assertEqual(self.hho.space, "poincare")


class TestWeightedCliqueExpansion(unittest.TestCase):
    def test_differs_from_classical_and_is_valid(self):
        sc = _cora()
        base = emb.ClassicalEmbedder(**presets.EMBEDDING_PARAMS["cora"]).fit(sc, seed=0)
        wce = emb.WeightedCliqueExpansionEmbedder(**presets.EMBEDDING_PARAMS["cora"]).fit(sc, seed=0)
        wce.check_non_degenerate()
        self.assertEqual(wce.coords.shape, (723, 2))
        self.assertGreater(float(np.abs(wce.coords - base.coords).max()), 1e-6)

    def test_is_tagged_higher_order(self):
        self.assertEqual(emb.WeightedCliqueExpansionEmbedder.structure, "higher_order")


class TestPairwiseBlindnessOnControl(unittest.TestCase):
    """The invariant underpinning claim C2 and gate G1."""

    def test_pairwise_embedder_is_bit_identical_across_control_conditions(self):
        a, b, _ = identical_skeleton_pair(n_nodes=150, K=3, n_planted_triangles=200, seed=0)
        e = emb.ClassicalEmbedder(**presets.EMBEDDING_PARAMS["synthetic_control"])
        np.testing.assert_array_equal(e.fit(a, seed=0).coords, e.fit(b, seed=0).coords)

    def test_hho_does_differ_across_control_conditions(self):
        a, b, _ = identical_skeleton_pair(n_nodes=150, K=3, n_planted_triangles=200, seed=0)
        e = emb.HHOEmbedder(mu=5.0, gamma=1.0, **presets.EMBEDDING_PARAMS["synthetic_control"])
        self.assertGreater(
            float(np.abs(e.fit(a, seed=0).coords - e.fit(b, seed=0).coords).max()), 1e-6
        )


class TestBaselineFamily(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = _cora()
        cls.registry = emb.build_registry(presets.EMBEDDING_PARAMS["cora"], mu=5.0, gamma=1.0)

    def test_registry_contains_all_eight(self):
        self.assertEqual(
            sorted(self.registry),
            sorted([
                "classical", "hho", "weighted_clique", "star_expansion",
                "zhou_hypergraph", "euclidean_spectral", "euclidean_isomap",
                "euclidean_star",
            ]),
        )

    def test_every_cell_of_the_2x2_grid_is_populated(self):
        cells = {(e.space, e.structure) for e in self.registry.values()}
        for cell in [("poincare", "pairwise"), ("poincare", "higher_order"),
                     ("euclidean", "pairwise"), ("euclidean", "higher_order")]:
            self.assertIn(cell, cells)

    def test_all_embedders_produce_valid_finite_output(self):
        for name, embedder in self.registry.items():
            with self.subTest(embedder=name):
                r = embedder.fit(self.sc, seed=0)
                self.assertEqual(r.coords.shape, (723, 2))
                self.assertTrue(np.all(np.isfinite(r.coords)))
                r.check_non_degenerate()

    def test_poincare_baselines_share_the_hho_radial_rule(self):
        """Same radii => any difference is purely angular, which is the point."""
        ref = np.linalg.norm(self.registry["classical"].fit(self.sc, seed=0).coords, axis=1)
        for name in ("star_expansion", "zhou_hypergraph"):
            with self.subTest(embedder=name):
                r = np.linalg.norm(self.registry[name].fit(self.sc, seed=0).coords, axis=1)
                self.assertLess(float(np.abs(np.sort(r) - np.sort(ref)).max()), 1e-9)

    def test_euclidean_embedders_are_tagged_euclidean(self):
        for name in ("euclidean_spectral", "euclidean_isomap", "euclidean_star"):
            self.assertEqual(self.registry[name].space, "euclidean")

    def test_all_embedders_are_reproducible(self):
        for name, embedder in self.registry.items():
            with self.subTest(embedder=name):
                np.testing.assert_allclose(
                    embedder.fit(self.sc, seed=0).coords,
                    embedder.fit(self.sc, seed=0).coords,
                    atol=1e-10,
                )

    def test_higher_order_affinities_are_connected(self):
        """A disconnected affinity makes spectral embedding return component
        indicators for isolated nodes — silently, since the radial rule still
        assigns them a plausible radius."""
        import networkx as nx
        for name, affinity in (
            ("star", baselines._star_expansion_affinity(self.sc)),
            ("zhou", baselines._zhou_hypergraph_affinity(self.sc)),
        ):
            with self.subTest(affinity=name):
                graph = nx.from_numpy_array(affinity)
                self.assertEqual(nx.number_connected_components(graph), 1)

    def test_star_expansion_matches_hho_structure_at_zero_gamma(self):
        """Pins the ablation claim: star expansion IS HHO minus the gamma coupling."""
        facets = self.sc.higher_order_simplices(min_dim=2)
        global_mat, _ = emb.HHOEmbedder.build_global_matrix(
            self.sc.adjacency, facets, mu=1.0, gamma=0.0
        )
        np.testing.assert_array_equal(
            baselines._star_expansion_affinity(self.sc) > 0, global_mat > 0
        )


if __name__ == "__main__":
    unittest.main()
