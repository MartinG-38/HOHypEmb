import networkx as nx
import numpy as np
import pytest

from hohypemb import SimplicialComplex, presets
from hohypemb.embedders import ClassicalEmbedder, WeightedCliqueExpansionEmbedder, build_registry

KEYS = ["classical", "hho", "weighted_clique", "star_expansion", "zhou_hypergraph",
        "euclidean_spectral", "euclidean_isomap", "euclidean_star"]


def _small():
    G = nx.karate_club_graph()
    triangles = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3][:12]
    return SimplicialComplex.from_graph(nx.Graph(G.edges()), triangles, name="karate")


def test_registry_keys_unchanged():
    assert list(build_registry(presets.EMBEDDING_PARAMS["cora"])) == KEYS


def test_every_embedder_leaves_global_rng_untouched():
    sc = _small()
    for name, embedder in build_registry(presets.EMBEDDING_PARAMS["cora"], max_dim=None).items():
        np.random.seed(11)
        before = np.random.get_state()[1].copy()
        embedder.fit(sc, seed=0)
        np.testing.assert_array_equal(before, np.random.get_state()[1], err_msg=name)


@pytest.mark.parametrize("preweight", [True, False])
def test_weighted_clique_is_classical_without_simplices(preweight):
    sc = _small()
    skeleton = SimplicialComplex(simplices=[], n_nodes=sc.n_nodes, base_graph=sc.one_skeleton)
    params = {"preweight": preweight, "beta": 5.0, "n_neighbors": 3}
    expected = ClassicalEmbedder(**params).fit(skeleton, seed=0).coords
    np.testing.assert_array_equal(WeightedCliqueExpansionEmbedder(**params).fit(skeleton, seed=0).coords, expected)


@pytest.mark.parametrize("preweight", [True, False])
def test_weighted_clique_keeps_classical_radii_and_reads_the_simplices(preweight):
    sc = _small()
    params = {"preweight": preweight, "beta": 5.0, "n_neighbors": 3}
    classical = ClassicalEmbedder(**params).fit(sc, seed=0).coords
    clique = WeightedCliqueExpansionEmbedder(**params).fit(sc, seed=0).coords
    np.testing.assert_allclose(np.linalg.norm(clique, axis=1), np.linalg.norm(classical, axis=1), atol=1e-12)
    assert not np.allclose(clique, classical)


def test_baselines_reject_disconnected_skeleton():
    G = nx.Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    sc = SimplicialComplex.from_graph(G, [(0, 1, 2), (3, 4, 5)])
    for embedder in build_registry(presets.EMBEDDING_PARAMS["cora"]).values():
        with pytest.raises(ValueError, match="giant_component"):
            embedder.fit(sc)
