import networkx as nx
import numpy as np
import pytest
from sklearn.manifold import spectral_embedding

from hohypemb._core import check_params, embed_layers
from hohypemb._core.preweight import ra_preweight
from hohypemb._core.radial import check_radius_rule, degree_radius
from hohypemb.presets import EMBEDDING_PARAMS


def _triangle_with_tail():
    # nodes 0-1-2 form a triangle, node 3 hangs off node 2
    A = np.zeros((4, 4))
    for u, v in [(0, 1), (1, 2), (0, 2), (2, 3)]:
        A[u, v] = A[v, u] = 1.0
    return A


def _karate():
    return nx.to_numpy_array(nx.karate_club_graph(), weight=None)


def test_ra_preweight_formula():
    G, W = ra_preweight(_triangle_with_tail())
    assert G[0][1]["weight"] == pytest.approx(4.0)   # d=2,2  CN=1 -> 8/2
    assert G[0][2]["weight"] == pytest.approx(5.5)   # d=2,3  CN=1 -> 11/2
    assert G[2][3]["weight"] == pytest.approx(7.0)   # d=3,1  CN=0 -> 7/1
    assert W[0, 1] == pytest.approx(np.log(5.0))
    assert W[2, 3] == pytest.approx(np.log(8.0))
    assert W[0, 3] == 0.0


def test_degree_radius():
    np.testing.assert_allclose(degree_radius([0.0, 50.0], 50), [1.0, 1 - np.tanh(1.0)])


@pytest.mark.parametrize("rule", ["order", "logdegree"])
def test_rejected_radius_rules_explain_why(rule):
    with pytest.raises(ValueError, match="not supported"):
        check_radius_rule(rule)


def test_check_params_rejects_unknown_and_fills_defaults():
    with pytest.raises(TypeError, match="Unknown embedding parameter"):
        check_params({"betta": 3})
    p = check_params({"beta": 7})
    assert p["beta"] == 7 and p["method"] == "Isomap" and p["radius"] == "degree"
    with pytest.raises(ValueError):
        check_params({"method": "UMAP"})


def test_presets_values():
    assert EMBEDDING_PARAMS["cora"] == EMBEDDING_PARAMS["default"]
    assert EMBEDDING_PARAMS["ecoli"]["beta"] == 10000
    assert EMBEDDING_PARAMS["synthetic_holme_kim"]["method"] == "Isomap"
    assert EMBEDDING_PARAMS["synthetic_holme_kim"]["beta"] == 2000
    assert EMBEDDING_PARAMS["synthetic_control"]["method"] == "Spectral"
    assert EMBEDDING_PARAMS["synthetic_barabasi_albert"]["beta"] == 300
    for p in EMBEDDING_PARAMS.values():
        assert check_params(p) == p


def test_spectral_random_state_equals_global_seed():
    A = _karate()
    np.random.seed(3)
    ref = spectral_embedding(A, n_components=2)
    got = spectral_embedding(A, n_components=2, random_state=3)
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_embed_layers_does_not_touch_global_rng():
    A = _karate()
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    embed_layers([A], np.zeros_like(A), seed=0, method="Spectral", preweight=True, beta=100)
    np.testing.assert_array_equal(before, np.random.get_state()[1])


def test_embed_layers_single_layer_inside_disk():
    A = _karate()
    (coords,) = embed_layers([A], np.zeros_like(A), seed=0, preweight=True, beta=100)
    r = np.linalg.norm(coords, axis=1)
    assert coords.shape == (34, 2)
    assert np.all(r > 0) and np.all(r < 1)


def test_radius_depends_only_on_weighted_degree():
    A = _karate()
    (coords,) = embed_layers([A], np.zeros_like(A), seed=0, preweight=False, beta=20)
    np.testing.assert_allclose(np.linalg.norm(coords, axis=1), 1 - np.tanh(A.sum(axis=1) / 20), atol=1e-12)
