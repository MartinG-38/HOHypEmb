"""HOHypEmb must reproduce MLNHypEmb on every golden fixture . Never loosen tolerances."""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from hohypemb import ClassicalEmbedder, HHOEmbedder, SimplicialComplex
from hohypemb.geometry import angular_dist_matrix, clip_to_disk

GOLDEN = Path(__file__).parent / "golden"
CASES = [("G1_holme_kim", "input", None), ("G2_small_world", "input", None),
         ("G3_cora", "input", 5), ("G5_ecoli_250", "all_simplices", 2),
         ("G6_control_spectral", "input", None)]


def _hyp_dist_exact(X):
    """Pairwise Poincare distance from coordinate differences (not the Gram trick).

    ``cdist(..., "sqeuclidean")`` sums squared coordinate differences, so coincident
    points give exactly 0 and a 1-ulp change in a coordinate moves a distance by
    ~1e-12 at most. The Gram form ``|x|^2 + |y|^2 - 2 x.y`` used by
    ``geometry.hyp_dist_matrix`` leaves rounding noise of up to ~1e-6 on coincident
    pairs, which ``arccosh(1 + eps) ~ sqrt(2 eps)`` amplifies past ``atol=1e-8``
    as soon as the coordinates differ by one ulp from the reference.
    """
    X = clip_to_disk(X)
    nrm = np.sum(X ** 2, axis=1)
    denom = (1 - nrm)[:, None] * (1 - nrm)[None, :]
    return np.arccosh(np.maximum(1 + 2 * cdist(X, X, "sqeuclidean") / denom, 1.0))


def _complex(d):
    flat, off = d["simplices_flat"], d["simplices_offsets"]
    simplices = [tuple(flat[off[i]:off[i + 1]]) for i in range(len(off) - 1)]
    adj = d["adjacency"]
    return SimplicialComplex(simplices=simplices, n_nodes=adj.shape[0], name="golden",
                             base_graph=nx.from_numpy_array(adj))


@pytest.mark.golden
@pytest.mark.parametrize("name,layer2,max_dim", CASES)
def test_matches_mlnhypemb(name, layer2, max_dim):
    d = np.load(GOLDEN / f"{name}.npz", allow_pickle=False)
    sc = _complex(d)
    params = json.loads(str(d["params_json"]))
    seed = int(d["seed"])
    classical = ClassicalEmbedder(**params).fit(sc, seed=seed)
    hho = HHOEmbedder(mu=float(d["mu"]), gamma=float(d["gamma"]), layer2=layer2, max_dim=max_dim,
                      **params).fit(sc, seed=seed)
    for got, ref in [(classical.coords, d["classical"]), (hho.coords, d["hho"])]:
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), np.linalg.norm(ref, axis=1), atol=1e-12)
        np.testing.assert_allclose(_hyp_dist_exact(got), _hyp_dist_exact(ref), atol=1e-8)
    got, ref = hho.layer2, d["hho_layer2"]
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), np.linalg.norm(ref, axis=1), atol=1e-12)
    np.testing.assert_allclose(angular_dist_matrix(got), angular_dist_matrix(ref), atol=1e-8)


@pytest.mark.golden
def test_hho_leaves_radii_untouched_on_cora():
    d = np.load(GOLDEN / "G3_cora.npz", allow_pickle=False)
    sc = _complex(d)
    params = json.loads(str(d["params_json"]))
    r_cl = ClassicalEmbedder(**params).fit(sc, seed=0).radii()
    r_hho = HHOEmbedder(mu=5.0, gamma=1.0, layer2="input", max_dim=5, **params).fit(sc, seed=0).radii()
    assert np.max(np.abs(r_cl - r_hho)) < 1e-12
