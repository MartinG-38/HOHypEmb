import os
import subprocess
import sys
import textwrap
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import hohypemb as hh
from hohypemb.embedders import base

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    G = nx.Graph()
    for u, v in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5), (2, 4)]:
        G.add_edge(u, v)
    return hh.SimplicialComplex.from_graph(G, [(0, 1, 2, 3), (0, 1, 2), (3, 4, 5)], name="toy")


PARAMS = dict(preweight=False, beta=10.0)


@pytest.mark.parametrize("layer2,max_dim,expected", [("input", None, 3), ("maximal", None, 2),
                                                     ("all_simplices", 2, 5)])
def test_layer2_modes(layer2, max_dim, expected):
    res = hh.HHOEmbedder(layer2=layer2, max_dim=max_dim, **PARAMS).fit(_toy(), seed=0)
    assert res.meta["n_layer2"] == expected
    assert res.layer2.shape == (expected, 2)
    assert res.coords.shape == (6, 2)


def test_layer2_argument_errors():
    with pytest.raises(ValueError, match="max_dim"):
        hh.HHOEmbedder(layer2="all_simplices")
    with pytest.raises(ValueError, match="layer2"):
        hh.HHOEmbedder(layer2="facets")


def test_no_layer2_simplex():
    sc = hh.SimplicialComplex.from_graph(nx.path_graph(4), [])
    with pytest.raises(ValueError, match="no layer-2 simplex"):
        hh.HHOEmbedder(**PARAMS).fit(sc)


def test_disconnected_skeleton_is_rejected():
    G = nx.Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    sc = hh.SimplicialComplex.from_graph(G, [(0, 1, 2), (3, 4, 5)])
    for embedder in (hh.ClassicalEmbedder(**PARAMS), hh.HHOEmbedder(**PARAMS)):
        with pytest.raises(ValueError, match="giant_component"):
            embedder.fit(sc)


def test_large_problem_warns(monkeypatch):
    monkeypatch.setattr(base, "DENSE_WARNING_SIZE", 3)
    with pytest.warns(UserWarning, match="GB of memory"):
        hh.HHOEmbedder(**PARAMS).fit(_toy())


def test_fit_leaves_global_rng_untouched_and_is_repeatable():
    sc = _toy()
    np.random.seed(7)
    before = np.random.get_state()[1].copy()
    a = hh.HHOEmbedder(method="Spectral", **PARAMS).fit(sc, seed=1)
    b = hh.HHOEmbedder(method="Spectral", **PARAMS).fit(sc, seed=1)
    np.testing.assert_array_equal(before, np.random.get_state()[1])
    np.testing.assert_array_equal(a.coords, b.coords)


def test_radii_identical_between_classical_and_hho():
    sc = _toy()
    r_cl = hh.ClassicalEmbedder(**PARAMS).fit(sc).radii()
    r_hho = hh.HHOEmbedder(**PARAMS).fit(sc).radii()
    assert np.max(np.abs(r_cl - r_hho)) < 1e-12


def test_equal_radii_warn_instead_of_failing():
    res = hh.EmbeddingResult(coords=np.array([[0.5, 0.0], [0.0, 0.5]]), space="poincare",
                             layer2=None, name="regular")
    with pytest.warns(UserWarning, match="identical"):
        res.check_non_degenerate()


def test_lazy_submodules():
    assert hh.geometry.hyp_dist_matrix is not None
    assert hh.presets.EMBEDDING_PARAMS["cora"]["beta"] == 50
    with pytest.raises(AttributeError):
        hh.not_a_module


def _coords_hash(hash_seed: str) -> str:
    script = textwrap.dedent("""
        import hashlib, json
        import networkx as nx, numpy as np
        from hohypemb import HHOEmbedder, SimplicialComplex
        d = np.load("tests/golden/G1_holme_kim.npz")
        flat, off = d["simplices_flat"], d["simplices_offsets"]
        simplices = [tuple(flat[off[i]:off[i + 1]]) for i in range(len(off) - 1)]
        sc = SimplicialComplex(simplices, d["adjacency"].shape[0],
                               base_graph=nx.from_numpy_array(d["adjacency"]))
        res = HHOEmbedder(mu=5.0, gamma=0.1, **json.loads(str(d["params_json"]))).fit(sc, seed=0)
        print(hashlib.sha256(np.ascontiguousarray(res.coords).tobytes()).hexdigest())
    """)
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True,
                         text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_results_do_not_depend_on_python_hash_seed():
    assert _coords_hash("0") == _coords_hash("1")
