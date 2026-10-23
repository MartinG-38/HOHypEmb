from pathlib import Path

import numpy as np
import pytest

from hohypemb.datasets.synthetic import MODELS, planted_surfaces

GOLDEN = Path(__file__).parent / "golden"


@pytest.mark.parametrize("model,fixture", [("holme_kim", "G1_holme_kim"), ("small_world", "G2_small_world")])
def test_reproduces_the_notebook_generator(model, fixture):
    d = np.load(GOLDEN / f"{fixture}.npz", allow_pickle=False)
    sc = planted_surfaces(model, p_surface=50)
    edges = {tuple(sorted(e)) for e in sc.one_skeleton.edges()}
    assert edges == {tuple(e) for e in d["edges"]}
    flat, off = d["surfaces_flat"], d["surfaces_offsets"]
    surfaces = {tuple(sorted(flat[off[i]:off[i + 1]])) for i in range(len(off) - 1)}
    assert set(sc.simplices) == surfaces


def test_all_models_and_sizes():
    sizes = {"holme_kim": 500, "small_world": 80, "erdos_renyi": 500, "barabasi_albert": 100}
    for model in MODELS:
        # randint(0, 100) is inclusive, so p_surface=101 is the value that fills every triangle
        sc = planted_surfaces(model, p_surface=101)
        assert sc.n_nodes == sizes[model]
        assert len(sc.simplices) == len(sc.triangles_in_skeleton())


def test_seed_graph_params_and_errors():
    a, b = planted_surfaces("erdos_renyi", seed=5), planted_surfaces("erdos_renyi", seed=5)
    assert a.simplices == b.simplices
    assert planted_surfaces("erdos_renyi", seed=6).simplices != a.simplices
    small = planted_surfaces("holme_kim", seed=1139, n=80, p=0.25)   # the earlier configuration
    assert small.n_nodes == 80
    with pytest.raises(ValueError, match="model"):
        planted_surfaces("lattice")
    with pytest.raises(TypeError, match="graph parameters"):
        planted_surfaces("holme_kim", k=3)
