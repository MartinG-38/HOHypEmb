import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / "golden"
EMBEDDED = ["G1_holme_kim", "G2_small_world", "G3_cora", "G5_ecoli_250", "G6_control_spectral"]
EMBEDDING_KEYS = ["adjacency", "simplices_flat", "simplices_offsets", "params_json", "mu", "gamma",
                  "seed", "classical", "hho", "hho_layer2", "versions_json"]


@pytest.mark.parametrize("name", EMBEDDED)
def test_embedding_fixture_complete(name):
    d = np.load(GOLDEN / f"{name}.npz", allow_pickle=False)
    for key in EMBEDDING_KEYS:
        assert key in d.files, key
    n = d["adjacency"].shape[0]
    assert d["classical"].shape == (n, 2)
    assert d["hho"].shape == (n, 2)
    assert d["hho_layer2"].shape == (len(d["simplices_offsets"]) - 1, 2)
    versions = json.loads(str(d["versions_json"]))
    assert versions["numpy"] == "2.0.2"
    assert versions["scikit-learn"] == "1.6.1"


@pytest.mark.parametrize("name", ["G1_holme_kim", "G2_small_world", "G6_control_spectral"])
def test_synthetic_fixture_has_generator_output(name):
    d = np.load(GOLDEN / f"{name}.npz", allow_pickle=False)
    assert d["edges"].shape[1] == 2
    assert len(d["surfaces_offsets"]) > 1


def test_ecoli_structure_fixture():
    d = np.load(GOLDEN / "G4_ecoli_300_structure.npz", allow_pickle=False)
    n_nodes, n_edges, n_filled, n_empty = (int(x) for x in d["counts"])
    assert len(d["nodelist"]) == n_nodes
    assert len(d["edges"]) == n_edges
    assert len(d["triangles"]) == n_filled
    assert n_empty > 0
