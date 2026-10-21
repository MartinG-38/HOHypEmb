"""Generate tests/golden/*.npz from the author's MLNHypEmb and notebooks. Run once, by the author:
it needs MLNHypEmb and the original, unpublished research notebooks.

    HHO_SRC=/path/to/Hyperbolic_Higher_Order MPLBACKEND=Agg .venv/bin/python tools/make_golden.py

Embeddings are produced by the notebooks' own functions: their cells are read from the
.ipynb files and executed verbatim, and each embedding call is preceded by
np.random.seed(SEED), exactly as hho_bench did. Deliberate choices:
- layer-2 simplices are passed in sorted order, the order HOHypEmb uses, so that the
  layer-2 points of both implementations correspond one to one;
- G3 (cora, simplices of mixed sizes) uses ecoli_1's vectorised HHO_embedding, because
  Test_HHO_3's HHO_3 branch builds np.array(surfaces), which requires equal sizes;
- the synthetic graphs are generated with simulate_network's code pattern but explicit
  parameters (SYNTHETIC_GRAPHS), because the author keeps editing simulate_network.
"""
import json
import os
import platform
import random
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import scipy
import sklearn

SRC = Path(os.environ["HHO_SRC"]).expanduser()
OUT = Path(__file__).resolve().parents[1] / "tests" / "golden"
SEED = 0
TEST_HHO = "Test_HHO_3_statistics.ipynb"
ECOLI = "ecoli_1.ipynb"
ECOLI_DATA = ["511145.protein.links.v12.0.txt", "511145.protein.aliases.v12.0.txt", "83333.complextab.tsv"]
PARAMS = {
    "synthetic_holme_kim": dict(preweight=True, weight="weight", beta=2000, eta="auto", method="Isomap",
                                beta_dist=5, metric="precomputed", n_neighbors="auto", radius="degree"),
    "synthetic_small_world": dict(preweight=True, weight="weight", beta=220, eta="auto", method="Isomap",
                                  beta_dist=2, metric="precomputed", n_neighbors="auto", radius="degree"),
    "synthetic_control": dict(preweight=True, weight="weight", beta=1400, eta="auto", method="Spectral",
                              beta_dist=10, metric="precomputed", n_neighbors="auto", radius="degree"),
    "cora": dict(preweight=False, weight="weight", beta=50, eta="auto", method="Isomap",
                 beta_dist=1.5, metric="precomputed", n_neighbors="auto", radius="degree"),
    "ecoli": dict(preweight=True, weight="weight", beta=10000, eta="auto", method="Isomap",
                  beta_dist=1.5, metric="precomputed", n_neighbors="auto", radius="degree"),
}
EXPECTED_ECOLI_300 = (286, 5027, 5193, 69345)   # printed by ecoli_1.ipynb cell 37
P_SURFACE = 50
SYNTHETIC_GRAPHS = {
    # simulate_network's configuration as of 2026-09-25, written out explicitly
    "holme_kim": (1000, lambda: nx.powerlaw_cluster_graph(500, 2, 0.3)),
    "small_world": (104, lambda: nx.watts_strogatz_graph(80, 4, 0.3)),
}
VERSIONS = json.dumps({
    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__, "networkx": nx.__version__,
})

sys.path.insert(0, str(SRC))


class _Last:
    obj = None


def run_cells(notebook, markers, ns):
    """Execute, in order, the unique code cell of ``notebook`` containing each marker."""
    nb = json.loads((SRC / notebook).read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for marker in markers:
        hits = [s for s in cells if marker in s]
        if len(hits) != 1:
            raise RuntimeError(f"{notebook}: marker {marker!r} matched {len(hits)} code cells, expected 1")
        exec(compile(hits[0], f"<{notebook}: {marker}>", "exec"), ns)


def record_layer2(ns):
    """Wrap MlHypEmbUnsym_v3 so the object built inside the notebook function is kept."""
    base = ns["MlHypEmbUnsym_v3"]

    class Recording(base):
        def Embedding(self, *args, **kwargs):
            super().Embedding(*args, **kwargs)
            _Last.obj = self

    ns["MlHypEmbUnsym_v3"] = Recording


def sorted_simplices(simplices):
    return [list(s) for s in sorted({tuple(sorted(int(v) for v in s)) for s in simplices})]


def pack(simplices):
    simplices = [[int(v) for v in s] for s in simplices]
    flat = np.array([v for s in simplices for v in s], dtype=np.int64)
    offsets = np.cumsum([0] + [len(s) for s in simplices]).astype(np.int64)
    return flat, offsets


def embed_pair(ns, mat, simplices, params, mu, gamma, extra):
    _Last.obj = None
    np.random.seed(SEED)
    classical = np.asarray(ns["classical_embedding"](mat, **params))
    np.random.seed(SEED)
    hho = np.asarray(ns["HHO_embedding"](mat, simplices, **params, **extra, mu=mu, gamma=gamma))
    layer2 = np.asarray(_Last.obj.embeddings[1])
    gap = float(np.max(np.abs(np.linalg.norm(classical, axis=1) - np.linalg.norm(hho, axis=1))))
    print(f"    max|r_classical - r_hho| = {gap:.3e}")
    return classical, hho, layer2


def save(name, **arrays):
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", versions_json=np.array(VERSIONS), **arrays)
    print(f"  wrote tests/golden/{name}.npz")


def embedding_arrays(mat, simplices, params, mu, gamma, classical, hho, layer2):
    flat, offsets = pack(simplices)
    return dict(adjacency=np.asarray(mat, dtype=float), simplices_flat=flat, simplices_offsets=offsets,
                params_json=np.array(json.dumps(params)), mu=np.array(mu), gamma=np.array(gamma),
                seed=np.array(SEED), classical=classical, hho=hho, hho_layer2=layer2)


def generate(model):
    """simulate_network's pattern: global random.seed, networkx generator, one randint per triangle."""
    seed, make = SYNTHETIC_GRAPHS[model]
    random.seed(seed)
    G = make()
    mat = nx.to_numpy_array(G)
    triangles = [list(t) for t in nx.enumerate_all_cliques(G) if len(t) == 3]
    surfaces = [t for t in triangles if random.randint(0, 100) < P_SURFACE]
    return G, mat, surfaces


def synthetic(ns, model, name, params_key, mu, gamma):
    G, mat, surfaces = generate(model)
    assert surfaces, f"{model}: no surfaces"
    params = PARAMS[params_key]
    ordered = sorted_simplices(surfaces)
    cl, hho, l2 = embed_pair(ns, mat, ordered, params, mu, gamma, {"emb_method": "HHO_3"})
    sflat, soffsets = pack(surfaces)
    edges = np.array(sorted(tuple(sorted(e)) for e in G.edges()), dtype=np.int64)
    save(name, edges=edges, surfaces_flat=sflat, surfaces_offsets=soffsets,
         **embedding_arrays(mat, ordered, params, mu, gamma, cl, hho, l2))


def cora(ns):
    from hho_bench.complexes import load_cora_coauthorship
    sc = load_cora_coauthorship(str(SRC / "data_simplicial" / "cora-coauthorship.txt"), max_dim=5)
    sc = sc.giant_component()
    mat = sc.adjacency
    simplices = sorted_simplices(sc.higher_order_facets(min_dim=2, max_dim=5))
    params = PARAMS["cora"]
    cl, hho, l2 = embed_pair(ns, mat, simplices, params, 5.0, 1.0, {})
    save("G3_cora", **embedding_arrays(mat, simplices, params, 5.0, 1.0, cl, hho, l2))


def ecoli_build(ns, max_nodes):
    """Run ecoli_1's construction and preparation cells for ``max_nodes`` in a scratch dir."""
    ns["MAX_SIMPLEX_DIM"] = 2
    ns["MAX_NODES"] = max_nodes
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        for name in ECOLI_DATA:
            os.symlink(SRC / name, Path(tmp) / name)
        os.chdir(tmp)
        try:
            run_cells(ECOLI, ["CONSTRUCTION DU DATASET SIMPLICIAL E. COLI",
                              "PREPARATION DES DONNEES (composante geante"], ns)
        finally:
            os.chdir(here)
    G = ns["G_idx"]
    triangles = sorted_simplices(ns["higher_order_simplices"])
    assert all(len(t) == 3 for t in triangles)
    all_tri = []
    for c in nx.enumerate_all_cliques(G):
        if len(c) > 3:
            break
        if len(c) == 3:
            all_tri.append(tuple(sorted(c)))
    n_empty = len(set(all_tri) - {tuple(t) for t in triangles})
    edges = np.array(sorted(tuple(sorted(e)) for e in G.edges()), dtype=np.int64)
    counts = (G.number_of_nodes(), G.number_of_edges(), len(triangles), n_empty)
    return np.array(ns["nodelist"]), edges, triangles, counts


def main():
    print("Test_HHO_3 synthetic fixtures")
    ns_test = {"__name__": "golden_test_hho"}
    run_cells(TEST_HHO, ["# Libriaries importation", "def classical_embedding("], ns_test)
    record_layer2(ns_test)
    synthetic(ns_test, "holme_kim", "G1_holme_kim", "synthetic_holme_kim", 15.0, 0.5)
    synthetic(ns_test, "small_world", "G2_small_world", "synthetic_small_world", 15.0, 0.5)
    synthetic(ns_test, "small_world", "G6_control_spectral", "synthetic_control", 5.0, 1.0)

    print("ecoli_1 functions")
    # ecoli_1 imports nx in an earlier cell; cell 4.2 annotates nx.Graph
    ns_ecoli = {"__name__": "golden_ecoli", "nx": nx}
    run_cells(ECOLI, ["IMPORTS DE LA PARTIE HYPERBOLIQUE", "4.0 IMPORT DU PACKAGE LOCAL MLNHypEmb_101",
                      "4.2 EMBEDDING CLASSIQUE ET EMBEDDING HHO"], ns_ecoli)
    record_layer2(ns_ecoli)

    print("G3 cora")
    cora(ns_ecoli)

    print("G5 ecoli (max_nodes=250, embedded)")
    nodelist, edges, triangles, counts = ecoli_build(ns_ecoli, 250)
    assert triangles, "no filled triangle at max_nodes=250: report to the author"
    params = PARAMS["ecoli"]
    cl, hho, l2 = embed_pair(ns_ecoli, ns_ecoli["mat"], triangles, params, 15.0, 0.5, {})
    save("G5_ecoli_250", nodelist=nodelist,
         **embedding_arrays(ns_ecoli["mat"], triangles, params, 15.0, 0.5, cl, hho, l2))

    print("G4 ecoli (max_nodes=300, structure only)")
    nodelist, edges, triangles, counts = ecoli_build(ns_ecoli, 300)
    save("G4_ecoli_300_structure", nodelist=nodelist, edges=edges,
         triangles=np.array(triangles, dtype=np.int64), counts=np.array(counts, dtype=np.int64))
    print(f"  counts (nodes, edges, filled, empty) = {counts}")
    if tuple(counts) != EXPECTED_ECOLI_300:
        print(f"  WARNING: ecoli_1 cell 37 printed {EXPECTED_ECOLI_300}. REPORT THIS TO THE AUTHOR.")


if __name__ == "__main__":
    main()
