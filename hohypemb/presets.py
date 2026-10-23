"""Embedding parameter sets used in the notebooks and the paper .

``beta`` sets the radial scale and must be tuned to the weighted-degree scale of the graph.
``eta`` is accepted for compatibility with MLNHypEmb parameter sets; it has no effect under
the only supported radius rule, ``"degree"``.
"""
from typing import Any, Dict

PARAM_NAMES = ("preweight", "weight", "beta", "eta", "method", "beta_dist", "metric", "n_neighbors", "radius")


def _params(preweight: bool, beta: float, method: str, beta_dist: float) -> Dict[str, Any]:
    return dict(preweight=preweight, weight="weight", beta=beta, eta="auto", method=method,
                beta_dist=beta_dist, metric="precomputed", n_neighbors="auto", radius="degree")


EMBEDDING_PARAMS: Dict[str, Dict[str, Any]] = {
    "default": _params(False, 50, "Isomap", 1.5),
    "cora": _params(False, 50, "Isomap", 1.5),
    "ecoli": _params(True, 10000, "Isomap", 1.5),
    "synthetic_holme_kim": _params(True, 2000, "Isomap", 5),
    "synthetic_small_world": _params(True, 220, "Isomap", 2),
    # beta is on the order of each model's RA-preweighted strength, so that the radial rule does
    # not saturate (a much smaller beta puts many nodes at r < 1e-6, at the center of the disk)
    "synthetic_erdos_renyi": _params(True, 2000, "Isomap", 2),
    "synthetic_barabasi_albert": _params(True, 300, "Isomap", 2),
    # identical-skeleton control (E2) and the tests ported from the author's earlier research code
    "synthetic_control": _params(True, 1400, "Spectral", 10),
}
