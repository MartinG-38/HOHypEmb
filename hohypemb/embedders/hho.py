"""HHO_3: one second-layer point per filled triangle, coupled to its vertices."""
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from hohypemb._core import check_params, embed_layers
from hohypemb.complex import SimplicialComplex
from hohypemb.embedders.base import Embedder, EmbeddingResult, require_connected, warn_if_large


class HHOEmbedder(Embedder):
    """Higher-order hyperbolic embedding (method HHO_3).

    Parameters
    ----------
    mu
        Coupling between a node and each layer-2 simplex containing it.
    gamma
        Coupling between two layer-2 simplices, times their number of shared nodes, when they
        share more than one node.
    layer2
        ``"all_simplices"`` (default): one point per face of dim 2..max_dim; with the default
        ``max_dim=2``, one layer-2 point per filled triangle (a simplex of dimension k >= 3 enters
        through its triangles). This is HHO as defined in the paper. ``"input"``: one point per
        distinct input simplex of dim >= 2; ``"maximal"``: one per maximal input simplex.
    max_dim
        Upper dimension of layer-2 simplices. ``None``: 2 for ``"all_simplices"``, no upper bound
        for ``"input"`` and ``"maximal"``.
    **params
        Embedding parameters, see ``hohypemb.presets``.

    Notes
    -----
    ``fit`` returns node coordinates in ``coords`` and one point per layer-2 simplex in ``layer2``.
    A layer-2 point gets the same radial rule as a node, ``r = 1 - tanh(s / beta)``, where ``s`` is
    its strength in the simplex-simplex block (``gamma`` times the number of shared nodes, summed
    over the simplices sharing more than one node with it); the ``mu`` couplings to its vertices do
    not count. A simplex that shares at most one node with every other simplex therefore has
    ``s = 0`` and radius exactly 1: its point lies on the boundary circle, not inside the disk (24 of
    the 87 layer-2 points of the Holme-Kim golden fixture). This is faithful to MLNHypEmb.
    ``EmbeddingResult.check_non_degenerate`` checks ``coords`` only; the functions of ``geometry``
    pull such a point to radius ``1 - 1e-9`` (``clip_to_disk``) before measuring distances.
    """

    name = "hho"
    structure = "higher_order"
    space = "poincare"
    LAYER2_MODES = ("input", "maximal", "all_simplices")

    def __init__(self, mu: float = 5.0, gamma: float = 1.0, layer2: str = "all_simplices",
                 max_dim: Optional[int] = None, **params: Any) -> None:
        if layer2 not in self.LAYER2_MODES:
            raise ValueError(f"layer2 must be one of {self.LAYER2_MODES}, got {layer2!r}")
        if layer2 == "all_simplices" and max_dim is None:
            max_dim = 2
        self.mu = float(mu)
        self.gamma = float(gamma)
        self.layer2 = layer2
        self.max_dim = max_dim
        self.params = check_params(params)

    def layer2_simplices(self, sc: SimplicialComplex) -> List[Sequence[int]]:
        if self.layer2 == "input":
            return sc.higher_order_simplices(min_dim=2, max_dim=self.max_dim)
        if self.layer2 == "maximal":
            return sc.maximal_simplices(min_dim=2, max_dim=self.max_dim)
        return [list(f) for f in sc.faces(min_dim=2, max_dim=self.max_dim)]

    @staticmethod
    def build_global_matrix(mat: np.ndarray, simplices: Sequence[Sequence[int]], mu: float,
                            gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        """Two-layer connectivity matrix. Returns ``(global_mat, surf_mat)``."""
        n_nodes = mat.shape[0]
        n_ho = len(simplices)
        incidence = np.zeros((n_ho, n_nodes), dtype=np.float64)
        for i, nodes in enumerate(simplices):
            incidence[i, list(nodes)] = 1.0
        # Accelerate BLAS raises spurious FP flags in matmul; incidence is a finite 0/1 matrix
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            n_shared = incidence @ incidence.T
        surf_mat = np.where(n_shared > 1, gamma * n_shared, 0.0)
        np.fill_diagonal(surf_mat, 0.0)
        off_diag = incidence.T * mu
        global_mat = np.zeros((n_nodes + n_ho, n_nodes + n_ho))
        global_mat[:n_nodes, :n_nodes] = mat
        global_mat[n_nodes:, n_nodes:] = surf_mat
        global_mat[:n_nodes, n_nodes:] = off_diag
        global_mat[n_nodes:, :n_nodes] = off_diag.T
        return global_mat, surf_mat

    def fit(self, sc: SimplicialComplex, seed: int = 0) -> EmbeddingResult:
        require_connected(sc)
        simplices = self.layer2_simplices(sc)
        if not simplices:
            raise ValueError(f"{sc.name or 'complex'}: no layer-2 simplex (layer2={self.layer2!r}, "
                             f"max_dim={self.max_dim}); HHO needs at least one simplex of dimension >= 2")
        warn_if_large(sc.n_nodes + len(simplices))
        mat = sc.adjacency
        global_mat, surf_mat = self.build_global_matrix(mat, simplices, self.mu, self.gamma)
        coords, layer2 = embed_layers([mat, surf_mat], global_mat, seed=seed, **self.params)
        result = EmbeddingResult(
            coords=coords, space="poincare", layer2=layer2,
            name=f"hho(mu={self.mu:g},gamma={self.gamma:g})",
            meta={"mu": self.mu, "gamma": self.gamma, "layer2": self.layer2, "max_dim": self.max_dim,
                  "n_layer2": len(simplices), "params": dict(self.params), "seed": seed})
        result.check_non_degenerate()
        return result
