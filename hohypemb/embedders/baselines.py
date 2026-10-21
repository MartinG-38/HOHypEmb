"""Six baselines completing the 2x2 design (geometry x structure).

                    pairwise                higher-order
    euclidean       EuclideanSpectral       EuclideanStarExpansion
                    EuclideanIsomap
    poincare        Classical               HHO, WeightedCliqueExpansion,
                                            StarExpansion, ZhouHypergraph

Every Poincaré-row baseline uses the same radial FUNCTION as HHO, ``r = 1 - tanh(d / beta)``, but
not always the same degree ``d``:

- ``ClassicalEmbedder`` and ``HHOEmbedder``: ``d`` is the node strength in the pairwise graph
  ``sc.adjacency``, RA-preweighted first when ``preweight=True``.
- ``WeightedCliqueExpansionEmbedder``: the classical ``d`` (its radii are the classical ones); only
  the angles see the simplices.
- ``StarExpansionEmbedder`` and ``ZhouHypergraphEmbedder``: ``d`` is the raw weighted degree of the
  1-skeleton (edge attribute ``weight``), whatever ``preweight`` says (``geometry.apply_radial_rule``).

So with ``preweight=True`` (e.g. the ``synthetic_control`` preset used by E2) the star-expansion and
Zhou radii differ from the classical/HHO radii, and comparisons with those two baselines do not
isolate the angular ordering: radius and angle both differ.
"""
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.manifold import Isomap, spectral_embedding

from hohypemb import geometry as geo
from hohypemb._core import check_params
from hohypemb._core.manifold import reduce_2d
from hohypemb._core.preweight import ra_preweight
from hohypemb._core.radial import degree_radius, layer_strength
from hohypemb.complex import SimplicialComplex
from hohypemb.embedders.base import Embedder, EmbeddingResult, require_connected
from hohypemb.embedders.classical import ClassicalEmbedder
from hohypemb.embedders.hho import HHOEmbedder


def _weighted_degrees(G: nx.Graph, n_nodes: int, weight: Optional[str]) -> np.ndarray:
    d = dict(G.degree(weight=weight))
    return np.array([float(d.get(i, 0.0)) for i in range(n_nodes)])


def _spectral_2d(affinity: np.ndarray, seed: int) -> np.ndarray:
    """2-D spectral embedding of a symmetric non-negative affinity matrix."""
    return np.asarray(spectral_embedding(affinity, n_components=2, drop_first=True, random_state=seed))


def _star_expansion_affinity(sc: SimplicialComplex, mu: float = 1.0) -> np.ndarray:
    """Node-simplex bipartite incidence ON TOP OF the 1-skeleton (= HHO's global matrix at gamma=0)."""
    n = sc.n_nodes
    simplices = sc.higher_order_simplices(min_dim=2)
    m = len(simplices)
    A = np.zeros((n + m, n + m))
    A[:n, :n] = sc.adjacency
    for j, simplex in enumerate(simplices):
        for v in simplex:
            A[v, n + j] = mu
            A[n + j, v] = mu
    return A


def _zhou_hypergraph_affinity(sc: SimplicialComplex) -> np.ndarray:
    """Zhou et al. (2006) normalised hypergraph affinity Dv^-1/2 H W De^-1 H^T Dv^-1/2.

    Hyperedges = every input simplex plus every 1-skeleton edge (a graph is a 2-uniform
    hypergraph), which gives every node a nonzero degree.
    """
    hyperedges = [tuple(s) for s in sc.simplices] + [tuple(e) for e in sc.one_skeleton.edges()]
    if not hyperedges:
        raise ValueError(f"{sc.name}: no hyperedges for the Zhou Laplacian")
    n = sc.n_nodes
    rows, cols = [], []
    for j, edge in enumerate(hyperedges):
        for v in edge:
            rows.append(v)
            cols.append(j)
    H = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(hyperedges)))
    dv = np.asarray(H.sum(axis=1)).ravel()
    de = np.asarray(H.sum(axis=0)).ravel()
    dv_inv_sqrt = diags(1.0 / np.sqrt(np.clip(dv, 1e-12, None)))
    de_inv = diags(1.0 / np.clip(de, 1e-12, None))
    # Accelerate BLAS raises spurious FP flags in matmul; H, dv_inv_sqrt, de_inv are all finite
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        A = (dv_inv_sqrt @ H @ de_inv @ H.T @ dv_inv_sqrt).toarray()
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0.0)
    return np.clip(A, 0.0, None)


class WeightedCliqueExpansionEmbedder(Embedder):
    """Higher-order information folded into EDGE WEIGHTS, then embedded pairwise.

    The classical pipeline with one change: the clique expansion of the simplices (the shared simplex
    counts, ``sc.shared_count_matrix()``) is added to the edge weights of the matrix that sets the
    angles, after the RA preweighting when ``preweight=True``. The radii are the classical ones. Adding
    the counts before the preweighting would be pointless: it recomputes every weight from topology.
    """

    name = "weighted_clique"
    structure = "higher_order"
    space = "poincare"

    def __init__(self, **params: Any) -> None:
        self.params = check_params(params)

    def fit(self, sc: SimplicialComplex, seed: int = 0) -> EmbeddingResult:
        require_connected(sc)
        p = self.params
        adjacency = np.asarray(sc.adjacency, dtype=float)
        graph, block = ra_preweight(adjacency) if p["preweight"] else (nx.from_numpy_array(adjacency), adjacency)
        mat = np.where(adjacency > 0, block + sc.shared_count_matrix(), 0.0)
        coords = reduce_2d(mat, p["method"], p["beta_dist"], p["metric"], p["n_neighbors"], seed, -1)
        unit = coords / np.sqrt(np.sum(coords ** 2, axis=1, keepdims=True))
        radius = degree_radius(layer_strength(graph, p["weight"]), p["beta"])
        result = EmbeddingResult(coords=unit * radius[:, np.newaxis], space="poincare", layer2=None,
                                 name=self.name, meta={"params": dict(p), "seed": seed})
        result.check_non_degenerate()
        return result


class StarExpansionEmbedder(Embedder):
    """Spectral embedding of the node-simplex bipartite graph + the degree radial rule.

    The radius uses the raw weighted degree of the 1-skeleton, never the RA-preweighted strength,
    even when ``preweight=True``; its radii then differ from the classical/HHO radii.
    """

    name = "star_expansion"
    structure = "higher_order"
    space = "poincare"

    def __init__(self, **params: Any) -> None:
        self.params = check_params(params)

    def fit(self, sc: SimplicialComplex, seed: int = 0) -> EmbeddingResult:
        require_connected(sc)
        full = _spectral_2d(_star_expansion_affinity(sc), seed)
        degrees = _weighted_degrees(sc.one_skeleton, sc.n_nodes, self.params["weight"])
        coords = geo.apply_radial_rule(full[: sc.n_nodes], degrees, beta=self.params["beta"],
                                       rule=self.params["radius"])
        result = EmbeddingResult(coords=coords, space="poincare", layer2=full[sc.n_nodes:],
                                 name=self.name, meta={"params": dict(self.params), "seed": seed})
        result.check_non_degenerate()
        return result


class ZhouHypergraphEmbedder(Embedder):
    """Zhou normalised hypergraph Laplacian + the degree radial rule.

    The radius uses the raw weighted degree of the 1-skeleton, never the RA-preweighted strength,
    even when ``preweight=True``; its radii then differ from the classical/HHO radii.
    """

    name = "zhou_hypergraph"
    structure = "higher_order"
    space = "poincare"

    def __init__(self, **params: Any) -> None:
        self.params = check_params(params)

    def fit(self, sc: SimplicialComplex, seed: int = 0) -> EmbeddingResult:
        require_connected(sc)
        angles = _spectral_2d(_zhou_hypergraph_affinity(sc), seed)
        degrees = _weighted_degrees(sc.one_skeleton, sc.n_nodes, self.params["weight"])
        coords = geo.apply_radial_rule(angles, degrees, beta=self.params["beta"], rule=self.params["radius"])
        result = EmbeddingResult(coords=coords, space="poincare", layer2=None, name=self.name,
                                 meta={"params": dict(self.params), "seed": seed})
        result.check_non_degenerate()
        return result


class EuclideanSpectralEmbedder(Embedder):
    """Plain 2-D spectral embedding of the 1-skeleton. No radial rule."""

    name = "euclidean_spectral"
    structure = "pairwise"
    space = "euclidean"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def fit(self, sc: SimplicialComplex, seed: int = 0) -> EmbeddingResult:
        require_connected(sc)
        coords = _spectral_2d(sc.adjacency, seed)
        result = EmbeddingResult(coords=coords, space="euclidean", layer2=None, name=self.name,
                                 meta={"seed": seed})
        result.check_non_degenerate()
        return result


class EuclideanIsomapEmbedder(Embedder):
    """2-D Isomap on shortest-path distances of the 1-skeleton. No radial rule. Deterministic."""

    name = "euclidean_isomap"
    structure = "pairwise"
    space = "euclidean"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def fit(self, sc: SimplicialComplex, seed: int = 0) -> EmbeddingResult:
        require_connected(sc)
        G = sc.one_skeleton
        n = sc.n_nodes
        sp = dict(nx.all_pairs_shortest_path_length(G))
        D = np.full((n, n), 0.0)
        for i in range(n):
            row = sp.get(i, {})
            far = max(row.values()) + 1 if row else 1
            for j in range(n):
                D[i, j] = row.get(j, far)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0.0)
        model = Isomap(n_components=2, n_neighbors=min(10, n - 1), eigen_solver="dense", metric="precomputed")
        result = EmbeddingResult(coords=np.asarray(model.fit_transform(D)), space="euclidean",
                                 layer2=None, name=self.name, meta={"seed": seed})
        result.check_non_degenerate()
        return result


class EuclideanStarExpansionEmbedder(Embedder):
    """Star expansion without the hyperbolic radial rule. Isolates the geometry factor."""

    name = "euclidean_star"
    structure = "higher_order"
    space = "euclidean"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def fit(self, sc: SimplicialComplex, seed: int = 0) -> EmbeddingResult:
        require_connected(sc)
        full = _spectral_2d(_star_expansion_affinity(sc), seed)
        result = EmbeddingResult(coords=full[: sc.n_nodes], space="euclidean", layer2=full[sc.n_nodes:],
                                 name=self.name, meta={"seed": seed})
        result.check_non_degenerate()
        return result


def build_registry(params: Dict[str, Any], mu: float = 5.0, gamma: float = 1.0,
                   max_dim: Optional[int] = None) -> Dict[str, Embedder]:
    """All eight embedders keyed by registry name (``"hho"`` for HHO), sharing one parameter set."""
    embedders: List[Embedder] = [
        ClassicalEmbedder(**params),
        HHOEmbedder(mu=mu, gamma=gamma, max_dim=max_dim, **params),
        WeightedCliqueExpansionEmbedder(**params),
        StarExpansionEmbedder(**params),
        ZhouHypergraphEmbedder(**params),
        EuclideanSpectralEmbedder(**params),
        EuclideanIsomapEmbedder(**params),
        EuclideanStarExpansionEmbedder(**params),
    ]
    return {("hho" if isinstance(e, HHOEmbedder) else e.name): e for e in embedders}
