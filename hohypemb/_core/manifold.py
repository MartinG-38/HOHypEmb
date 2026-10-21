"""Global matrix -> 2-D coordinates -> Poincaré disk.

Ported from MLNHypEmb: EmbeddingMethod.py (MlHypEmb.Load_data / Embedding) and
EmbeddingMethodUnsym.py (MlHypEmbUnsym_v3.Load_data / Embedding). The numerical path is
unchanged; the global-RNG seeding of the original is replaced by an explicit
``random_state``.
Copyright (c) 2025, Martin Guillemaud, Sorbonne Université, INRIA (BSD-3-Clause).
"""
from typing import Any, Dict, List, Sequence

import networkx as nx
import numpy as np
from sklearn.manifold import Isomap, spectral_embedding

from hohypemb._core.preweight import ra_preweight
from hohypemb._core.radial import check_radius_rule, degree_radius, layer_strength
from hohypemb.presets import EMBEDDING_PARAMS, PARAM_NAMES

VALID_METRICS = ("euclidean", "manhattan", "cosine", "hamming", "jaccard", "minkowski",
                 "chebyshev", "seuclidean", "sqeuclidean", "cityblock", "precomputed")
METHODS = ("Isomap", "Spectral")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate embedding parameters and fill missing ones from ``EMBEDDING_PARAMS["default"]``."""
    unknown = sorted(set(params) - set(PARAM_NAMES))
    if unknown:
        raise TypeError(f"Unknown embedding parameter(s): {unknown}. Allowed: {list(PARAM_NAMES)}")
    p = dict(EMBEDDING_PARAMS["default"])
    p.update(params)
    if not isinstance(p["preweight"], bool):
        raise TypeError(f"preweight must be a bool, got {p['preweight']!r}")
    if not (p["weight"] is None or isinstance(p["weight"], str)):
        raise TypeError(f"weight must be a str or None, got {p['weight']!r}")
    for key in ("beta", "beta_dist"):
        if not _is_number(p[key]) or p[key] <= 0:
            raise TypeError(f"{key} must be a positive number, got {p[key]!r}")
    if not (p["eta"] == "auto" or _is_number(p["eta"])):
        raise TypeError(f"eta must be a number or 'auto', got {p['eta']!r}")
    if p["method"] not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {p['method']!r}")
    if p["metric"] not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS}, got {p['metric']!r}")
    if not (p["n_neighbors"] == "auto" or (isinstance(p["n_neighbors"], int) and p["n_neighbors"] > 0)):
        raise TypeError(f"n_neighbors must be a positive int or 'auto', got {p['n_neighbors']!r}")
    check_radius_rule(p["radius"])
    return p


def reduce_2d(L_global: np.ndarray, method: str, beta_dist: float, metric: str, n_neighbors: Any,
              seed: int, n_jobs: int = -1) -> np.ndarray:
    """Two-dimensional coordinates of every row of the global matrix."""
    if method == "Spectral":
        return spectral_embedding(L_global, n_components=2, random_state=seed)
    k = L_global.shape[0] - 1 if n_neighbors == "auto" else n_neighbors
    model = Isomap(n_components=2, n_neighbors=k, eigen_solver="dense", n_jobs=n_jobs,
                   metric=metric, neighbors_algorithm="brute")
    return model.fit_transform(np.exp(-L_global / beta_dist))


def embed_layers(matrices: Sequence[np.ndarray], L_global: np.ndarray, *, seed: int,
                 n_jobs: int = -1, **params: Any) -> List[np.ndarray]:
    """Embed a multi-layer global matrix; return one Poincaré-disk array per layer.

    ``matrices[i]`` is the connectivity of layer ``i`` (layer 0 = the pairwise graph);
    ``L_global`` holds the inter-layer couplings (its diagonal blocks are overwritten by
    the layer matrices, layer 0 pre-weighted when ``preweight`` is true).
    """
    p = check_params(params)
    mats = [np.asarray(m, dtype=float) for m in matrices]
    L = np.array(L_global, dtype=float, copy=True)
    graphs = [nx.from_numpy_array(m) for m in mats]
    blocks = list(mats)
    if p["preweight"]:
        graphs[0], blocks[0] = ra_preweight(mats[0])
    sizes = [m.shape[0] for m in mats]
    ends = np.cumsum(sizes)
    starts = ends - np.array(sizes)
    total = int(ends[-1])
    if L.shape != (total, total):
        raise ValueError(f"L_global must have shape {(total, total)}, got {L.shape}")
    for W, s, e in zip(blocks, starts, ends):
        L[s:e, s:e] = W
    coords = reduce_2d(L, p["method"], p["beta_dist"], p["metric"], p["n_neighbors"], seed, n_jobs)
    out = []
    for G, s, e in zip(graphs, starts, ends):
        emb = coords[s:e]
        unit = emb / np.sqrt(np.sum(emb ** 2, axis=1, keepdims=True))
        radius = degree_radius(layer_strength(G, p["weight"]), p["beta"])
        out.append(unit * radius[:, np.newaxis])
    return out
