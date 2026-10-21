"""RA pre-weighting of the pairwise layer.

Ported from MLNHypEmb: EmbeddingMethod.py (MlHypEmb.Load_data, pre-weighting loop) and
EmbeddingMethodUnsym.py (MlHypEmbUnsym_v3.Load_data, first-layer pre-weighting).
Copyright (c) 2025, Martin Guillemaud, Sorbonne Université, INRIA (BSD-3-Clause).
"""
from typing import Tuple

import networkx as nx
import numpy as np


def ra_preweight(mat: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
    """Replace each edge weight by (d_u + d_v + d_u d_v) / (1 + CN_uv), then take log(w + 1).

    Degrees and common-neighbour counts are unweighted (pure topology), exactly as in
    MLNHypEmb. Returns the re-weighted graph (used for the radius rule) and the matrix
    ``log(W + 1)`` (used as the pairwise block of the global matrix).
    """
    G = nx.from_numpy_array(mat)
    for u, v in G.edges():
        cn = len(list(nx.common_neighbors(G, u, v)))
        d_u = nx.degree(G, u)
        d_v = nx.degree(G, v)
        G[u][v]["weight"] = (d_u + d_v + d_u * d_v) / (1 + cn)
    W = np.log(nx.adjacency_matrix(G).toarray() + 1)
    return G, W
