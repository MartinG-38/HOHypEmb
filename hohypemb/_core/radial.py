"""Radius rule of the Poincaré-disk projection.

Ported from MLNHypEmb: EmbeddingMethod.py (MlHypEmb.Embedding) and EmbeddingMethodUnsym.py
(MlHypEmbUnsym_v3.Embedding), rule "degree" only.
Copyright (c) 2025, Martin Guillemaud, Sorbonne Université, INRIA (BSD-3-Clause).
"""
from typing import Optional, Sequence, Union

import networkx as nx
import numpy as np

_REJECTED = {
    "order": (
        "radius='order' is not supported: MLNHypEmb computes it with N = number of nodes for the "
        "classical embedding but N = nodes + layer-2 points for HHO, so the two embeddings' radii "
        "would not be comparable."
    ),
    "logdegree": (
        "radius='logdegree' is not supported: MLNHypEmb uses 1 - tanh(log(s+1)/beta) for the "
        "classical embedding but 1/(s+1)**beta for HHO, so the two embeddings' radii would not be "
        "comparable."
    ),
}


def check_radius_rule(rule: str) -> None:
    """Raise ``ValueError`` unless ``rule == "degree"``."""
    if rule == "degree":
        return
    if rule in _REJECTED:
        raise ValueError(_REJECTED[rule])
    raise ValueError(f"radius must be 'degree', got {rule!r}")


def layer_strength(G: nx.Graph, weight: Optional[str]) -> np.ndarray:
    """Weighted degree of every node, in ``G.nodes()`` order."""
    return np.array(list(dict(nx.degree(G, weight=weight)).values()), dtype=float)


def degree_radius(strength: Union[Sequence[float], np.ndarray], beta: float) -> np.ndarray:
    """Euclidean radius in the Poincaré disk: ``1 - tanh(strength / beta)``."""
    return 1 - np.tanh(np.asarray(strength, dtype=float) / beta)
