"""Hyperbolic embedding of the weighted 1-skeleton alone: the pairwise baseline of HHO."""
from typing import Any

import numpy as np

from hohypemb._core import check_params, embed_layers
from hohypemb.complex import SimplicialComplex
from hohypemb.embedders.base import Embedder, EmbeddingResult, require_connected, warn_if_large


class ClassicalEmbedder(Embedder):
    """Classical (pairwise) hyperbolic embedding, ``MlHypEmb`` in MLNHypEmb."""

    name = "classical"
    structure = "pairwise"
    space = "poincare"

    def __init__(self, **params: Any) -> None:
        self.params = check_params(params)

    def fit(self, sc: SimplicialComplex, seed: int = 0) -> EmbeddingResult:
        require_connected(sc)
        warn_if_large(sc.n_nodes)
        mat = sc.adjacency
        (coords,) = embed_layers([mat], np.zeros_like(mat), seed=seed, **self.params)
        result = EmbeddingResult(coords=coords, space="poincare", layer2=None, name=self.name,
                                 meta={"params": dict(self.params), "seed": seed})
        result.check_non_degenerate()
        return result
