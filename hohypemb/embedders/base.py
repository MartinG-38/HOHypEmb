"""EmbeddingResult and the Embedder interface."""
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np

from hohypemb import geometry as geo
from hohypemb.complex import SimplicialComplex

DENSE_WARNING_SIZE = 5000


@dataclass(eq=False)
class EmbeddingResult:
    """What every embedder returns. Identity equality: arrays make structural equality ill-defined.

    Attributes
    ----------
    coords
        ``(n_nodes, 2)`` node coordinates, in the Poincaré disk or the Euclidean plane (``space``).
    space
        ``"poincare"`` or ``"euclidean"``.
    layer2
        Second-layer points (one per layer-2 simplex for HHO, per simplex for the star expansion),
        or ``None``. For HHO, the points of simplices that share at most one node with every other
        simplex have radius exactly 1 (on the boundary circle), see ``HHOEmbedder``.
    name
        Name of the embedder that produced the result.
    meta
        Parameters and seed of the fit.
    """

    coords: np.ndarray
    space: str
    layer2: Optional[np.ndarray]
    name: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def distance_matrix(self) -> np.ndarray:
        if self.space == "poincare":
            return geo.hyp_dist_matrix(self.coords)
        if self.space == "euclidean":
            return geo.euclidean_dist_matrix(self.coords)
        raise ValueError(f"unknown space {self.space!r}")

    def angular_matrix(self) -> np.ndarray:
        return geo.angular_dist_matrix(self.coords)

    def radii(self) -> np.ndarray:
        return geo.radii(self.coords)

    def check_non_degenerate(self) -> None:
        """Fail on non-finite or out-of-disk coordinates; warn when all radii are equal."""
        if not np.all(np.isfinite(self.coords)):
            raise ValueError(f"{self.name}: embedding contains non-finite coordinates")
        r = np.linalg.norm(self.coords, axis=1)
        if self.space == "poincare" and np.any(r >= 1.0):
            raise ValueError(f"{self.name}: coordinates on or outside the unit circle")
        if float(np.ptp(r)) < 1e-9:
            warnings.warn(
                f"{self.name}: all radii are identical ({r[0]:.3e}). Expected for a degree-regular "
                "graph; otherwise the radial rule has saturated -- recalibrate beta against the "
                "graph's degree scale.", UserWarning, stacklevel=2)


class Embedder(ABC):
    name: str = "abstract"
    structure: str = "pairwise"      # "pairwise" | "higher_order"
    space: str = "poincare"          # "poincare" | "euclidean"

    @abstractmethod
    def fit(self, sc: SimplicialComplex, seed: int = 0) -> EmbeddingResult:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"


def require_connected(sc: SimplicialComplex) -> None:
    """Spectral and Isomap embeddings degenerate on a disconnected graph."""
    G = sc.one_skeleton
    if sc.n_nodes > 1 and not nx.is_connected(G):
        n_comp = nx.number_connected_components(G)
        raise ValueError(
            f"{sc.name or 'complex'}: the 1-skeleton has {n_comp} connected components; spectral and "
            "Isomap embeddings degenerate on a disconnected graph. Use sc.giant_component() first.")


def warn_if_large(n_points: int) -> None:
    if n_points > DENSE_WARNING_SIZE:
        gigabytes = 4 * n_points ** 2 * 8 / 1e9
        warnings.warn(
            f"Embedding {n_points} points with dense matrices needs roughly {gigabytes:.1f} GB of memory "
            "and may take minutes (HOHypEmb uses dense matrices).", UserWarning, stacklevel=3)
