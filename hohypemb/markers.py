"""Markers quantifying what HHO changes relative to the classical embedding.

Every function is pure: it computes; it does not print, plot or save. Each returns a frozen
dataclass whose ``to_dict()`` gives its scalar statistics.
All p-values are two-sided. Degenerate triangles (non-finite area) are excluded and counted.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from hohypemb import geometry as geo
from hohypemb.complex import SimplicialComplex
from hohypemb.embedders.base import EmbeddingResult
from hohypemb.evaluation import partial_spearman, rank_normalised_silhouette, silhouette_with_ci

NAN = float("nan")


def _median(x) -> float:
    return float(np.median(x)) if len(x) else NAN


def _require_poincare(*results: EmbeddingResult) -> None:
    for r in results:
        if r.space != "poincare":
            raise ValueError(f"{r.name}: this marker needs a Poincaré-disk embedding, got space={r.space!r}")


def _mannwhitney_p(a, b) -> float:
    if len(a) == 0 or len(b) == 0:
        return NAN
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def _spearman(x, y) -> Tuple[float, float]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return NAN, NAN
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def _triangles(tris) -> np.ndarray:
    return np.asarray(tris, dtype=int).reshape(-1, 3)


@dataclass(frozen=True, eq=False)
class AreaRatioResult:
    triangles: np.ndarray
    area_classical: np.ndarray
    area_hho: np.ndarray
    ratio: np.ndarray
    n_excluded: int
    median_ratio: float
    frac_below_1: float
    wilcoxon_p: float

    def to_dict(self) -> Dict[str, float]:
        return {"n": float(len(self.ratio)), "n_excluded": float(self.n_excluded),
                "median_ratio": self.median_ratio, "frac_below_1": self.frac_below_1,
                "wilcoxon_p": self.wilcoxon_p}


def area_ratio(sc: SimplicialComplex, classical: EmbeddingResult, hho: EmbeddingResult) -> AreaRatioResult:
    """Area of every filled triangle under both embeddings; paired Wilcoxon signed-rank test."""
    _require_poincare(classical, hho)
    tris = _triangles(sc.filled_triangles())
    a_cl = geo.triangle_areas(classical.coords, tris)
    a_hho = geo.triangle_areas(hho.coords, tris)
    ok = np.isfinite(a_cl) & np.isfinite(a_hho) & (a_cl > 0)
    ratio = a_hho[ok] / a_cl[ok]
    diff = a_hho[ok] - a_cl[ok]
    p = float(wilcoxon(a_hho[ok], a_cl[ok]).pvalue) if np.count_nonzero(diff) > 0 else NAN
    return AreaRatioResult(triangles=tris[ok], area_classical=a_cl[ok], area_hho=a_hho[ok], ratio=ratio,
                           n_excluded=int((~ok).sum()), median_ratio=_median(ratio),
                           frac_below_1=float(np.mean(ratio < 1)) if len(ratio) else NAN, wilcoxon_p=p)


@dataclass(frozen=True, eq=False)
class SurfaceEmptyResult:
    area_filled: np.ndarray
    area_empty: np.ndarray
    n_excluded: int
    median_filled: float
    median_empty: float
    fold: float
    mannwhitney_p: float

    def to_dict(self) -> Dict[str, float]:
        return {"n_filled": float(len(self.area_filled)), "n_empty": float(len(self.area_empty)),
                "n_excluded": float(self.n_excluded), "median_filled": self.median_filled,
                "median_empty": self.median_empty, "fold": self.fold, "mannwhitney_p": self.mannwhitney_p}


def surface_vs_empty_areas(sc: SimplicialComplex, res: EmbeddingResult) -> SurfaceEmptyResult:
    """Areas of filled vs empty triangles within ONE embedding; Mann-Whitney U test."""
    _require_poincare(res)
    a_f = geo.triangle_areas(res.coords, _triangles(sc.filled_triangles()))
    a_e = geo.triangle_areas(res.coords, _triangles(sc.empty_triangles()))
    ok_f, ok_e = np.isfinite(a_f), np.isfinite(a_e)
    a_f, a_e = a_f[ok_f], a_e[ok_e]
    med_f, med_e = _median(a_f), _median(a_e)
    fold = med_e / med_f if (np.isfinite(med_f) and med_f > 0 and np.isfinite(med_e)) else NAN
    return SurfaceEmptyResult(area_filled=a_f, area_empty=a_e,
                              n_excluded=int((~ok_f).sum() + (~ok_e).sum()), median_filled=med_f,
                              median_empty=med_e, fold=fold, mannwhitney_p=_mannwhitney_p(a_f, a_e))


@dataclass(frozen=True, eq=False)
class EdgeDistanceResult:
    dist_edges: np.ndarray
    dist_nonedges: np.ndarray
    median_edges: float
    median_nonedges: float
    ratio: float
    mannwhitney_p: float

    def to_dict(self) -> Dict[str, float]:
        return {"median_edges": self.median_edges, "median_nonedges": self.median_nonedges,
                "ratio": self.ratio, "mannwhitney_p": self.mannwhitney_p}


def edge_vs_nonedge_distances(sc: SimplicialComplex, res: EmbeddingResult) -> EdgeDistanceResult:
    """Embedding distance of connected vs unconnected node pairs; Mann-Whitney U test."""
    D = res.distance_matrix()
    iu = np.triu_indices(sc.n_nodes, k=1)
    is_edge = (sc.adjacency > 0)[iu]
    d = D[iu]
    d_e, d_n = d[is_edge], d[~is_edge]
    med_e, med_n = _median(d_e), _median(d_n)
    ratio = med_n / med_e if (np.isfinite(med_e) and med_e > 0) else NAN
    return EdgeDistanceResult(dist_edges=d_e, dist_nonedges=d_n, median_edges=med_e, median_nonedges=med_n,
                              ratio=ratio, mannwhitney_p=_mannwhitney_p(d_e, d_n))


@dataclass(frozen=True, eq=False)
class DistortionResult:
    hop_values: np.ndarray
    per_hop: Tuple[np.ndarray, ...]
    spearman_rho: float
    spearman_p: float

    def to_dict(self) -> Dict[str, float]:
        return {"spearman_rho": self.spearman_rho, "spearman_p": self.spearman_p}


def global_distortion(sc: SimplicialComplex, res: EmbeddingResult) -> DistortionResult:
    """Spearman correlation between hop distance and embedding distance over all node pairs."""
    n = sc.n_nodes
    hops = np.full((n, n), np.inf)
    for i, row in nx.all_pairs_shortest_path_length(sc.one_skeleton):
        for j, h in row.items():
            hops[i, j] = h
    iu = np.triu_indices(n, k=1)
    h, d = hops[iu], res.distance_matrix()[iu]
    ok = np.isfinite(h)
    h, d = h[ok], d[ok]
    rho, p = _spearman(h, d)
    hop_values = np.unique(h).astype(int)
    return DistortionResult(hop_values=hop_values, per_hop=tuple(d[h == v] for v in hop_values),
                            spearman_rho=rho, spearman_p=p)


@dataclass(frozen=True, eq=False)
class LabelSeparationResult:
    n: int
    n_groups: int
    silhouette: float
    silhouette_lo: float
    silhouette_hi: float
    silhouette_rank: float
    silhouette_rank_lo: float
    silhouette_rank_hi: float

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def label_separation(res: EmbeddingResult, labels, n_boot: int = 300, seed: int = 0,
                     exclude: Sequence[Hashable] = (), min_group_size: int = 2) -> LabelSeparationResult:
    """Raw and rank-normalised silhouette of ``labels`` on the embedding distances, with bootstrap CIs.

    Nodes whose label is ``None``, in ``exclude``, or in a group smaller than ``min_group_size``
    are left out.
    """
    labels = np.asarray(labels, dtype=object)
    excluded = set(exclude)
    keep = np.array([lab is not None and lab not in excluded for lab in labels], dtype=bool)
    counts = Counter(labels[keep])
    keep &= np.array([counts.get(lab, 0) >= min_group_size for lab in labels], dtype=bool)
    n_groups = len(set(labels[keep]))
    if n_groups < 2:
        return LabelSeparationResult(int(keep.sum()), n_groups, NAN, NAN, NAN, NAN, NAN, NAN)
    D = res.distance_matrix()[np.ix_(keep, keep)]
    raw = silhouette_with_ci(D, labels[keep], n_boot=n_boot, seed=seed)
    rank = rank_normalised_silhouette(D, labels[keep], n_boot=n_boot, seed=seed)
    return LabelSeparationResult(int(keep.sum()), n_groups, raw["score"], raw["lo"], raw["hi"],
                                 rank["score"], rank["lo"], rank["hi"])


@dataclass(frozen=True, eq=False)
class PartnerPrecisionResult:
    k: int
    eligible: np.ndarray
    precision: np.ndarray
    mean: float
    median: float

    def to_dict(self) -> Dict[str, float]:
        return {"k": float(self.k), "n_eligible": float(len(self.eligible)), "mean": self.mean,
                "median": self.median}


def partner_precision_at_k(sc: SimplicialComplex, res: EmbeddingResult, k: int = 5,
                           groups: Optional[Sequence[Sequence[int]]] = None) -> PartnerPrecisionResult:
    """Fraction of each node's k nearest neighbours (embedding distance) that share a group with it.

    ``groups`` defaults to the input simplices of dimension >= 2. Only nodes with at least one
    partner are evaluated. Ties are broken by node index (stable sort).
    """
    n = sc.n_nodes
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < n_nodes ({n}), got {k}")
    if groups is None:
        groups = sc.higher_order_simplices(min_dim=2)
    mates = [set() for _ in range(n)]
    for g in groups:
        members = {int(v) for v in g}
        bad = [v for v in members if not 0 <= v < n]
        if bad:
            raise ValueError(f"group {sorted(members)} has node ids {bad} outside 0..{n - 1}")
        for v in members:
            mates[v] |= members - {v}
    eligible = np.array([i for i in range(n) if mates[i]], dtype=int)
    D = res.distance_matrix()
    precision = np.empty(len(eligible))
    for idx, i in enumerate(eligible):
        order = np.argsort(D[i], kind="stable")
        neighbours = order[order != i][:k]
        precision[idx] = len(set(neighbours.tolist()) & mates[i]) / k
    return PartnerPrecisionResult(k=k, eligible=eligible, precision=precision,
                                  mean=float(precision.mean()) if len(precision) else NAN,
                                  median=_median(precision))


def group_compactness(res: EmbeddingResult, groups: Sequence[Sequence[int]]) -> np.ndarray:
    """Mean pairwise embedding distance within each group (NaN for groups of fewer than 2 nodes)."""
    D = res.distance_matrix()
    return np.array([geo.group_compactness(D, list(g)) for g in groups], dtype=float)


GROUPS = ("all", "filled", "empty")


@dataclass(frozen=True, eq=False)
class TriangleClusteringResult:
    triangles: np.ndarray
    clustering_index: np.ndarray
    area: np.ndarray
    angular_gap: np.ndarray
    mean_radius: np.ndarray
    filled: np.ndarray
    stats: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, float]:
        return {f"{g}.{key}": value for g, d in self.stats.items() for key, value in d.items()}


def triangle_clustering(sc: SimplicialComplex, res: EmbeddingResult, min_triangles: int = 5
                        ) -> TriangleClusteringResult:
    """Triangle clustering index vs size of the triangle in the embedding.

    Clustering index of a triangle = mean unweighted local clustering coefficient of its three
    vertices. For each group ("all", "filled", "empty"): Spearman clustering-area, clustering-area
    controlling for the mean hyperbolic radius (partial), clustering-radius, clustering-angular gap.
    Statistics need at least ``min_triangles`` triangles; otherwise they are NaN.
    """
    _require_poincare(res)
    tris = _triangles(sc.triangles_in_skeleton())
    coefficients = nx.clustering(sc.one_skeleton, weight=None)
    per_node = np.array([coefficients[i] for i in range(sc.n_nodes)])
    index = per_node[tris].mean(axis=1)
    filled = np.array([sc.is_filled(t) for t in tris], dtype=bool)
    area = geo.triangle_areas(res.coords, tris)
    gap = geo.angular_gaps(res.coords, tris)
    radius = geo.hyperbolic_radius(res.coords)[tris].mean(axis=1)
    masks = {"all": np.ones(len(tris), dtype=bool), "filled": filled, "empty": ~filled}
    stats: Dict[str, Dict[str, float]] = {}
    for group, mask in masks.items():
        m_area = mask & np.isfinite(area) & (area > 0)
        s = {"n": float(mask.sum()), "n_area": float(m_area.sum())}
        if m_area.sum() >= min_triangles:
            s["rho_area"], s["p_area"] = _spearman(index[m_area], area[m_area])
            s["rho_area_partial"], s["p_area_partial"] = partial_spearman(index[m_area], area[m_area],
                                                                          radius[m_area])
            s["rho_radius"], s["p_radius"] = _spearman(index[m_area], radius[m_area])
        else:
            for key in ("rho_area", "p_area", "rho_area_partial", "p_area_partial", "rho_radius", "p_radius"):
                s[key] = NAN
        if mask.sum() >= min_triangles:
            s["rho_angular"], s["p_angular"] = _spearman(index[mask], gap[mask])
        else:
            s["rho_angular"], s["p_angular"] = NAN, NAN
        stats[group] = s
    return TriangleClusteringResult(triangles=tris, clustering_index=index, area=area, angular_gap=gap,
                                    mean_radius=radius, filled=filled, stats=stats)
