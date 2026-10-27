"""Publication figure style and reusable panels.

All figure text is in English. Edges and triangles are drawn with
a single collection each -- a per-edge ax.plot loop is both unreadable and slow
at the scale of these networks.
"""

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch

from hohypemb import geometry as geo
from hohypemb import markers as mk
from hohypemb.complex import SimplicialComplex
from hohypemb.embedders.base import EmbeddingResult

# Okabe-Ito derived, colour-blind safe. One stable colour per registry key so
# that a method keeps its identity across every figure in the paper.
PALETTE: Dict[str, str] = {
    "hho": "#d55e00",
    "classical": "#0072b2",
    "weighted_clique": "#56b4e9",
    "star_expansion": "#009e73",
    "zhou_hypergraph": "#cc79a7",
    "euclidean_spectral": "#999999",
    "euclidean_isomap": "#666666",
    "euclidean_star": "#e69f00",
}


def set_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "pdf.fonttype": 42,          # embed TrueType, keeps text editable in Illustrator
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })


def _geodesic_polygon(points: np.ndarray) -> np.ndarray:
    """Closed polygon through ``points`` (ordered by angle around their Klein barycenter)."""
    center = geo.klein_barycenter(points)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    ring = points[order]
    pieces = [geo.poincare_geodesic(ring[i], ring[(i + 1) % len(ring)], 8)[:-1] for i in range(len(ring))]
    return np.vstack(pieces)


def plot_poincare(ax, sc: SimplicialComplex, res: EmbeddingResult, node_colors: Optional[Sequence] = None,
                  show_simplices: bool = True, simplex_color: str = PALETTE["hho"], max_edges: int = 8000,
                  title: str = "", node_scale: float = 1.0) -> None:
    """One Poincaré-disk panel: unit circle, geodesic edges, filled simplices, nodes.

    Edges are one ``LineCollection`` and simplices one ``PolyCollection``. Simplices with more
    than three vertices are drawn as the geodesic polygon of their vertices. Node markers have an
    area proportional to sqrt(degree + 1); ``node_scale`` multiplies that area (1.0 keeps the default).
    """
    coords = res.coords
    G = sc.one_skeleton
    ax.plot([0], [0], "o", color="black", markersize=2.5, zorder=4)
    ax.add_artist(plt.Circle((0, 0), 1, edgecolor="0.6", facecolor="none", linewidth=0.9))

    edges = list(G.edges())
    if len(edges) > max_edges:
        rng = np.random.default_rng(0)
        edges = [edges[i] for i in rng.choice(len(edges), size=max_edges, replace=False)]
    if edges:
        segments = [geo.poincare_geodesic(coords[u], coords[v], num_points=12) for u, v in edges]
        ax.add_collection(LineCollection(segments, colors=[(0.35, 0.35, 0.35, 0.25)], linewidths=0.45, zorder=1))

    if show_simplices:
        polys = [_geodesic_polygon(coords[list(s)]) for s in sc.higher_order_simplices(min_dim=2)]
        if polys:
            ax.add_collection(PolyCollection(polys, facecolors=simplex_color, edgecolors="none",
                                             alpha=0.16, zorder=2))

    degrees = np.array([G.degree(i) for i in range(sc.n_nodes)], dtype=float)
    ax.scatter(coords[:, 0], coords[:, 1], s=np.sqrt(degrees + 1) * 9 * node_scale,
               c=node_colors if node_colors is not None else PALETTE["classical"],
               edgecolors="black", linewidths=0.3, zorder=3, alpha=0.9)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    if title:
        ax.set_title(title)


def plot_response_surface(ax, mu_values, gamma_values, grid,
                          cbar_label: str = "") -> None:
    """Heatmap of a metric over the (mu, gamma) grid. ``grid`` is (n_gamma, n_mu).

    Cells are drawn at their INDEX position (0, 1, 2, ...), with tick labels
    naming each sampled value, rather than at the value itself via
    ``imshow(extent=...)``. The mu/gamma grids sampled by the response-surface
    experiment are non-uniform (e.g. mu = [0.5, 1, 2, 3, 5, 7, 10, 15, 20]),
    and ``extent`` spreads columns UNIFORMLY across the given range regardless
    of the actual value spacing -- so a non-uniform grid's cells are drawn at
    the wrong positions relative to their true values (e.g. the
    true optimum at mu=3 was drawn at mu=8.08, disagreeing with the adjacent
    line panel by a factor of 2.7). Indexing by position is honest about what
    the grid actually is -- a set of sampled values, not a continuum -- and
    guarantees the drawn position of every cell matches its data value,
    uniform or not.
    """
    mu_values = np.asarray(mu_values)
    gamma_values = np.asarray(gamma_values)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(mu_values)))
    ax.set_xticklabels([f"{v:g}" for v in mu_values])
    ax.set_yticks(range(len(gamma_values)))
    ax.set_yticklabels([f"{v:g}" for v in gamma_values])
    ax.set_xlabel(r"coupling strength $\mu$")
    ax.set_ylabel(r"simplex–simplex coupling $\gamma$")
    cbar = ax.figure.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)


def plot_slice_with_ci(ax, x, mean, lo, hi, label: str, color: str) -> None:
    ax.plot(x, mean, color=color, linewidth=1.6, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.20, linewidth=0)


def plot_ablation_grid(ax, table: pd.DataFrame, metric_col: str,
                       err_cols: Optional[Sequence[str]] = None) -> None:
    """Horizontal bars, one per method, coloured by PALETTE and sorted by metric."""
    ordered = table.sort_values(metric_col, ascending=True)
    colors = [PALETTE.get(m, "#444444") for m in ordered["method"]]
    err = None
    if err_cols is not None:
        err = np.vstack([
            ordered[metric_col] - ordered[err_cols[0]],
            ordered[err_cols[1]] - ordered[metric_col],
        ])
    ax.barh(ordered["method"], ordered[metric_col], color=colors,
            edgecolor="black", linewidth=0.5, xerr=err, capsize=3)
    ax.set_xlabel(metric_col.replace("_", " "))


def plot_paired_bars(ax, labels, values, ci_low, ci_high, colors) -> None:
    err = np.vstack([np.asarray(values) - np.asarray(ci_low),
                     np.asarray(ci_high) - np.asarray(values)])
    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5,
           yerr=err, capsize=4)
    ax.axhline(0.0, color="black", linewidth=0.8)


def _no_data(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=9)


def _binned_medians(x, y, n_bins: int = 8):
    """Median of ``y`` in equal-count bins of ``x`` (bins with fewer than 3 points are skipped)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.array([]), np.array([])
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return np.array([]), np.array([])
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    cx, cy = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= 3:
            cx.append(np.median(x[sel]))
            cy.append(np.median(y[sel]))
    return np.array(cx), np.array(cy)


def plot_comparison(sc: SimplicialComplex, classical: EmbeddingResult, hho: EmbeddingResult,
                    node_colors: Optional[Sequence] = None, legend: Optional[Dict[str, str]] = None,
                    titles: Sequence[str] = ("Classical", "HHO"), node_scale: float = 1.0,
                    figsize: Sequence[float] = (10, 4.8)):
    """Classical | HHO side by side (``figsize`` in inches, default 10 x 4.8).

    ``legend`` maps label -> colour; ``node_scale`` multiplies the node marker area in both panels.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, res, title in zip(axes, (classical, hho), titles):
        plot_poincare(ax, sc, res, node_colors=node_colors, title=title, node_scale=node_scale)
    if legend:
        handles = [Patch(facecolor=c, edgecolor="black", label=lab) for lab, c in legend.items()]
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_area_ratio(ax, res: "mk.AreaRatioResult") -> None:
    """Histogram of log10(HHO / classical) filled-triangle area ratios.

    Only positive ratios can be drawn on the log axis; the annotation gives the number plotted and
    the number of non-positive ratios left out. The median and the "% smaller" figure are those of
    ``res``, i.e. over all ratios, non-positive ones included.
    """
    ratio = res.ratio[res.ratio > 0]
    if len(ratio) == 0:
        _no_data(ax, "no filled triangle")
        return
    heights, _, _ = ax.hist(np.log10(ratio), bins=30, color=PALETTE["classical"], edgecolor="black", alpha=0.85)
    ax.set_ylim(0, 1.45 * heights.max())  # headroom so the legend and the annotation clear the bars
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="ratio = 1")
    ax.set_xlabel("log10 area ratio (HHO / classical)")
    ax.set_ylabel("Filled triangles")
    n_hidden = len(res.ratio) - len(ratio)
    counts = f"n = {len(ratio)} plotted" + (f"\n({n_hidden} non-positive ratios not shown)" if n_hidden else "")
    ax.text(0.03, 0.95, f"median = {res.median_ratio:.3g}\n{100 * res.frac_below_1:.1f}% smaller under HHO\n"
            + counts, transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9), zorder=5)
    ax.legend(loc="upper right", fontsize=8)


def plot_surface_vs_empty(ax, res: "mk.SurfaceEmptyResult") -> None:
    filled = res.area_filled[res.area_filled > 0]
    empty = res.area_empty[res.area_empty > 0]
    if len(filled) == 0 or len(empty) == 0:
        _no_data(ax, "needs both filled and empty triangles")
        return
    lf, le = np.log10(filled), np.log10(empty)
    bins = np.linspace(min(lf.min(), le.min()), max(lf.max(), le.max()), 30)
    ax.hist(le, bins=bins, density=True, alpha=0.6, color="0.55", label=f"empty (n={len(empty)})")
    ax.hist(lf, bins=bins, density=True, alpha=0.75, color=PALETTE["hho"], label=f"filled (n={len(filled)})")
    ax.set_xlabel("log10 hyperbolic area")
    ax.set_ylabel("Density")
    ax.legend()


def plot_distortion(ax, classical: "mk.DistortionResult", hho: "mk.DistortionResult") -> None:
    for res, shift, color in ((classical, -0.15, PALETTE["classical"]), (hho, 0.15, PALETTE["hho"])):
        if len(res.hop_values):
            ax.boxplot(list(res.per_hop), positions=res.hop_values + shift, widths=0.28, showfliers=False,
                       patch_artist=True, medianprops={"color": "black"},
                       boxprops={"facecolor": color, "alpha": 0.8})
    hops = sorted(set(classical.hop_values.tolist()) | set(hho.hop_values.tolist()))
    ax.set_xticks(hops)
    ax.set_xticklabels([str(h) for h in hops])
    ax.set_xlabel("Graph distance (hops)")
    ax.set_ylabel("Embedding distance")
    ax.legend(handles=[Patch(facecolor=PALETTE["classical"], label=f"Classical (rho={classical.spearman_rho:.2f})"),
                       Patch(facecolor=PALETTE["hho"], label=f"HHO (rho={hho.spearman_rho:.2f})")])


def plot_triangle_clustering(classical: "mk.TriangleClusteringResult", hho: "mk.TriangleClusteringResult",
                             axes=None):
    """Left: classical, all triangles. Right: HHO, filled vs empty. Log area, binned medians."""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = axes[0].figure
    panels = [(axes[0], classical, [("all", np.ones(len(classical.area), dtype=bool), PALETTE["classical"])],
               "Classical"),
              (axes[1], hho, [("empty", ~hho.filled, "0.55"), ("filled", hho.filled, PALETTE["hho"])],
               "HHO: filled vs empty")]
    for ax, res, groups, title in panels:
        for group, mask, color in groups:
            m = mask & np.isfinite(res.area) & (res.area > 0)
            if not m.any():
                continue
            rho = res.stats[group]["rho_area"]
            ax.scatter(res.clustering_index[m], res.area[m], s=12, alpha=0.5, color=color, edgecolors="none",
                       label=f"{group} (n={m.sum()}, rho={rho:.2f})")
            cx, cy = _binned_medians(res.clustering_index[m], res.area[m])
            ax.plot(cx, cy, color="black", linewidth=1.5, marker="o", markersize=3)
        if ax.has_data():
            ax.set_yscale("log")
            ax.legend(fontsize=8)
        ax.set_xlabel("Triangle clustering index")
        ax.set_ylabel("Hyperbolic area")
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_partner_precision(ax, classical: "mk.PartnerPrecisionResult", hho: "mk.PartnerPrecisionResult") -> None:
    bars = ax.bar(["Classical", "HHO"], [classical.mean, hho.mean],
                  color=[PALETTE["classical"], PALETTE["hho"]], edgecolor="black")
    ax.bar_label(bars, fmt="%.3f")
    ax.set_ylabel(f"Mean precision@{classical.k}")


def plot_highlighted_groups(ax, sc: SimplicialComplex, res: EmbeddingResult, groups: Sequence[Sequence[int]],
                            colors: Sequence[str], labels: Optional[Sequence[str]] = None, node_scale: float = 1.0) -> None:
    """Whole network in light grey, each group's nodes in its colour (case-study figure).

    ``node_scale`` multiplies the marker area of the background nodes and of the group markers.
    """
    plot_poincare(ax, sc, res, node_colors=["0.85"] * sc.n_nodes, show_simplices=False, node_scale=node_scale)
    for i, (group, color) in enumerate(zip(groups, colors)):
        pts = res.coords[list(group)]
        ax.scatter(pts[:, 0], pts[:, 1], s=40 * node_scale, color=color, edgecolors="black", linewidths=0.4, zorder=5,
                   label=None if labels is None else labels[i])
    if labels is not None:
        ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)


def plot_synthesis(area_ratio_res: "mk.AreaRatioResult", surface_empty_res: "mk.SurfaceEmptyResult",
                   distortion_cl: "mk.DistortionResult", distortion_hho: "mk.DistortionResult"):
    """Three-panel synthesis figure: areas, filled vs empty, distortion (10 x 3.4 in)."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    plot_area_ratio(axes[0], area_ratio_res)
    plot_surface_vs_empty(axes[1], surface_empty_res)
    plot_distortion(axes[2], distortion_cl, distortion_hho)
    for ax, letter in zip(axes, "ABC"):
        ax.set_title(f"({letter})", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig
