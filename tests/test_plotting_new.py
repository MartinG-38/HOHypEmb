import os

import matplotlib
import networkx as nx
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hohypemb import ClassicalEmbedder, HHOEmbedder, SimplicialComplex, markers, plotting  # noqa: E402

PARAMS = dict(preweight=True, beta=100.0, method="Isomap", beta_dist=2)


@pytest.fixture(scope="module")
def setup():
    G = nx.Graph(nx.karate_club_graph().edges())
    triangles = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    sc = SimplicialComplex.from_graph(G, triangles[::2] + [(0, 1, 2, 3)], name="karate")
    cl = ClassicalEmbedder(**PARAMS).fit(sc, seed=0)
    hho = HHOEmbedder(mu=5, gamma=0.5, **PARAMS).fit(sc, seed=0)
    return sc, cl, hho


def _width(fig):
    return fig.get_size_inches()[0]


def test_poincare_draws_all_simplices(setup):
    sc, _, hho = setup
    fig, ax = plt.subplots()
    plotting.plot_poincare(ax, sc, hho)
    polys = [c for c in ax.collections if isinstance(c, matplotlib.collections.PolyCollection)]
    assert len(polys) == 1 and len(polys[0].get_paths()) == len(sc.higher_order_simplices())
    plt.close(fig)


def test_comparison_and_synthesis_figures(setup, tmp_path, monkeypatch):
    sc, cl, hho = setup
    before = set(os.listdir(tmp_path))
    monkeypatch.chdir(tmp_path)
    fig = plotting.plot_comparison(sc, cl, hho, node_colors=["C0"] * sc.n_nodes, legend={"club": "C0"})
    assert len(fig.axes) == 2 and _width(fig) <= 10
    syn = plotting.plot_synthesis(markers.area_ratio(sc, cl, hho), markers.surface_vs_empty_areas(sc, hho),
                                  markers.global_distortion(sc, cl), markers.global_distortion(sc, hho))
    assert len(syn.axes) == 3 and _width(syn) <= 10
    assert set(os.listdir(tmp_path)) == before       # plotting never writes files
    plt.close("all")


def test_marker_panels(setup):
    sc, cl, hho = setup
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    plotting.plot_area_ratio(axes[0, 0], markers.area_ratio(sc, cl, hho))
    plotting.plot_surface_vs_empty(axes[0, 1], markers.surface_vs_empty_areas(sc, hho))
    plotting.plot_distortion(axes[0, 2], markers.global_distortion(sc, cl), markers.global_distortion(sc, hho))
    plotting.plot_partner_precision(axes[1, 0], markers.partner_precision_at_k(sc, cl, 3),
                                    markers.partner_precision_at_k(sc, hho, 3))
    plotting.plot_highlighted_groups(axes[1, 1], sc, hho, [[0, 1, 2], [30, 32, 33]], ["crimson", "navy"],
                                     labels=["g1", "g2"])
    for ax in axes.flat[:5]:
        assert ax.has_data()
    plt.close(fig)


def test_triangle_clustering_figure(setup):
    sc, cl, hho = setup
    fig = plotting.plot_triangle_clustering(markers.triangle_clustering(sc, cl),
                                            markers.triangle_clustering(sc, hho))
    assert len(fig.axes) == 2 and _width(fig) <= 10
    plt.close(fig)


def test_empty_results_do_not_crash():
    fig, ax = plt.subplots()
    empty = markers.AreaRatioResult(np.empty((0, 3), int), np.empty(0), np.empty(0), np.empty(0), 0,
                                     float("nan"), float("nan"), float("nan"))
    plotting.plot_area_ratio(ax, empty)
    assert any("no filled triangle" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_area_ratio_label_counts_what_is_plotted():
    """Non-positive ratios cannot go on a log axis; the label must say how many were left out."""
    ratio = np.array([0.5, 2.0, 0.25, -1e-3, 0.0])
    res = markers.AreaRatioResult(np.zeros((5, 3), int), np.ones(5), ratio, ratio, 0,
                                  float(np.median(ratio)), float(np.mean(ratio < 1)), float("nan"))
    fig, ax = plt.subplots()
    plotting.plot_area_ratio(ax, res)
    text = " ".join(t.get_text() for t in ax.texts)
    assert "n = 3 plotted" in text and "2 non-positive ratios not shown" in text
    assert sum(p.get_height() for p in ax.patches) == 3
    plt.close(fig)
