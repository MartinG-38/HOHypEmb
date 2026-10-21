import networkx as nx
import numpy as np
import pytest

from hohypemb import ClassicalEmbedder, EmbeddingResult, HHOEmbedder, SimplicialComplex, markers

PARAMS = dict(preweight=True, beta=100.0, method="Isomap", beta_dist=2)


@pytest.fixture(scope="module")
def karate():
    G = nx.Graph(nx.karate_club_graph().edges())
    triangles = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    sc = SimplicialComplex.from_graph(G, triangles[::2], name="karate")
    return sc, ClassicalEmbedder(**PARAMS).fit(sc, seed=0), HHOEmbedder(mu=5, gamma=0.5, **PARAMS).fit(sc, seed=0)


def test_area_ratio(karate):
    sc, cl, hho = karate
    res = markers.area_ratio(sc, cl, hho)
    assert len(res.ratio) + res.n_excluded == len(sc.filled_triangles())
    np.testing.assert_allclose(res.ratio, res.area_hho / res.area_classical)
    d = res.to_dict()
    assert set(d) == {"n", "n_excluded", "median_ratio", "frac_below_1", "wilcoxon_p"}
    assert 0 <= d["frac_below_1"] <= 1


def test_surface_vs_empty(karate):
    sc, _, hho = karate
    res = markers.surface_vs_empty_areas(sc, hho)
    assert len(res.area_filled) + len(res.area_empty) + res.n_excluded == len(sc.triangles_in_skeleton())
    assert 0 <= res.mannwhitney_p <= 1


def test_edge_vs_nonedge(karate):
    sc, cl, _ = karate
    res = markers.edge_vs_nonedge_distances(sc, cl)
    assert len(res.dist_edges) == sc.one_skeleton.number_of_edges()
    assert len(res.dist_edges) + len(res.dist_nonedges) == 34 * 33 // 2
    assert res.ratio == pytest.approx(res.median_nonedges / res.median_edges)


def test_global_distortion(karate):
    sc, cl, _ = karate
    res = markers.global_distortion(sc, cl)
    assert list(res.hop_values) == list(range(1, 6))
    assert sum(len(v) for v in res.per_hop) == 34 * 33 // 2
    assert -1 <= res.spearman_rho <= 1


def test_label_separation(karate):
    sc, cl, _ = karate
    labels = np.array(["A"] * 17 + ["B"] * 16 + ["solo"], dtype=object)
    res = markers.label_separation(cl, labels, n_boot=20, exclude=("solo",))
    assert res.n == 33 and res.n_groups == 2
    assert res.silhouette_lo <= res.silhouette <= res.silhouette_hi


def test_partner_precision(karate):
    sc, cl, _ = karate
    res = markers.partner_precision_at_k(sc, cl, k=3)
    assert len(res.precision) == len(res.eligible) > 0
    assert np.all((res.precision >= 0) & (res.precision <= 1))


def test_partner_precision_edge_cases(karate):
    sc, cl, _ = karate
    with pytest.raises(ValueError, match="k must satisfy"):
        markers.partner_precision_at_k(sc, cl, k=34)
    res = markers.partner_precision_at_k(sc, cl, k=3, groups=[[5]])
    assert len(res.eligible) == 0 and np.isnan(res.mean)


def test_group_compactness(karate):
    _, cl, _ = karate
    out = markers.group_compactness(cl, [[0, 1, 2], [5]])
    assert out[0] > 0 and np.isnan(out[1])


def test_triangle_clustering(karate):
    sc, _, hho = karate
    res = markers.triangle_clustering(sc, hho)
    n = len(sc.triangles_in_skeleton())
    assert len(res.clustering_index) == n and res.filled.sum() == len(sc.filled_triangles())
    assert set(res.stats) == {"all", "filled", "empty"}
    assert res.stats["all"]["n"] == n
    assert "all.rho_area_partial" in res.to_dict()


def test_markers_tolerate_missing_triangle_classes():
    G = nx.cycle_graph(8)
    G.add_edges_from([(0, 2)])
    sc = SimplicialComplex.from_graph(G, [(0, 1, 2)])
    hho = HHOEmbedder(**PARAMS).fit(sc, seed=0)
    cl = ClassicalEmbedder(**PARAMS).fit(sc, seed=0)
    empty = markers.surface_vs_empty_areas(sc, hho)
    assert len(empty.area_empty) == 0 and np.isnan(empty.mannwhitney_p) and np.isnan(empty.fold)
    tc = markers.triangle_clustering(sc, hho)
    assert tc.stats["empty"]["n"] == 0 and np.isnan(tc.stats["empty"]["rho_area"])
    ar = markers.area_ratio(sc, cl, hho)
    assert ar.to_dict()["n"] <= 1


def test_poincare_only_markers_reject_euclidean(karate):
    sc, cl, hho = karate
    eu = EmbeddingResult(coords=cl.coords, space="euclidean", layer2=None, name="eu")
    with pytest.raises(ValueError, match="Poincaré"):
        markers.area_ratio(sc, eu, hho)
