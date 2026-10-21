import warnings

import numpy as np
import pytest

from hohypemb import geometry as geo


def test_hyp_dist_scalar_matches_matrix():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.6, 0.6, size=(6, 2))
    D = geo.hyp_dist_matrix(pts)
    for i in range(6):
        for j in range(6):
            assert geo.hyp_dist(pts[i], pts[j]) == pytest.approx(D[i, j], abs=1e-10)


def test_hyp_dist_pairs_rowwise():
    P = np.array([[0.0, 0.0], [0.1, 0.2]])
    Q = np.array([[0.5, 0.0], [0.1, 0.2]])
    np.testing.assert_allclose(geo.hyp_dist_pairs(P, Q), [geo.hyp_dist(P[0], Q[0]), 0.0], atol=1e-12)


def test_triangle_areas_match_scalar_version():
    rng = np.random.default_rng(1)
    emb = rng.uniform(-0.7, 0.7, size=(10, 2))
    tris = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [1, 5, 9]])
    expected = [geo.hyp_triangle_area(*emb[t]) for t in tris]
    np.testing.assert_allclose(geo.triangle_areas(emb, tris), expected, atol=1e-10)


def test_small_triangle_area_uses_conformal_factor():
    # near the origin the area element is (2 / (1 - r^2))^2 = 4 times the Euclidean one
    eps = 1e-3
    emb = np.array([[0.0, 0.0], [eps, 0.0], [0.0, eps]])
    assert geo.triangle_areas(emb, [[0, 1, 2]])[0] == pytest.approx(4 * eps ** 2 / 2, rel=1e-2)


def test_triangle_areas_degenerate_and_empty():
    emb = np.array([[0.1, 0.1], [0.1, 0.1], [0.3, 0.0]])
    assert np.isnan(geo.triangle_areas(emb, [[0, 1, 2]])[0])
    assert geo.triangle_areas(emb, np.empty((0, 3), dtype=int)).shape == (0,)


def test_angular_gaps():
    emb = np.array([[0.5, 0.0], [0.0, 0.5], [-0.5, 0.0]])   # angles 0, pi/2, pi
    assert geo.angular_gaps(emb, [[0, 1, 2]])[0] == pytest.approx((np.pi / 2 + np.pi + np.pi / 2) / 3)


def test_hyperbolic_radius():
    emb = np.array([[0.0, 0.0], [np.tanh(0.5), 0.0]])
    np.testing.assert_allclose(geo.hyperbolic_radius(emb), [0.0, 1.0], atol=1e-12)


def test_apply_radial_rule_rejects_other_rules():
    with pytest.raises(ValueError, match="not supported"):
        geo.apply_radial_rule(np.ones((2, 2)), np.ones(2), 5.0, rule="logdegree")


def test_dist_matrices_raise_no_warnings_and_are_finite():
    # Regression for the macOS Accelerate-BLAS false-positive FP-flag bug in the
    # Gram-matrix matmul inside hyp_dist_matrix/euclidean_dist_matrix.
    rng = np.random.default_rng(2)
    pts = rng.uniform(-0.8, 0.8, size=(20, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        D_hyp = geo.hyp_dist_matrix(pts)
        D_euc = geo.euclidean_dist_matrix(pts)
    assert np.all(np.isfinite(D_hyp))
    assert np.all(np.isfinite(D_euc))
