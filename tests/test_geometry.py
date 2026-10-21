import unittest

import numpy as np

from hohypemb.geometry import hyp_dist
from hohypemb import geometry as geo


class TestClipToDisk(unittest.TestCase):
    def test_boundary_point_is_pulled_inside(self):
        emb = np.array([[1.0, 0.0], [0.0, 0.5]])
        out = geo.clip_to_disk(emb)
        self.assertLess(np.linalg.norm(out[0]), 1.0)
        np.testing.assert_allclose(out[1], [0.0, 0.5])

    def test_interior_points_are_untouched(self):
        emb = np.array([[0.1, 0.2], [-0.3, 0.4]])
        np.testing.assert_array_equal(geo.clip_to_disk(emb), emb)


class TestHypDistMatrix(unittest.TestCase):
    def test_matches_reference(self):
        rng = np.random.default_rng(0)
        emb = rng.uniform(-0.6, 0.6, size=(25, 2))
        D = geo.hyp_dist_matrix(emb)
        for i in range(25):
            for j in range(25):
                self.assertAlmostEqual(
                    D[i, j], hyp_dist(emb[i], emb[j]), delta=1e-10,
                    msg=f"mismatch at ({i}, {j})",
                )

    def test_degree_zero_node_does_not_produce_inf(self):
        emb = np.array([[1.0, 0.0], [0.2, 0.1], [0.0, 0.0]])
        D = geo.hyp_dist_matrix(emb)
        self.assertTrue(np.all(np.isfinite(D)))

    def test_diagonal_is_zero_and_matrix_symmetric(self):
        rng = np.random.default_rng(1)
        D = geo.hyp_dist_matrix(rng.uniform(-0.5, 0.5, size=(10, 2)))
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
        np.testing.assert_allclose(D, D.T, atol=1e-12)


class TestAngularDistMatrix(unittest.TestCase):
    def test_values_in_zero_pi(self):
        rng = np.random.default_rng(2)
        A = geo.angular_dist_matrix(rng.uniform(-0.5, 0.5, size=(30, 2)))
        self.assertTrue(np.all(A >= -1e-12) and np.all(A <= np.pi + 1e-12))

    def test_opposite_points_are_pi_apart(self):
        emb = np.array([[0.5, 0.0], [-0.5, 0.0]])
        self.assertAlmostEqual(geo.angular_dist_matrix(emb)[0, 1], np.pi, places=10)

    def test_ignores_radius(self):
        emb = np.array([[0.1, 0.0], [0.9, 0.0]])
        self.assertAlmostEqual(geo.angular_dist_matrix(emb)[0, 1], 0.0, places=12)


class TestTriangleArea(unittest.TestCase):
    def test_area_equals_pi_minus_angle_sum(self):
        p1, p2, p3 = np.array([0.1, 0.0]), np.array([0.0, 0.3]), np.array([-0.25, -0.1])
        area = geo.hyp_triangle_area(p1, p2, p3)
        self.assertGreater(area, 0.0)
        self.assertLess(area, np.pi)

    def test_degenerate_triangle_returns_nan_not_exception(self):
        p = np.array([0.2, 0.2])
        self.assertTrue(np.isnan(geo.hyp_triangle_area(p, p, np.array([0.3, 0.1]))))

    def test_smaller_triangle_has_smaller_area(self):
        small = geo.hyp_triangle_area(
            np.array([0.01, 0.0]), np.array([0.0, 0.01]), np.array([-0.01, -0.01])
        )
        large = geo.hyp_triangle_area(
            np.array([0.8, 0.0]), np.array([0.0, 0.8]), np.array([-0.8, -0.8])
        )
        self.assertLess(small, large)

    def test_distinct_collinear_points_give_finite_near_zero_area(self):
        """Three distinct points on one geodesic have a true area of 0, not nan.

        Documented explicitly because the contract is easy to misread: only
        COINCIDENT vertices are undefined and yield nan.
        """
        geodesic = geo.poincare_geodesic(
            np.array([0.6, 0.1]), np.array([-0.3, 0.5]), num_points=5
        )
        area = geo.hyp_triangle_area(geodesic[0], geodesic[2], geodesic[4])
        self.assertTrue(np.isfinite(area))
        self.assertLess(abs(area), 1e-6)

    def test_equilateral_triangle_matches_closed_form(self):
        """Known answer: equilateral triangle centred at the origin, vertices at radius r.

        The vertices sit at angles 0, 2pi/3, 4pi/3 (all shifted by 0.1 rad below: the area is
        rotation invariant), so every pair is a chord of length r*sqrt(3):
        |p - q|^2 = 3 r^2. The side ``a`` follows from the Poincare-disk closed form
        cosh a = 1 + 2 |p - q|^2 / ((1 - |p|^2)(1 - |q|^2)) = 1 + 6 r^2 / (1 - r^2)^2.
        The hyperbolic law of cosines with b = c = a gives cosh a = cosh^2 a - sinh^2 a cos(alpha), so
        cos(alpha) = (cosh^2 a - cosh a) / sinh^2 a = cosh a (cosh a - 1) / ((cosh a - 1)(cosh a + 1))
                   = cosh a / (1 + cosh a),
        and Gauss-Bonnet gives the area pi - 3 alpha (r = 0.3: 0.447; r = 0.8: 2.385).

        Tolerance 1e-12: the inputs are exact to ~1e-16 relative, and the only amplification is
        arccos near cos(alpha) = 0.97 (slope 1/sin(alpha) ~ 4) and cosh/sinh of sides ~ 4, so the
        expected error is ~1e-14; 1e-12 leaves two orders of margin while still catching any
        formula slip (which would be O(1e-2) or larger).
        """
        for r in (0.3, 0.8):
            angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3]) + 0.1   # arbitrary rotation
            pts = r * np.column_stack([np.cos(angles), np.sin(angles)])
            cosh_a = 1 + 6 * r ** 2 / (1 - r ** 2) ** 2
            expected = np.pi - 3 * np.arccos(cosh_a / (1 + cosh_a))
            self.assertAlmostEqual(geo.hyp_triangle_area(*pts), expected, delta=1e-12)
            np.testing.assert_allclose(geo.triangle_areas(pts, [[0, 1, 2]]), [expected], rtol=0, atol=1e-12)


class TestGeodesic(unittest.TestCase):
    def test_endpoints_are_preserved(self):
        p1, p2 = np.array([0.3, 0.1]), np.array([-0.2, 0.4])
        g = geo.poincare_geodesic(p1, p2, num_points=15)
        self.assertEqual(g.shape, (15, 2))
        np.testing.assert_allclose(g[0], p1, atol=1e-9)
        np.testing.assert_allclose(g[-1], p2, atol=1e-9)

    def test_collinear_with_origin_is_a_straight_line(self):
        g = geo.poincare_geodesic(np.array([0.4, 0.0]), np.array([-0.4, 0.0]), 11)
        np.testing.assert_allclose(g[:, 1], 0.0, atol=1e-9)

    def test_stays_inside_the_disk(self):
        g = geo.poincare_geodesic(np.array([0.85, 0.1]), np.array([-0.1, 0.85]), 40)
        self.assertTrue(np.all(np.linalg.norm(g, axis=1) < 1.0 + 1e-9))


class TestKleinBarycenter(unittest.TestCase):
    def test_barycenter_of_identical_points_is_that_point(self):
        pts = np.tile(np.array([0.2, -0.3]), (5, 1))
        np.testing.assert_allclose(geo.klein_barycenter(pts), [0.2, -0.3], atol=1e-9)

    def test_symmetric_pair_has_barycenter_at_origin(self):
        pts = np.array([[0.4, 0.0], [-0.4, 0.0]])
        np.testing.assert_allclose(geo.klein_barycenter(pts), [0.0, 0.0], atol=1e-9)

    def test_barycenter_lies_inside_the_disk(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(-0.7, 0.7, size=(8, 2))
        self.assertLess(np.linalg.norm(geo.klein_barycenter(pts)), 1.0)


class TestRadialRule(unittest.TestCase):
    def test_reproduces_library_formula(self):
        coords = np.array([[3.0, 4.0], [-1.0, 0.0]])
        degrees = np.array([2.0, 8.0])
        beta = 5.0
        out = geo.apply_radial_rule(coords, degrees, beta, rule="degree")
        expected_r = 1 - np.tanh(degrees / beta)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), expected_r, atol=1e-12)

    def test_high_degree_maps_closer_to_origin(self):
        coords = np.array([[1.0, 0.0], [1.0, 0.0]])
        out = geo.apply_radial_rule(coords, np.array([1.0, 50.0]), 5.0)
        self.assertLess(np.linalg.norm(out[1]), np.linalg.norm(out[0]))

    def test_preserves_angles(self):
        coords = np.array([[1.0, 1.0], [-2.0, 0.5]])
        out = geo.apply_radial_rule(coords, np.array([3.0, 4.0]), 7.0)
        for before, after in zip(coords, out):
            self.assertAlmostEqual(
                np.arctan2(before[1], before[0]), np.arctan2(after[1], after[0]), places=12
            )


class TestGroupCompactness(unittest.TestCase):
    def test_mean_aggregation(self):
        D = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 5.0], [3.0, 5.0, 0.0]])
        self.assertAlmostEqual(geo.group_compactness(D, [0, 1, 2], "mean"), 3.0)

    def test_max_aggregation(self):
        D = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 5.0], [3.0, 5.0, 0.0]])
        self.assertAlmostEqual(geo.group_compactness(D, [0, 1, 2], "max"), 5.0)


if __name__ == "__main__":
    unittest.main()
