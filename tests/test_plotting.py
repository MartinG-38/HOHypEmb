import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hohypemb import plotting as plot
from hohypemb.complex import SimplicialComplex
from hohypemb.embedders.base import EmbeddingResult


class TestStyleAndPalette(unittest.TestCase):
    def test_set_paper_style_sets_savefig_defaults(self):
        plot.set_paper_style()
        self.assertEqual(matplotlib.rcParams["savefig.bbox"], "tight")
        self.assertGreaterEqual(matplotlib.rcParams["savefig.dpi"], 300)

    def test_palette_covers_every_registry_key(self):
        for key in ["classical", "hho", "weighted_clique", "star_expansion",
                    "zhou_hypergraph", "euclidean_spectral", "euclidean_isomap",
                    "euclidean_star"]:
            self.assertIn(key, plot.PALETTE)

    def test_palette_entries_are_hex_colours(self):
        for value in plot.PALETTE.values():
            self.assertRegex(value, r"^#[0-9a-fA-F]{6}$")


class TestPoincarePanel(unittest.TestCase):
    def test_draws_collections_not_per_edge_lines(self):
        sc = SimplicialComplex(
            simplices=[(0, 1, 2), (1, 2, 3), (3, 4, 5)], n_nodes=6, name="t"
        )
        rng = np.random.default_rng(0)
        result = EmbeddingResult(
            coords=rng.uniform(-0.6, 0.6, (6, 2)), space="poincare",
            layer2=None, name="t", meta={},
        )
        fig, ax = plt.subplots()
        plot.plot_poincare(ax, sc, result, title="panel")
        self.assertGreaterEqual(len(ax.collections), 2)  # edges + triangles + nodes
        self.assertLessEqual(len(ax.lines), 2)           # origin marker only
        self.assertEqual(ax.get_title(), "panel")
        plt.close(fig)

    def test_axes_are_square_and_clipped_to_the_disk(self):
        sc = SimplicialComplex(simplices=[(0, 1, 2)], n_nodes=3, name="t")
        result = EmbeddingResult(
            coords=np.array([[0.3, 0.0], [0.0, 0.3], [-0.3, -0.3]]),
            space="poincare", layer2=None, name="t", meta={},
        )
        fig, ax = plt.subplots()
        plot.plot_poincare(ax, sc, result)
        self.assertAlmostEqual(ax.get_xlim()[1], 1.05, places=2)
        self.assertEqual(ax.get_aspect(), 1.0)
        plt.close(fig)


def _tiny_complex():
    sc = SimplicialComplex(simplices=[(0, 1, 2), (1, 2, 3), (3, 4, 5)], n_nodes=6, name="t")
    rng = np.random.default_rng(0)
    res = EmbeddingResult(coords=rng.uniform(-0.6, 0.6, (6, 2)), space="poincare",
                          layer2=None, name="t", meta={})
    return sc, res


def _node_sizes(ax):
    """Marker sizes of the node scatter (the last PathCollection of a Poincare panel)."""
    from matplotlib.collections import PathCollection
    return np.asarray([c for c in ax.collections if isinstance(c, PathCollection)][-1].get_sizes())


class TestNodeScale(unittest.TestCase):
    def test_default_node_size_is_unchanged(self):
        sc, res = _tiny_complex()
        fig, ax = plt.subplots()
        plot.plot_poincare(ax, sc, res)
        degrees = np.array([sc.one_skeleton.degree(i) for i in range(sc.n_nodes)], dtype=float)
        np.testing.assert_allclose(_node_sizes(ax), np.sqrt(degrees + 1) * 9)
        plt.close(fig)

    def test_node_scale_multiplies_the_node_sizes(self):
        sc, res = _tiny_complex()
        fig, (a1, a2) = plt.subplots(1, 2)
        plot.plot_poincare(a1, sc, res)
        plot.plot_poincare(a2, sc, res, node_scale=2.0)
        np.testing.assert_allclose(_node_sizes(a2), 2 * _node_sizes(a1))
        plt.close(fig)

    def test_comparison_honors_figsize_and_node_scale(self):
        sc, res = _tiny_complex()
        fig0 = plot.plot_comparison(sc, res, res)
        self.assertEqual(tuple(fig0.get_size_inches()), (10, 4.8))
        fig = plot.plot_comparison(sc, res, res, node_scale=3.0, figsize=(14, 7))
        self.assertEqual(tuple(fig.get_size_inches()), (14, 7))
        for a0, a1 in zip(fig0.axes, fig.axes):
            np.testing.assert_allclose(_node_sizes(a1), 3 * _node_sizes(a0))
        plt.close(fig0)
        plt.close(fig)

    def test_highlighted_groups_scales_background_and_group_markers(self):
        from matplotlib.collections import PathCollection
        sc, res = _tiny_complex()

        def sizes(scale):
            fig, ax = plt.subplots()
            plot.plot_highlighted_groups(ax, sc, res, [[0, 1, 2]], ["red"], node_scale=scale)
            pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
            out = (np.asarray(pcs[0].get_sizes()), np.asarray(pcs[-1].get_sizes()))
            plt.close(fig)
            return out

        (bg1, grp1), (bg2, grp2) = sizes(1.0), sizes(2.0)
        np.testing.assert_allclose(grp1, [40.0])
        np.testing.assert_allclose(bg2, 2 * bg1)
        np.testing.assert_allclose(grp2, 2 * grp1)


class TestOtherPanels(unittest.TestCase):
    def test_response_surface_adds_an_image(self):
        fig, ax = plt.subplots()
        plot.plot_response_surface(
            ax, np.array([1.0, 2.0]), np.array([0.5, 1.0]),
            np.array([[0.1, 0.2], [0.3, 0.4]]), cbar_label="accuracy",
        )
        self.assertGreaterEqual(len(ax.images), 1)
        plt.close(fig)

    def test_response_surface_draws_the_maximum_at_its_true_grid_position(self):
        """C2 pinning test. The mu grid used by the real response-surface
        experiment is deliberately NON-UNIFORM (mirrors [0.5,1,2,3,5,7,10,15,20]),
        which under ``imshow(extent=...)`` spreads columns uniformly and draws
        every cell at the WRONG position relative to its true value -- the
        true optimum at mu=3 was drawn at mu=8.08. This
        pins the fix: the maximum cell's true value must appear as the tick
        label at its own INDEX position, not be linearly interpolated."""
        mu = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20])   # non-uniform, as in production
        gamma = np.array([0.0, 0.5, 1.0, 2.0])
        grid = np.zeros((len(gamma), len(mu)))
        true_gamma_idx, true_mu_idx = 2, 3     # gamma=1.0, mu=3 -- the real optimum's slot
        grid[true_gamma_idx, true_mu_idx] = 1.0

        fig, ax = plt.subplots()
        plot.plot_response_surface(ax, mu, gamma, grid, cbar_label="accuracy")

        argmax_gamma_idx, argmax_mu_idx = np.unravel_index(np.argmax(grid), grid.shape)
        self.assertEqual((argmax_gamma_idx, argmax_mu_idx), (true_gamma_idx, true_mu_idx))

        # The tick label AT the maximum's index position must equal its true
        # data value -- this is what an extent-based, uniformly-spread axis
        # would get wrong for a non-uniform grid like this one.
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        yticklabels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertEqual(xticklabels[argmax_mu_idx], f"{mu[argmax_mu_idx]:g}")
        self.assertEqual(yticklabels[argmax_gamma_idx], f"{gamma[argmax_gamma_idx]:g}")

        # And the image itself must be drawn on the plain index grid (no
        # extent stretching a non-uniform axis across a uniform pixel range).
        im = ax.images[0]
        x0, x1, y0, y1 = im.get_extent()
        self.assertAlmostEqual(x0, -0.5, places=6)
        self.assertAlmostEqual(x1, len(mu) - 0.5, places=6)
        self.assertAlmostEqual(y0, -0.5, places=6)
        self.assertAlmostEqual(y1, len(gamma) - 0.5, places=6)
        plt.close(fig)

    def test_slice_with_ci_adds_line_and_band(self):
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 5)
        plot.plot_slice_with_ci(ax, x, x * 0.1, x * 0.08, x * 0.12, "hho", "#1f77b4")
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(len(ax.collections), 1)
        plt.close(fig)

    def test_ablation_grid_draws_one_bar_per_row(self):
        table = pd.DataFrame({
            "method": ["hho", "classical", "star_expansion"],
            "accuracy": [0.42, 0.38, 0.39],
        })
        fig, ax = plt.subplots()
        plot.plot_ablation_grid(ax, table, "accuracy")
        self.assertEqual(len(ax.patches), 3)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()


class TestAreaRatioLabel(unittest.TestCase):
    def test_small_median_and_share_are_not_rounded_away(self):
        from hohypemb.markers import AreaRatioResult
        ratio = np.array([7.9e-4, 5e-4, 2e-3, 1.5])
        res = AreaRatioResult(triangles=np.zeros((4, 3), dtype=int), area_classical=np.ones(4),
                              area_hho=ratio, ratio=ratio, n_excluded=0, median_ratio=7.9e-4,
                              frac_below_1=0.996, wilcoxon_p=0.01)
        fig, ax = plt.subplots()
        plot.plot_area_ratio(ax, res)
        label = "\n".join(t.get_text() for t in ax.texts)
        plt.close(fig)
        self.assertIn("median = 0.00079", label)
        self.assertIn("99.6% smaller", label)

    def test_label_sits_above_the_bars(self):
        from hohypemb.markers import AreaRatioResult
        ratio = np.array([7.9e-4, 5e-4, 2e-3, 1.5])
        res = AreaRatioResult(triangles=np.zeros((4, 3), dtype=int), area_classical=np.ones(4),
                              area_hho=ratio, ratio=ratio, n_excluded=0, median_ratio=7.9e-4,
                              frac_below_1=0.75, wilcoxon_p=0.01)
        fig, ax = plt.subplots()
        plot.plot_area_ratio(ax, res)
        tallest = max(p.get_height() for p in ax.patches)
        top = ax.get_ylim()[1]
        plt.close(fig)
        self.assertGreaterEqual(top, 1.4 * tallest)
