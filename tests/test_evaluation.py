import unittest

import numpy as np

from hohypemb import evaluation as tasks
from hohypemb import embedders as emb
from hohypemb import presets
from hohypemb.datasets.cora import load_cora_coauthorship
from hohypemb.embedders.base import EmbeddingResult
from tests._data import CORA_PATH, require


def _two_blobs(n=60, sep=3.0, seed=0):
    """Two well-separated clusters and their labels, as a distance matrix."""
    rng = np.random.default_rng(seed)
    pts = np.vstack([rng.normal(0, 0.3, (n // 2, 2)),
                     rng.normal(sep, 0.3, (n // 2, 2))])
    labels = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return D, labels


class TestSplits(unittest.TestCase):
    def test_splits_are_deterministic(self):
        _, labels = _two_blobs()
        a = tasks.make_splits(labels, n_splits=5, n_repeats=2, seed=0)
        b = tasks.make_splits(labels, n_splits=5, n_repeats=2, seed=0)
        self.assertEqual(len(a), 10)
        for (tr1, te1), (tr2, te2) in zip(a, b):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_train_and_test_are_disjoint_and_cover_everything(self):
        _, labels = _two_blobs()
        for train, test in tasks.make_splits(labels, n_splits=5, n_repeats=1, seed=0):
            self.assertEqual(len(set(train) & set(test)), 0)
            self.assertEqual(len(set(train) | set(test)), len(labels))


class TestNodeClassification(unittest.TestCase):
    def test_separable_data_scores_near_one(self):
        D, labels = _two_blobs()
        splits = tasks.make_splits(labels, n_splits=5, n_repeats=2, seed=0)
        out = tasks.node_classification(D, labels, splits, k=5)
        self.assertEqual(len(out["accuracy"]), 10)
        self.assertGreater(out["accuracy"].mean(), 0.95)
        self.assertGreater(out["macro_f1"].mean(), 0.95)

    def test_random_distances_score_near_chance(self):
        rng = np.random.default_rng(0)
        n = 60
        D = rng.uniform(0, 1, (n, n))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0.0)
        labels = np.array(["a"] * 30 + ["b"] * 30)
        splits = tasks.make_splits(labels, n_splits=5, n_repeats=2, seed=0)
        self.assertLess(tasks.node_classification(D, labels, splits, k=5)["accuracy"].mean(), 0.75)

    def test_same_splits_give_identical_results(self):
        D, labels = _two_blobs()
        splits = tasks.make_splits(labels, n_splits=5, n_repeats=1, seed=0)
        np.testing.assert_array_equal(
            tasks.node_classification(D, labels, splits, k=5)["accuracy"],
            tasks.node_classification(D, labels, splits, k=5)["accuracy"],
        )


class TestSilhouette(unittest.TestCase):
    def test_separated_clusters_give_high_silhouette_with_ci(self):
        D, labels = _two_blobs()
        out = tasks.silhouette_with_ci(D, labels, n_boot=50, seed=0)
        self.assertGreater(out["score"], 0.5)
        self.assertLessEqual(out["lo"], out["score"])
        self.assertGreaterEqual(out["hi"], out["score"])


class TestRankNormalisedSilhouette(unittest.TestCase):
    """C3: raw silhouette is scale-sensitive; the rank-normalised version must not be."""

    def test_separated_clusters_still_give_high_rank_silhouette(self):
        D, labels = _two_blobs()
        out = tasks.rank_normalised_silhouette(D, labels, n_boot=50, seed=0)
        self.assertGreater(out["score"], 0.5)
        self.assertLessEqual(out["lo"], out["score"])
        self.assertGreaterEqual(out["hi"], out["score"])

    def test_invariant_to_a_monotone_reshaping_that_moves_raw_silhouette(self):
        """The whole point of the transform: rank-normalised
        silhouette is invariant under ANY monotone reshaping of the distances,
        not merely a uniform rescale (which raw silhouette already tolerates,
        since it is scale-invariant on its own). A cube is monotone on
        non-negative distances but changes the distribution's SHAPE -- exactly
        the kind of mismatch that made euclidean_spectral's collapsed-radius
        map look far more incoherent than it structurally is. Raw silhouette
        moves under this reshaping; rank-normalised silhouette does not."""
        D, labels = _two_blobs()
        D_warped = D ** 3  # monotone (D >= 0) but non-linear: reshapes, doesn't just rescale
        raw_full = tasks.silhouette_with_ci(D, labels, n_boot=1)["score"]
        raw_warped = tasks.silhouette_with_ci(D_warped, labels, n_boot=1)["score"]
        rank_full = tasks.rank_normalised_silhouette(D, labels, n_boot=1)["score"]
        rank_warped = tasks.rank_normalised_silhouette(D_warped, labels, n_boot=1)["score"]
        self.assertGreater(abs(raw_full - raw_warped), 0.01)     # raw score DOES move
        self.assertAlmostEqual(rank_full, rank_warped, places=9)  # rank score does not

    def test_rank_normalised_distance_is_symmetric_zero_diagonal_and_bounded(self):
        rng = np.random.default_rng(0)
        D = rng.uniform(1, 50, (10, 10))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0.0)
        R = tasks.rank_normalised_distance(D)
        np.testing.assert_allclose(R, R.T)
        np.testing.assert_allclose(np.diag(R), 0.0)
        self.assertGreaterEqual(R.min(), 0.0)
        self.assertLessEqual(R.max(), 1.0)


class TestCommunityRecovery(unittest.TestCase):
    def test_recovers_planted_clusters(self):
        rng = np.random.default_rng(0)
        coords = np.vstack([rng.normal(-0.5, 0.02, (30, 2)), rng.normal(0.5, 0.02, (30, 2))])
        labels = np.array([0] * 30 + [1] * 30)
        r = EmbeddingResult(coords=coords, space="euclidean", layer2=None, name="t", meta={})
        out = tasks.community_recovery(r, labels, K=2, seed=0)
        self.assertGreater(out["nmi"], 0.9)
        self.assertGreater(out["ari"], 0.9)

    def test_returns_near_zero_for_random_labels(self):
        rng = np.random.default_rng(1)
        coords = rng.normal(0, 0.3, (60, 2))
        labels = rng.integers(0, 2, 60)
        r = EmbeddingResult(coords=coords, space="euclidean", layer2=None, name="t", meta={})
        self.assertLess(tasks.community_recovery(r, labels, K=2, seed=0)["nmi"], 0.3)


class TestStatistics(unittest.TestCase):
    def test_holm_correction_is_monotone_and_bounded(self):
        p = np.array([0.001, 0.02, 0.04, 0.5])
        adj = tasks.holm_correction(p)
        self.assertTrue(np.all(adj >= p - 1e-12))
        self.assertTrue(np.all(adj <= 1.0))
        self.assertTrue(np.all(np.diff(adj[np.argsort(p)]) >= -1e-12))

    def test_holm_matches_hand_computation(self):
        adj = tasks.holm_correction(np.array([0.01, 0.04]))
        np.testing.assert_allclose(adj, [0.02, 0.04], atol=1e-12)

    def test_bootstrap_ci_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(5.0, 1.0, 200)
        lo, hi = tasks.bootstrap_ci(vals, n_boot=500, seed=0)
        self.assertLess(lo, vals.mean())
        self.assertGreater(hi, vals.mean())

    def test_paired_table_flags_a_real_difference(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0.40, 0.01, 40)
        scores = {"hho": base + 0.05, "classical": base}
        table = tasks.paired_wilcoxon_table(scores, reference="hho")
        row = table.set_index("method").loc["classical"]
        self.assertLess(row["p_holm"], 0.05)
        self.assertGreater(row["delta_mean"], 0.0)

    def test_paired_table_excludes_the_reference_row(self):
        scores = {"hho": np.arange(20.0), "classical": np.arange(20.0)}
        self.assertNotIn("hho", tasks.paired_wilcoxon_table(scores, "hho")["method"].tolist())

    def test_paired_table_refuses_mismatched_fold_counts(self):
        """The guard that keeps comparisons paired. Without it, methods scored on
        different splits would be compared as if they shared them, invalidating
        every p-value in the analysis."""
        scores = {"hho": np.arange(20.0), "classical": np.arange(10.0)}
        with self.assertRaises(ValueError) as ctx:
            tasks.paired_wilcoxon_table(scores, reference="hho")
        self.assertIn("folds", str(ctx.exception))


class TestDegreeMatchedNegatives(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.degrees = np.concatenate([rng.uniform(2, 10, 200), rng.uniform(30, 60, 200)])
        self.candidates = [tuple(rng.choice(400, 3, replace=False)) for _ in range(4000)]
        self.positives = [
            tuple(rng.choice(200, 3, replace=False)) for _ in range(40)  # low-degree only
        ]

    def test_returns_requested_count(self):
        negs = tasks.degree_matched_negatives(
            self.candidates, self.positives, self.degrees, n_per_pos=5, seed=0
        )
        self.assertGreater(len(negs), 0)
        self.assertLessEqual(len(negs), 5 * len(self.positives))

    def test_negatives_match_the_positive_degree_profile(self):
        negs = tasks.degree_matched_negatives(
            self.candidates, self.positives, self.degrees, n_per_pos=5, seed=0
        )
        quality = tasks.degree_match_quality(self.positives, negs, self.degrees)
        self.assertLess(abs(quality["mean_pos"] - quality["mean_neg"]), 5.0)

    def test_uniform_sampling_would_be_badly_mismatched(self):
        """Guards the reason the matcher exists."""
        rng = np.random.default_rng(1)
        uniform = [self.candidates[i] for i in rng.choice(len(self.candidates), 200)]
        quality = tasks.degree_match_quality(self.positives, uniform, self.degrees)
        self.assertGreater(abs(quality["mean_pos"] - quality["mean_neg"]), 5.0)

    def test_never_returns_a_positive_as_a_negative(self):
        negs = tasks.degree_matched_negatives(
            self.candidates, self.positives, self.degrees, n_per_pos=5, seed=0
        )
        pos_set = {frozenset(p) for p in self.positives}
        for n in negs:
            self.assertNotIn(frozenset(n), pos_set)

    def test_deterministic_under_seed(self):
        a = tasks.degree_matched_negatives(
            self.candidates, self.positives, self.degrees, n_per_pos=3, seed=7
        )
        b = tasks.degree_matched_negatives(
            self.candidates, self.positives, self.degrees, n_per_pos=3, seed=7
        )
        self.assertEqual(a, b)


class TestPredictionTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = load_cora_coauthorship(require(CORA_PATH), max_dim=5).giant_component()

    def test_pairwise_link_prediction_beats_chance(self):
        out = tasks.pairwise_link_prediction(
            self.sc, emb.ClassicalEmbedder(**presets.EMBEDDING_PARAMS["cora"]),
            frac=0.2, seed=0,
        )
        self.assertGreater(out["auc"], 0.6)   # spike measured 0.747
        self.assertIn("ap", out)

    def test_simplex_prediction_returns_finite_metrics(self):
        out = tasks.simplex_prediction(
            self.sc, emb.ClassicalEmbedder(**presets.EMBEDDING_PARAMS["cora"]),
            frac=0.2, n_neg_per_pos=5, seed=0,
        )
        for key in ("auc", "ap", "n_pos", "n_neg", "degree_ks"):
            self.assertIn(key, out)
        self.assertTrue(np.isfinite(out["auc"]))

    def test_simplex_prediction_only_tests_maximal_triangles(self):
        """A triangle that is a sub-face of a larger retained facet is not truly
        held out -- its group information survives via the parent, so it must
        never become a candidate positive. Calls the real ``simplex_prediction``
        (not a local reimplementation of its filter) on a fixture built so the
        maximal-only candidate count and the naive (no-filter) candidate count
        round to DIFFERENT ``n_pos`` values under ``frac=0.2`` -- 12 vs 16,
        giving max(1, int(0.2*12))=2 vs max(1, int(0.2*16))=3 -- so removing
        the maximality filter changes the observed ``n_pos``, not just an
        internal detail this test would otherwise have to recompute itself."""
        from hohypemb.complex import SimplicialComplex
        facets = []
        # 4 nested groups: a non-maximal triangle plus its size-4 parent.
        # Every sub-triangle of a K4 facet is EXCLUDED as an "open" negative
        # too (C1), so this fixture never needs negatives to exist -- it only
        # exercises how many candidates become positives.
        for k in range(4):
            base = 4 * k
            facets.append((base, base + 1, base + 2))
            facets.append((base, base + 1, base + 2, base + 3))
        # 12 disjoint, genuinely maximal triangles.
        for i in range(12):
            base = 16 + 3 * i
            facets.append((base, base + 1, base + 2))
        sc = SimplicialComplex(simplices=facets, n_nodes=16 + 3 * 12, name="nested_npos")

        out = tasks.simplex_prediction(
            sc, emb.ClassicalEmbedder(**presets.EMBEDDING_PARAMS["synthetic_control"]),
            frac=0.2, n_neg_per_pos=1, seed=0,
        )
        self.assertEqual(out["n_pos"], 2.0)

    def test_simplex_prediction_retains_non_maximal_triangles_in_training(self):
        """Non-maximal triangles are untestable (their parent would leak them back)
        but they are real observed structure and must stay in the TRAINING
        complex. Uses a recording embedder to observe what facets actually
        reach ``.fit()`` -- calling the real ``simplex_prediction`` end to end,
        not reimplementing its retention logic locally."""
        from hohypemb.complex import SimplicialComplex

        class _RecordingEmbedder:
            """Delegates to a real embedder but records the facets of whatever
            complex it is asked to fit, so a test can observe the TRAINING
            complex ``simplex_prediction`` builds without reimplementing its
            filtering logic."""

            def __init__(self, inner):
                self._inner = inner
                self.seen_facets = None

            def fit(self, sc, seed=0):
                self.seen_facets = list(sc.simplices)
                return self._inner.fit(sc, seed=seed)

        facets = [(0, 1, 2), (0, 1, 2, 3)]  # non-maximal triangle + its parent
        # 10 disjoint maximal triangles, to clear the "< 10 candidates" guard.
        for i in range(10):
            base = 4 + 3 * i
            facets.append((base, base + 1, base + 2))
        # 15 genuinely OPEN triangles: three separate edges closing a triangle
        # in the 1-skeleton without any facet (or sub-face of one) covering
        # all three nodes at once -- ample degree-matched negative supply.
        open_base = 4 + 3 * 10
        for j in range(15):
            a, b, c = open_base + 3 * j, open_base + 3 * j + 1, open_base + 3 * j + 2
            facets.extend([(a, b), (b, c), (a, c)])
        n_nodes = open_base + 3 * 15
        # Adaptation (not in the original fixture): the groups
        # above are pairwise node-disjoint, so the 1-skeleton has 26 connected
        # components. The original code tolerated that; this
        # library's ClassicalEmbedder calls require_connected() unconditionally
        # and refuses a fragmented graph. A
        # single-edge spanning-tree chain across the component representatives
        # connects everything without closing any new triangle (a tree has no
        # cycles), so it changes none of the candidate/negative counts this test
        # depends on.
        reps = [3] + [4 + 3 * i for i in range(10)] + [open_base + 3 * j for j in range(15)]
        facets.extend(zip(reps, reps[1:]))
        sc = SimplicialComplex(simplices=facets, n_nodes=n_nodes, name="nested_retain")

        # frac=1.0 makes EVERY candidate triangle land in the test set and
        # NONE in `train` (n_test = len(triangles) exactly), regardless of
        # the permutation seed. That removes randomness from the assertion:
        # with the maximality filter, (0, 1, 2) never becomes a candidate at
        # all, so it can only reach training via `retained` -- deterministically
        # present. Without the filter, (0, 1, 2) becomes an 11th candidate, so
        # at frac=1.0 it is swept into `test` (and dropped from `train`, which
        # is empty) and is deterministically ABSENT from training instead.
        spy = _RecordingEmbedder(emb.ClassicalEmbedder(**presets.EMBEDDING_PARAMS["synthetic_control"]))
        out = tasks.simplex_prediction(sc, spy, frac=1.0, n_neg_per_pos=1, seed=0)

        self.assertTrue(np.isfinite(out["auc"]), "fixture must reach a full run")
        self.assertIsNotNone(spy.seen_facets, ".fit() must have been called")
        self.assertIn((0, 1, 2), spy.seen_facets)      # non-maximal triangle: retained
        self.assertIn((0, 1, 2, 3), spy.seen_facets)   # its parent: retained


class TestDistortion(unittest.TestCase):
    def test_identical_matrices_have_zero_distortion(self):
        rng = np.random.default_rng(0)
        D = rng.uniform(1, 5, (20, 20))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0.0)
        out = tasks.distortion_and_map(D, D)
        self.assertLess(out["avg_distortion"], 1e-9)

    def test_scaled_matrix_still_has_low_distortion(self):
        rng = np.random.default_rng(1)
        D = rng.uniform(1, 5, (20, 20))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0.0)
        self.assertLess(tasks.distortion_and_map(2.0 * D, D)["avg_distortion"], 1e-9)


if __name__ == "__main__":
    unittest.main()
