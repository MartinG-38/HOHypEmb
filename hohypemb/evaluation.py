"""Evaluation statistics: classification, silhouettes, prediction tasks, multiple testing.

Every headline number in the paper is produced by this module: node
classification and clustering scores per embedding method, and the paired
statistical machinery (Wilcoxon signed-rank test with Holm-Bonferroni
correction, percentile bootstrap) used to decide whether HHO's advantage
over each baseline is real.
"""

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, wilcoxon
from scipy.stats import t as student_t
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, average_precision_score, f1_score,
    normalized_mutual_info_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from hohypemb import geometry as geo
from hohypemb.complex import SimplicialComplex
from hohypemb.embedders.base import EmbeddingResult


def make_splits(labels, n_splits: int = 5, n_repeats: int = 10,
                seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified CV splits, generated ONCE and reused across every method.

    This function must be called exactly once per dataset, and the returned
    list handed unchanged to ``node_classification`` for every method under
    comparison. That sharing is what makes the comparison PAIRED: fold i for
    HHO and fold i for a baseline train and test on the identical node sets,
    so the per-fold score difference isolates the method, not sampling noise
    from re-drawn splits. A paired Wilcoxon test (see
    ``paired_wilcoxon_table``) is only valid under that alignment -- if each
    method regenerated its own splits, the "paired" differences would just be
    noise and every downstream p-value would be meaningless. The folds are
    therefore drawn once, from ``seed``, and shared by all methods.
    """
    labels = np.asarray(labels)
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    return list(rskf.split(np.zeros(len(labels)), labels))


def node_classification(D: np.ndarray, labels, splits, k: int = 10) -> Dict[str, np.ndarray]:
    """k-NN classification on a precomputed distance matrix, one score per fold.

    Takes ``splits`` from ``make_splits`` rather than generating its own, so
    that calling this once per method on the SAME ``splits`` list yields
    per-fold arrays that line up index-for-index across methods -- the
    alignment the paired test in ``paired_wilcoxon_table`` depends on.
    """
    labels = np.asarray(labels)
    accuracy, macro_f1 = [], []
    for train, test in splits:
        clf = KNeighborsClassifier(n_neighbors=min(k, len(train)), metric="precomputed")
        clf.fit(D[np.ix_(train, train)], labels[train])
        pred = clf.predict(D[np.ix_(test, train)])
        accuracy.append(float((pred == labels[test]).mean()))
        macro_f1.append(float(f1_score(labels[test], pred, average="macro", zero_division=0)))
    return {"accuracy": np.array(accuracy), "macro_f1": np.array(macro_f1)}


def silhouette_with_ci(D: np.ndarray, labels, n_boot: int = 300,
                       seed: int = 0) -> Dict[str, float]:
    """Silhouette on a precomputed distance matrix, with a bootstrap 95% interval.

    The point estimate is a single silhouette score; the interval comes from
    resampling nodes with replacement and recomputing the score, which is the
    only way to attach uncertainty to a statistic (silhouette) that has no
    closed-form standard error.
    """
    labels = np.asarray(labels)
    score = float(silhouette_score(D, labels, metric="precomputed"))

    rng = np.random.default_rng(seed)
    n = D.shape[0]
    draws = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        lab = labels[idx]
        if len(set(lab.tolist())) < 2:
            continue
        draws.append(silhouette_score(D[np.ix_(idx, idx)], lab, metric="precomputed"))
    if not draws:
        return {"score": score, "lo": score, "hi": score}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"score": score, "lo": float(lo), "hi": float(hi)}


def rank_normalised_distance(D: np.ndarray) -> np.ndarray:
    """Replace every pairwise distance with its RANK among all pairs, rescaled to [0, 1].

    Helper for ``rank_normalised_silhouette`` below; exposed separately so the
    transform itself can be tested and reused. Ranking is computed once over
    the condensed upper triangle (``scipy.stats.rankdata``, ties averaged),
    then mirrored back into a symmetric matrix with a zero diagonal.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    if len(iu[0]) == 0:
        return np.zeros_like(D)
    ranks = rankdata(D[iu])
    denom = len(ranks) - 1
    scaled = (ranks - 1) / denom if denom > 0 else np.zeros_like(ranks)
    R = np.zeros_like(D)
    R[iu] = scaled
    R = R + R.T
    return R


def rank_normalised_silhouette(D: np.ndarray, labels, n_boot: int = 300,
                               seed: int = 0) -> Dict[str, float]:
    """Silhouette computed on a RANK-transformed distance matrix, with a bootstrap CI.

    Why this exists: raw silhouette is scale
    INVARIANT (uniformly rescaling every distance by a constant does not
    change it) but is NOT invariant to a change in the overall SHAPE of the
    distance distribution -- and the embedders compared in this project are
    not commensurable on that axis. On cora, ``euclidean_spectral``'s radii
    span from 0.0016 (median) to 0.172 (max), a ~108x spread that collapses
    99% of the map into a speck near the origin; silhouette penalises that
    collapse heavily, producing a very negative score that mostly reflects
    SCALE, not cluster incoherence. The hyperbolic embedders' radii, by
    contrast, are pinned into [0.88, 0.98] by the shared radial rule -- a
    narrow spread that is exactly what raw silhouette rewards, regardless of
    whether the angular (similarity) coordinate is any good. Comparing raw
    silhouette across these two families therefore partly measures which
    radial rule was used, not which embedding organises the data better.

    Replacing each distance with its RANK among all pairwise distances is a
    monotone, metric-free transform: it preserves every point's
    near/far ORDERING (which is all silhouette actually uses) while
    discarding the distance distribution's scale and shape entirely, making
    the resulting silhouette comparable across embedders whose raw distance
    distributions are not. This does not replace ``silhouette_with_ci``'s
    raw score -- both are reported, because the raw comparison is still a
    real (if scale-confounded) property of the two spaces once that caveat
    is stated.
    """
    labels = np.asarray(labels)
    R = rank_normalised_distance(D)
    score = float(silhouette_score(R, labels, metric="precomputed"))

    rng = np.random.default_rng(seed)
    n = D.shape[0]
    draws = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        lab = labels[idx]
        if len(set(lab.tolist())) < 2:
            continue
        draws.append(silhouette_score(R[np.ix_(idx, idx)], lab, metric="precomputed"))
    if not draws:
        return {"score": score, "lo": score, "hi": score}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"score": score, "lo": float(lo), "hi": float(hi)}


def community_recovery(result: EmbeddingResult, labels, K: int,
                       seed: int = 0) -> Dict[str, float]:
    """Cluster the embedding coordinates and score against the planted partition.

    ``labels`` here is used strictly for scoring the clustering found by
    KMeans on ``result.coords`` -- it never feeds back into the embedding
    itself: ground-truth labels are used for evaluation only.
    """
    labels = np.asarray(labels)
    pred = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(result.coords)
    return {
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "ari": float(adjusted_rand_score(labels, pred)),
    }


# ------------------------------------------------------------------ statistics


def holm_correction(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values.

    HHO is compared against several baselines at once (multiple comparisons),
    so uncorrected p-values would overstate significance -- some baseline
    would look "significantly worse" by chance alone even with no real
    effect. Holm's step-down procedure controls the family-wise error rate
    without being as conservative as a flat Bonferroni correction: sort
    p-values ascending, multiply the k-th smallest by the number of
    hypotheses still "in play" (m - rank), then enforce monotonicity with a
    running maximum so that adjusted p-values never decrease as the raw
    p-value increases.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap interval for the mean.

    Used both standalone and inside ``paired_wilcoxon_table`` (to interval
    the mean paired difference).
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_wilcoxon_table(scores_by_method: Dict[str, np.ndarray],
                          reference: str = "hho") -> pd.DataFrame:
    """Compare every method against ``reference`` on shared folds.

    Reports the mean difference, its bootstrap CI, the raw Wilcoxon p-value and
    the Holm-adjusted p-value across all baselines.

    This is a PAIRED test (Wilcoxon signed-rank on per-fold differences), not
    an unpaired one, because every method's per-fold scores come from the
    exact same ``splits`` produced once by ``make_splits`` -- fold i for
    every method trained and tested on identical node sets. That shared
    randomness is exactly what a paired test is designed to exploit: it
    removes fold-to-fold difficulty as a nuisance variable and tests only
    whether one method beats another on the SAME folds. Mismatched fold
    counts would mean the "pairs" no longer correspond to the same split, so
    that is refused with a ``ValueError`` rather than silently truncated or
    zipped by position, which could pair scores for entirely different
    train/test sets. The raw p-values are then corrected via ``holm_correction``
    because several baselines are compared against the same reference at
    once (see ``holm_correction`` docstring).
    """
    if reference not in scores_by_method:
        raise KeyError(f"reference {reference!r} not in {sorted(scores_by_method)}")
    ref = np.asarray(scores_by_method[reference], dtype=float)

    rows, raw_p = [], []
    for method, scores in scores_by_method.items():
        if method == reference:
            continue
        other = np.asarray(scores, dtype=float)
        if len(other) != len(ref):
            raise ValueError(
                f"{method}: {len(other)} folds but reference has {len(ref)}; "
                "methods must share the same splits"
            )
        delta = ref - other
        if np.allclose(delta, 0.0):
            p = 1.0
        else:
            p = float(wilcoxon(ref, other).pvalue)
        raw_p.append(p)
        lo, hi = bootstrap_ci(delta)
        rows.append({
            "method": method,
            "mean": float(other.mean()),
            "reference_mean": float(ref.mean()),
            "delta_mean": float(delta.mean()),
            "delta_lo": lo,
            "delta_hi": hi,
            "wins": int((delta > 0).sum()),
            "n_folds": len(ref),
            "p_raw": p,
        })

    table = pd.DataFrame(rows)
    table["p_holm"] = holm_correction(raw_p) if raw_p else []
    return table.sort_values("delta_mean", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------- prediction tasks


def degree_matched_negatives(candidates, positives, degrees, n_per_pos: int = 10,
                             tol: float = 0.35, seed: int = 0) -> List[Tuple[int, ...]]:
    """Draw negatives whose sorted degree profile matches each positive's.

    Uniform negatives are not an option. On cora, closed triangles have mean node
    degree 8.2 against 23.0 for open ones; because the radial rule places
    high-degree nodes near the origin, uniform negatives are automatically
    'compact' and the task degenerates into measuring degree.
    """
    degrees = np.asarray(degrees, dtype=float)
    rng = np.random.default_rng(seed)
    positive_set = {frozenset(p) for p in positives}

    pool = [c for c in candidates if frozenset(c) not in positive_set]
    if not pool:
        return []
    pool_arr = np.array([sorted(c) for c in pool], dtype=int)
    pool_profile = np.sort(np.log1p(degrees[pool_arr]), axis=1)

    chosen: List[Tuple[int, ...]] = []
    used = set()
    for p in positives:
        target = np.sort(np.log1p(degrees[np.asarray(sorted(p), dtype=int)]))
        if pool_profile.shape[1] != len(target):
            continue
        gap = np.abs(pool_profile - target[None, :]).mean(axis=1)
        eligible = np.flatnonzero(gap <= tol)
        if len(eligible) < n_per_pos:
            eligible = np.argsort(gap)[: max(n_per_pos * 4, n_per_pos)]
        rng.shuffle(eligible)
        taken = 0
        for i in eligible:
            key = frozenset(pool[i])
            if key in used:
                continue
            used.add(key)
            chosen.append(tuple(pool[i]))
            taken += 1
            if taken >= n_per_pos:
                break
    return chosen


def degree_match_quality(positives, negatives, degrees) -> Dict[str, float]:
    """Report how well the negatives match the positives on degree."""
    degrees = np.asarray(degrees, dtype=float)
    pos = np.array([degrees[list(p)].mean() for p in positives])
    neg = np.array([degrees[list(n)].mean() for n in negatives])
    if len(pos) == 0 or len(neg) == 0:
        return {"ks_stat": float("nan"), "ks_p": float("nan"),
                "mean_pos": float("nan"), "mean_neg": float("nan")}
    ks = ks_2samp(pos, neg)
    return {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "mean_pos": float(pos.mean()), "mean_neg": float(neg.mean())}


def pairwise_link_prediction(sc: SimplicialComplex, embedder, frac: float = 0.2,
                             seed: int = 0) -> Dict[str, float]:
    """Hide a fraction of edges, embed, and rank them against degree-matched non-edges.

    Held-out edges are chosen one at a time, skipping any edge whose removal
    would leave one of its endpoints with degree 0 in the training graph
    (checked against the graph as it is being thinned, not the original, so
    two edges that would jointly isolate a degree-2 node are also caught),
    or would split the training graph into more than one connected
    component (``ClassicalEmbedder`` -- and any other embedder built on
    Isomap or spectral coordinates -- refuses a disconnected 1-skeleton
    unconditionally). On cora (giant component, 723 nodes), naive
    random removal of 20% of edges isolates 26 nodes and fragments the graph
    into several components; the library's radial rule maps a degree-0 node
    to radius 1, on the unit circle, which ``EmbeddingResult.check_non_degenerate``
    then refuses, and ``require_connected`` refuses the fragmented graph
    outright, rather than either silently returning a degenerate embedding. This is a graph-validity guard applied identically to every
    embedder -- it does not touch scoring or negative sampling, so it favours
    no method.

    Consequence to disclose, not just the mechanism: the guard systematically
    excludes edges incident to degree-1 nodes from the held-out set, and
    those are exactly the hardest edges to predict (a leaf's only neighbour
    is removed from the training graph entirely, leaving nothing local to
    infer it from). Excluding them makes the reported AUC mildly optimistic
    relative to an unbiased holdout. Treat this function's output as a
    sanity check that the embedding pipeline carries real, learnable
    structure -- not as a headline benchmark number.
    """
    rng = np.random.default_rng(seed)
    G = sc.one_skeleton
    edges = list(G.edges())
    n_hold = max(1, int(frac * len(edges)))

    G_train = G.copy()
    held = []
    for i in rng.permutation(len(edges)):
        if len(held) >= n_hold:
            break
        u, v = edges[i]
        if G_train.degree(u) > 1 and G_train.degree(v) > 1:
            G_train.remove_edge(u, v)
            if nx.is_connected(G_train):
                held.append(edges[i])
            else:
                G_train.add_edge(u, v)

    train_complex = SimplicialComplex(
        simplices=[f for f in sc.simplices
                if all(G_train.has_edge(u, v) for u, v in combinations(f, 2))],
        n_nodes=sc.n_nodes, name=f"{sc.name}_lp", base_graph=G_train,
    )

    degrees = np.array([G_train.degree(i) for i in range(sc.n_nodes)], dtype=float)
    non_edges = []
    attempts = 0
    while len(non_edges) < len(held) * 4 and attempts < 200000:
        attempts += 1
        u, v = int(rng.integers(0, sc.n_nodes)), int(rng.integers(0, sc.n_nodes))
        if u != v and not G.has_edge(u, v):
            non_edges.append(tuple(sorted((u, v))))
    negatives = degree_matched_negatives(non_edges, held, degrees, n_per_pos=1, seed=seed)
    if not negatives:
        negatives = non_edges[: len(held)]

    result = embedder.fit(train_complex, seed=seed)
    D = result.distance_matrix()
    groups = list(held) + list(negatives)
    y = np.r_[np.ones(len(held)), np.zeros(len(negatives))]
    scores = np.array([-D[u, v] for u, v in groups])

    return {"auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "n_pos": float(len(held)), "n_neg": float(len(negatives)),
            **{f"degree_{k}": v
               for k, v in degree_match_quality(held, negatives, degrees).items()}}


def simplex_prediction(sc: SimplicialComplex, embedder, frac: float = 0.2,
                       n_neg_per_pos: int = 10, seed: int = 0,
                       agg: str = "mean") -> Dict[str, float]:
    """Hold out facets, embed on the rest, rank them against degree-matched open cliques.

    EXPECTED OUTCOME IS NEGATIVE for HHO (classical beats HHO at this task on
    cora). This function exists to document that honestly in notebook
    section 8. Do not tune parameters until the number looks good.

    The negative class (``open_triangles`` below) excludes any triangle that
    is a sub-face of a larger retained facet, not just triangles that are
    themselves facets -- a sub-face is a genuine simplex whose parent facet
    is always in the training complex, so scoring it as a negative would
    reward HHO for complying with structure it was explicitly told to pull
    together, rather than for predicting unseen structure. (This mirrors,
    on the negative side, the maximality filter applied to the positive
    class below -- both exist to prevent the training complex from leaking
    group membership into a nominally held-out group.)

    Candidate positives are restricted to MAXIMAL triangle facets -- those
    not strictly contained in any larger facet. ``SimplicialComplex``
    deduplicates and sorts facets but does not enforce maximality, and cora
    genuinely nests (a 3-author article whose authors all also appear on a
    4-author article yields both as facets: 24 of cora's 179 triangle facets
    are such sub-faces). Holding out a non-maximal triangle would leak: the
    larger facet that contains it stays in the training complex, so the
    held-out group's membership information survives training anyway.

    A non-maximal triangle is merely untestable, not fake: it is still real
    observed structure and MUST stay in the training complex (unlike the
    triangles actually chosen for the held-out ``test`` set, which are
    removed from training). The retained set passed to ``train_complex`` is
    therefore everything that was never a maximal-triangle candidate --
    every non-triangle facet AND every non-maximal triangle -- not merely
    every non-triangle facet; the latter would silently drop the non-maximal
    triangles from the embedding itself, corrupting the measurement rather
    than just narrowing the test set.
    """
    rng = np.random.default_rng(seed)
    facet_sets = [frozenset(f) for f in sc.simplices]
    candidates = [
        f for f in facet_sets
        if len(f) == 3 and not any(f < other for other in facet_sets)
    ]
    candidate_keys = set(candidates)
    triangles = [tuple(sorted(f)) for f in candidates]
    if len(triangles) < 10:
        return {"auc": float("nan"), "ap": float("nan"),
                "n_pos": 0.0, "n_neg": 0.0, "degree_ks": float("nan")}

    perm = rng.permutation(len(triangles))
    n_test = max(1, int(frac * len(triangles)))
    test = [triangles[i] for i in perm[:n_test]]
    train = [triangles[i] for i in perm[n_test:]]
    # Everything not eligible to be held out stays in training: non-triangle
    # facets AND non-maximal triangles (real structure, just untestable).
    retained = [tuple(sorted(f)) for f in facet_sets if f not in candidate_keys]

    train_complex = SimplicialComplex(
        simplices=train + retained, n_nodes=sc.n_nodes,
        name=f"{sc.name}_sp", base_graph=sc.one_skeleton,
    )

    # A triangle that is a SUB-FACE of a larger retained facet (e.g. {a,b,c}
    # subset of {a,b,c,d}) is a genuine simplex of the complex, not an "open"
    # (never-closed) one: its parent facet is size >= 4, so it can never be a
    # candidate positive, and it is always retained in training (see
    # ``retained`` above). HHO is explicitly instructed to pull those three
    # nodes together via the parent's layer-2 coupling, so scoring such a
    # triangle as a negative measures compliance with the input, not
    # prediction. ``<=`` (not ``<``) excludes a triangle both when it IS a
    # facet and when it is a proper sub-face of one -- reusing ``facet_sets``
    # from the maximality filter above, which is exactly the same test.
    open_triangles = [
        t for t in sc.triangles_in_skeleton()
        if not any(frozenset(t) <= f for f in facet_sets)
    ]
    degrees = np.array(
        [sc.one_skeleton.degree(i) for i in range(sc.n_nodes)], dtype=float
    )
    negatives = degree_matched_negatives(
        open_triangles, test, degrees, n_per_pos=n_neg_per_pos, seed=seed
    )
    if len(negatives) < len(test):
        return {"auc": float("nan"), "ap": float("nan"),
                "n_pos": float(len(test)), "n_neg": float(len(negatives)),
                "degree_ks": float("nan")}

    result = embedder.fit(train_complex, seed=seed)
    D = result.distance_matrix()
    groups = list(test) + list(negatives)
    y = np.r_[np.ones(len(test)), np.zeros(len(negatives))]
    scores = np.array([-geo.group_compactness(D, g, agg) for g in groups])

    quality = degree_match_quality(test, negatives, degrees)
    return {"auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "n_pos": float(len(test)), "n_neg": float(len(negatives)),
            "degree_ks": quality["ks_stat"],
            "degree_mean_pos": quality["mean_pos"],
            "degree_mean_neg": quality["mean_neg"]}


def distortion_and_map(D_emb: np.ndarray, D_ref: np.ndarray) -> Dict[str, float]:
    """Scale-invariant average distortion between an embedding and a reference metric.

    SUPPORTING METRIC ONLY: HHO already optimises a
    Laplacian containing the higher-order relations, so a distortion advantage
    measured against those same relations would be partly circular.
    """
    iu = np.triu_indices(D_ref.shape[0], k=1)
    ref = D_ref[iu]
    emb_d = D_emb[iu]
    mask = (ref > 0) & np.isfinite(emb_d) & np.isfinite(ref)
    if not mask.any():
        return {"avg_distortion": float("nan"), "scale": float("nan")}
    scale = float(np.sum(emb_d[mask] * ref[mask]) / np.sum(ref[mask] ** 2))
    if scale <= 0:
        scale = 1.0
    rel = np.abs(emb_d[mask] / scale - ref[mask]) / ref[mask]
    return {"avg_distortion": float(rel.mean()), "scale": scale}


def partial_spearman(x, y, z) -> Tuple[float, float]:
    """Spearman correlation of ``x`` and ``y`` after removing the effect of ``z``.

    Ranks of ``x`` and ``y`` are regressed linearly (with intercept) on the rank of ``z``;
    the result is the Pearson correlation of the residuals, with a two-sided p-value from a
    t-test with n - 3 degrees of freedom. Returns ``(nan, nan)`` for n < 4 or constant residuals.
    """
    x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones(n), rz])
    # Accelerate BLAS raises spurious FP flags on this matmul for some n; Z and the lstsq
    # solution are finite (same false positive as _zhou_hypergraph_affinity in baselines.py).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        res_x = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        res_y = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if np.std(res_x) < 1e-12 or np.std(res_y) < 1e-12:
        return float("nan"), float("nan")
    r = float(np.corrcoef(res_x, res_y)[0, 1])
    t_stat = r * np.sqrt((n - 3) / max(1 - r ** 2, 1e-12))
    return r, float(2 * student_t.sf(abs(t_stat), n - 3))
