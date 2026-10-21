"""Hyperbolic geometry primitives for the Poincare disk.

All matrix-valued functions are vectorised; ``hyp_dist`` is the scalar
reference implementation, copied from MLNHypEmb.
"""

from typing import Sequence

import numpy as np

from hohypemb._core.radial import check_radius_rule, degree_radius

EPS_DISK = 1e-9


def clip_to_disk(emb: np.ndarray, eps: float = EPS_DISK) -> np.ndarray:
    """Pull any point on or outside the unit circle back to radius ``1 - eps``.

    A node of degree 0 receives radius ``1 - tanh(0) = 1`` from the library's
    radial rule, which makes ``(1 - ||u||^2)(1 - ||v||^2) = 0`` and every
    distance infinite. Clipping is therefore mandatory, not cosmetic.
    """
    emb = np.asarray(emb, dtype=float)
    single = emb.ndim == 1
    if single:
        emb = emb[None, :]
    r = np.linalg.norm(emb, axis=1)
    bad = r >= 1 - eps
    if bad.any():
        emb = emb.copy()
        emb[bad] = emb[bad] * ((1 - eps) / r[bad])[:, None]
    return emb[0] if single else emb


def hyp_dist_matrix(emb: np.ndarray, eps: float = EPS_DISK) -> np.ndarray:
    """Full pairwise hyperbolic distance matrix.

    d(x, y) = arccosh(1 + 2||x - y||^2 / ((1 - ||x||^2)(1 - ||y||^2)))
    """
    emb = clip_to_disk(emb, eps)
    nrm = np.sum(emb ** 2, axis=1)
    # Accelerate BLAS raises spurious FP flags in matmul; inputs are clipped disk points
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        diff_sq = np.clip(nrm[:, None] + nrm[None, :] - 2 * emb @ emb.T, 0.0, None)
    denom = (1 - nrm[:, None]) * (1 - nrm[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        cosh_d = np.where(denom > 0, 1 + 2 * diff_sq / denom, np.inf)
    D = np.arccosh(np.clip(cosh_d, 1.0, None))
    np.fill_diagonal(D, 0.0)
    return D


def euclidean_dist_matrix(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=float)
    nrm = np.sum(emb ** 2, axis=1)
    # Accelerate BLAS raises spurious FP flags in matmul; inputs are clipped disk points
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        D = np.sqrt(np.clip(nrm[:, None] + nrm[None, :] - 2 * emb @ emb.T, 0.0, None))
    np.fill_diagonal(D, 0.0)
    return D


def angular_dist_matrix(emb: np.ndarray) -> np.ndarray:
    """Angular separation in [0, pi], ignoring the radial coordinate entirely.

    HHO leaves radii untouched (verified: max|r_classical - r_HHO| = 3.3e-16),
    so this isolates exactly what the method changes.
    """
    emb = np.asarray(emb, dtype=float)
    theta = np.arctan2(emb[:, 1], emb[:, 0])
    d = np.abs(theta[:, None] - theta[None, :])
    A = np.minimum(d, 2 * np.pi - d)
    np.fill_diagonal(A, 0.0)
    return A


def radii(emb: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(emb, dtype=float), axis=1)


def hyp_triangle_area(p1, p2, p3) -> float:
    """Hyperbolic area via the angular excess formula: pi - (alpha + beta + gamma).

    Returns ``nan`` only when two vertices coincide, which makes the angle
    sum undefined; this is returned rather than raising, so that callers can
    filter it out with ``np.isfinite``. Three *distinct* points on a common
    geodesic are not undefined: their true hyperbolic area is 0 (angle sum
    pi), and the formula returns a finite value near 0 for that case. A
    caller that wants to exclude collinear triangles must filter on
    ``area > tol``, not on ``np.isfinite``.
    """
    p1, p2, p3 = (clip_to_disk(np.asarray(p, dtype=float)) for p in (p1, p2, p3))
    stack = np.vstack([p1, p2, p3])
    D = hyp_dist_matrix(stack)
    a, b, c = D[1, 2], D[0, 2], D[0, 1]
    if min(a, b, c) <= 0:
        return float("nan")

    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.arccos(
            np.clip((np.cosh(b) * np.cosh(c) - np.cosh(a)) / (np.sinh(b) * np.sinh(c)), -1, 1)
        )
        beta = np.arccos(
            np.clip((np.cosh(a) * np.cosh(c) - np.cosh(b)) / (np.sinh(a) * np.sinh(c)), -1, 1)
        )
        gamma = np.arccos(
            np.clip((np.cosh(a) * np.cosh(b) - np.cosh(c)) / (np.sinh(a) * np.sinh(b)), -1, 1)
        )
    area = float(np.pi - (alpha + beta + gamma))
    return area if np.isfinite(area) else float("nan")


def poincare_geodesic(p1, p2, num_points: int = 20) -> np.ndarray:
    """Exact geodesic: an arc of the circle orthogonal to the unit circle.

    Degenerates to a straight segment when the two points are collinear with
    the origin.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    def _segment():
        t = np.linspace(0, 1, num_points)
        return (1 - t)[:, None] * p1[None, :] + t[:, None] * p2[None, :]

    if np.linalg.norm(p1 - p2) < 1e-8:
        return np.repeat(p1[None, :], num_points, axis=0)
    if abs(p1[0] * p2[1] - p1[1] * p2[0]) < 1e-8:
        return _segment()

    mid = (p1 + p2) / 2
    direction = p2 - p1
    perp = np.array([-direction[1], direction[0]])
    norm_perp_sq = float(perp @ perp)
    if norm_perp_sq < 1e-10:
        return _segment()
    perp = perp / np.sqrt(norm_perp_sq)

    mid_p1 = mid - p1
    A = 2 * (mid @ perp) - 2 * (mid_p1 @ perp)
    B = (mid @ mid) - (mid_p1 @ mid_p1) - 1
    if abs(A) < 1e-10:
        return _segment()

    center = mid + (-B / A) * perp
    radius = float(np.linalg.norm(center - p1))
    angle1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    angle2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
    if angle2 - angle1 > np.pi:
        angle2 -= 2 * np.pi
    elif angle2 - angle1 < -np.pi:
        angle2 += 2 * np.pi
    angles = np.linspace(angle1, angle2, num_points)
    return center[None, :] + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def klein_barycenter(points: np.ndarray) -> np.ndarray:
    """Einstein midpoint of Poincare-disk points, returned in Poincare coordinates.

    Note: MLNHypEmb, Tools.py (Klein_baycenter) weights by
    ``sqrt(1 - ||x||^2)``, the reciprocal of the Lorentz factor. This function
    implements the standard Einstein midpoint (weights ``1/sqrt(1 - ||x||^2)``).
    It is a new function, not a modification of the library.
    """
    p = clip_to_disk(np.asarray(points, dtype=float))
    denom = 1 + np.sum(p ** 2, axis=1, keepdims=True)
    k = 2 * p / denom                                   # Poincare -> Klein
    lorentz = 1.0 / np.sqrt(np.clip(1 - np.sum(k ** 2, axis=1, keepdims=True), 1e-15, None))
    bary_k = np.sum(lorentz * k, axis=0) / np.sum(lorentz)
    nrm_sq = float(bary_k @ bary_k)
    return bary_k / (1 + np.sqrt(max(1 - nrm_sq, 0.0)))  # Klein -> Poincare


def group_compactness(D: np.ndarray, group: Sequence[int], agg: str = "mean") -> float:
    """Aggregate pairwise distance within a group. Lower means more compact."""
    idx = np.asarray(list(group), dtype=int)
    sub = D[np.ix_(idx, idx)]
    vals = sub[np.triu_indices(len(idx), k=1)]
    if vals.size == 0:
        return float("nan")
    if agg == "mean":
        return float(np.mean(vals))
    if agg == "max":
        return float(np.max(vals))
    raise ValueError(f"agg must be 'mean' or 'max', got {agg!r}")


def apply_radial_rule(coords2d, degrees, beta: float, rule: str = "degree") -> np.ndarray:
    """Project 2D coordinates onto the Poincare disk using the library's radial rule.

    Normalises each row to unit norm (keeping only its angle), then rescales by
    a radius ``1 - tanh(degrees / beta)``. Reproduces EmbeddingMethod.py:965
    exactly. The radius is only as comparable to HHO's as ``degrees`` is: the
    star-expansion and Zhou baselines pass raw weighted degrees, whereas HHO uses
    RA-preweighted strengths when ``preweight=True``.
    """
    check_radius_rule(rule)
    coords = np.asarray(coords2d, dtype=float)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    unit = coords / np.where(norms > 0, norms, 1.0)
    return clip_to_disk(unit * degree_radius(degrees, beta)[:, None])


def hyp_dist(coord_1, coord_2) -> float:
    """Scalar hyperbolic distance between two points of the Poincaré disk.

    Copied from MLNHypEmb, Tools.py (hyp_dist).
    Copyright (c) 2025, Martin Guillemaud, Sorbonne Université, INRIA (BSD-3-Clause).
    """
    xi, yi = np.asarray(coord_1, dtype=float).flatten()
    xj, yj = np.asarray(coord_2, dtype=float).flatten()
    numerator = 2 * ((xi - xj) ** 2 + (yi - yj) ** 2)
    denominator = (1 - (xi ** 2 + yi ** 2)) * (1 - (xj ** 2 + yj ** 2))
    cosh_d = 1 + numerator / denominator if denominator > 0 else np.inf
    return float(np.arccosh(cosh_d)) if cosh_d >= 1 else 0.0


def hyp_dist_pairs(P: np.ndarray, Q: np.ndarray, eps: float = EPS_DISK) -> np.ndarray:
    """Hyperbolic distance between the rows of ``P`` and the rows of ``Q``, row by row."""
    P = clip_to_disk(np.atleast_2d(np.asarray(P, dtype=float)), eps)
    Q = clip_to_disk(np.atleast_2d(np.asarray(Q, dtype=float)), eps)
    diff_sq = np.sum((P - Q) ** 2, axis=1)
    denom = (1 - np.sum(P ** 2, axis=1)) * (1 - np.sum(Q ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        cosh_d = np.where(denom > 0, 1 + 2 * diff_sq / denom, np.inf)
    return np.arccosh(np.clip(cosh_d, 1.0, None))


def triangle_areas(emb: np.ndarray, triangles) -> np.ndarray:
    """Hyperbolic area of every triangle (rows of node indices), vectorised ``hyp_triangle_area``.

    Returns NaN where two vertices coincide, like the scalar version.
    """
    T = np.asarray(triangles, dtype=int).reshape(-1, 3)
    X = clip_to_disk(np.asarray(emb, dtype=float))
    p1, p2, p3 = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    a, b, c = hyp_dist_pairs(p2, p3), hyp_dist_pairs(p1, p3), hyp_dist_pairs(p1, p2)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.arccos(np.clip((np.cosh(b) * np.cosh(c) - np.cosh(a)) / (np.sinh(b) * np.sinh(c)), -1, 1))
        beta = np.arccos(np.clip((np.cosh(a) * np.cosh(c) - np.cosh(b)) / (np.sinh(a) * np.sinh(c)), -1, 1))
        gamma = np.arccos(np.clip((np.cosh(a) * np.cosh(b) - np.cosh(c)) / (np.sinh(a) * np.sinh(b)), -1, 1))
        area = np.pi - (alpha + beta + gamma)
    valid = (np.minimum(np.minimum(a, b), c) > 0) & np.isfinite(area)
    return np.where(valid, area, np.nan)


def angular_gaps(emb: np.ndarray, triangles) -> np.ndarray:
    """Mean of the three pairwise angular separations of each triangle's vertices (in [0, pi])."""
    T = np.asarray(triangles, dtype=int).reshape(-1, 3)
    X = np.asarray(emb, dtype=float)
    theta = np.arctan2(X[:, 1], X[:, 0])[T]
    gaps = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        d = np.abs(theta[:, i] - theta[:, j]) % (2 * np.pi)
        gaps.append(np.minimum(d, 2 * np.pi - d))
    return np.mean(gaps, axis=0) if len(T) else np.empty(0)


def hyperbolic_radius(emb: np.ndarray, eps: float = EPS_DISK) -> np.ndarray:
    """Hyperbolic distance of every point to the origin, ``2 artanh(r)``."""
    return 2 * np.arctanh(radii(clip_to_disk(emb, eps)))
