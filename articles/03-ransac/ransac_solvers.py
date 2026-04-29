# ransac_solvers.py
# OLS and RANSAC line fitting.
# Reference: Fischler & Bolles, CACM 1981.

import numpy as np
from math import log, ceil


def _sample_pair(n, rng):
    return rng.choice(n, size=2, replace=False)


def _line_through(pts, i, j):
    # returns None if nearly vertical
    dx = pts[j, 0] - pts[i, 0]
    if abs(dx) < 1e-9:
        return None, None
    slope = (pts[j, 1] - pts[i, 1]) / dx
    intercept = pts[i, 1] - slope * pts[i, 0]
    return slope, intercept


def _consensus(pts, slope, intercept, threshold):
    residuals = np.abs(pts[:, 1] - (slope * pts[:, 0] + intercept))
    mask = residuals < threshold
    return int(mask.sum()), mask


def _required_iters(n_consensus, n_total, confidence=0.99):
    # standard formula from Fischler & Bolles
    # N = log(1 - conf) / log(1 - w^2),  w = inlier fraction estimate
    w = n_consensus / n_total
    p = w ** 2
    if p <= 0:
        return 99_999
    if p >= 1:
        return 1
    return ceil(log(1 - confidence) / log(1 - p))


def fit_ols(pts):
    X = np.column_stack([pts[:, 0], np.ones(len(pts))])
    (slope, intercept), *_ = np.linalg.lstsq(X, pts[:, 1], rcond=None)
    return float(slope), float(intercept)


def fit_ransac(pts, inlier_threshold=5.0, confidence=0.99, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pts)

    # start pessimistic — assume 10% inliers, shrink budget as we find better models
    budget = _required_iters(max(2, round(0.10 * n)), n, confidence)

    best_count = 0
    best_slope = 0.0
    best_intercept = 0.0
    best_mask = np.zeros(n, dtype=bool)
    iteration = 0

    while iteration < budget:
        i, j = _sample_pair(n, rng)
        slope, intercept = _line_through(pts, i, j)

        if slope is None:
            iteration += 1
            continue

        count, mask = _consensus(pts, slope, intercept, inlier_threshold)

        if count > best_count:
            best_count = count
            best_slope = slope
            best_intercept = intercept
            best_mask = mask
            budget = min(_required_iters(count, n, confidence), 50_000)

        iteration += 1

    return best_slope, best_intercept, iteration, best_mask
