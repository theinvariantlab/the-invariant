import numpy as np
from scipy.integrate import quad


def select_bins(data):
    n = len(data)
    if n < 3:
        return 1

    D_max = int(np.floor(n / np.log(n)))

    d_min, d_max = np.min(data), np.max(data)
    if d_min == d_max:
        return 1
    u = (data - d_min) / (d_max - d_min)

    best_D = 1
    best_crit = -np.inf

    for D in range(1, D_max + 1):
        counts, _ = np.histogram(u, bins=D, range=(0.0, 1.0))
        nz = counts[counts > 0]
        log_lik = np.sum(nz * np.log(D * nz / n))
        pen = (D - 1) + (np.log(D) ** 2.5 if D > 1 else 0.0)
        crit = log_lik - pen
        if crit > best_crit:
            best_crit = crit
            best_D = D

    return best_D


def l1_distance(pdf_true, D, data):
    n = len(data)
    counts, edges = np.histogram(data, bins=D, range=(0.0, 1.0))
    bin_width = 1.0 / D

    total = 0.0
    for k in range(D):
        a, b = edges[k], edges[k + 1]
        p_hat = counts[k] / (n * bin_width)
        val, _ = quad(
            lambda x: abs(pdf_true(np.atleast_1d(x))[0] - p_hat),
            a, b, limit=200, epsabs=1e-6,
        )
        total += val
    return total


def optimal_D(pdf_true, data):
    n = len(data)
    D_max = int(np.floor(n / np.log(n)))
    risks = {D: l1_distance(pdf_true, D, data) for D in range(1, D_max + 1)}
    min_risk = min(risks.values())
    tol = max(1e-6, 1e-4 * min_risk)
    minimisers = [D for D, r in risks.items() if r - min_risk < tol]
    return max(minimisers)