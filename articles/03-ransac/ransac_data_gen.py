# True line is y = x + 5.

import numpy as np

TRUE_SLOPE = 1.0
TRUE_INTERCEPT = 5.0
WALL_NOISE_STD = 3.0


def generate(N=100, inlier_frac=0.65, seed=0):
    rng = np.random.default_rng(seed)
    M = round(N * inlier_frac)

    xs_wall = rng.uniform(5, 95, M)
    ys_wall = TRUE_SLOPE * xs_wall + TRUE_INTERCEPT + rng.normal(0, WALL_NOISE_STD, M)

    xs_noise = rng.uniform(0, 100, N - M)
    ys_noise = rng.uniform(0, 100, N - M)

    pts = np.vstack([
        np.column_stack([xs_wall, ys_wall]),
        np.column_stack([xs_noise, ys_noise]),
    ])

    true_mask = np.zeros(N, dtype=bool)
    true_mask[:M] = True

    return pts, true_mask
