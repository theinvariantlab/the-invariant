import numpy as np


def sample_p1(n, rng):
    u = rng.random(n)
    return np.where(u < 0.25, 2.0 * u, (2.0 * u + 1.0) / 3.0)


def sample_p2(n, rng):
    u = rng.random(n)
    x = np.empty(n)
    m0 = u < 0.25
    m1 = (u >= 0.25) & (u < 0.75)
    m2 = u >= 0.75
    x[m0] = u[m0] / 2.0
    x[m1] = 5.0 * u[m1] / 4.0 - 3.0 / 16.0
    x[m2] = u[m2]
    return x


def sample_p3(n, rng):
    u = rng.random(n)
    return np.tan((np.pi / 4.0) * u)


def sample_p4(n, rng):
    u = rng.random(n)
    x = np.empty(n)
    m0 = u < 1.0 / 3.0
    m1 = (u >= 1.0 / 3.0) & (u < 2.0 / 3.0)
    m2 = u >= 2.0 / 3.0
    x[m0] = np.sqrt(u[m0] / 3.0)
    x[m1] = (1.0 + np.sqrt(3.0 * u[m1] - 1.0)) / 3.0
    x[m2] = (2.0 + np.sqrt(3.0 * u[m2] - 2.0)) / 3.0
    return x


def sample_p5(n, rng):
    c = 2.0 * np.pi / 3.0
    samples = []
    collected = 0
    while collected < n:
        batch = min(10 * n, 200_000)
        x_prop = rng.random(batch)
        v_prop = rng.random(batch) * c
        left = x_prop < 0.5
        f_x = np.where(
            left,
            (np.pi / 3.0) * np.sin(2.0 * np.pi * x_prop),
            (2.0 * np.pi / 3.0) * np.sin(2.0 * np.pi * (x_prop - 0.5)),
        )
        accepted = x_prop[v_prop < f_x]
        samples.append(accepted)
        collected += len(accepted)
    return np.concatenate(samples)[:n]


def p1_pdf(x):
    return np.where(
        (x >= 0) & (x < 0.5), 0.5,
        np.where((x >= 0.5) & (x <= 1.0), 1.5, 0.0)
    )


def p2_pdf(x):
    return np.where(
        (x >= 0) & (x <= 1.0 / 8.0), 2.0,
        np.where(
            (x > 1.0 / 8.0) & (x < 3.0 / 4.0), 4.0 / 5.0,
            np.where((x >= 3.0 / 4.0) & (x <= 1.0), 1.0, 0.0)
        )
    )


def p3_pdf(x):
    norm = np.arctan(1.0)
    return np.where((x >= 0) & (x <= 1.0), 1.0 / (norm * (1.0 + x ** 2)), 0.0)


def p4_pdf(x):
    return np.where(
        (x >= 0) & (x <= 1.0),
        2.0 * (3.0 * x - np.floor(3.0 * x)),
        0.0,
    )


def p5_pdf(x):
    left = (x >= 0) & (x < 0.5)
    right = (x >= 0.5) & (x <= 1.0)
    return np.where(
        left, (np.pi / 3.0) * np.sin(2.0 * np.pi * x),
        np.where(
            right,
            (2.0 * np.pi / 3.0) * np.sin(2.0 * np.pi * (x - 0.5)),
            0.0,
        )
    )


DENSITIES = [
    ("p1", sample_p1, p1_pdf),
    ("p2", sample_p2, p2_pdf),
    ("p3", sample_p3, p3_pdf),
    ("p4", sample_p4, p4_pdf),
    ("p5", sample_p5, p5_pdf),
]