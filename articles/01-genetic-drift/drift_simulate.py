# The Invariant -- Genetic Drift
# Simulation engine for the Wright-Fisher model
# github.com/theinvariantlab/the-invariant

import numpy as np


def run(N, generations, M, p0=0.5):
    """
    Run M independent Wright-Fisher chains of size N for a fixed number
    of generations. Returns the full trajectory matrix and absorption times.

    N           : population size
    generations : number of steps to simulate
    M           : number of independent populations
    p0          : initial frequency of red (same for all populations)
    """

    # All M populations start at the same count
    X = np.full(M, int(round(p0 * N)), dtype=int)

    # Pre-allocate the trajectory matrix: one row per generation, one column per population
    trajectories = np.empty((generations + 1, M), dtype=int)
    trajectories[0] = X

    # nan means not yet absorbed
    absorbed_at = np.full(M, np.nan)

    # True = still evolving, False = absorbed at 0 or N
    active = np.ones(M, dtype=bool)

    for t in range(1, generations + 1):

        # If every population has absorbed, fill remaining rows and stop early
        if not active.any():
            trajectories[t:] = trajectories[t - 1]
            break

        # Wright-Fisher step: only update active populations
        # X[active] is a vector of current counts for active populations
        # p is the corresponding vector of red frequencies
        p = X[active] / N
        X[active] = np.random.binomial(N, p)

        # Check which populations just hit an absorbing state (0 or N)
        newly_absorbed = active & ((X == 0) | (X == N))
        absorbed_at[newly_absorbed] = t
        active[newly_absorbed] = False

        trajectories[t] = X

    return trajectories, absorbed_at


def run_until_absorbed(N, M, p0=0.5, max_generations=100_000):
    """
    Like run(), but continues until every population has absorbed
    or max_generations is reached. Only returns final states and
    absorption times, not the full trajectories.
    """

    X = np.full(M, int(round(p0 * N)), dtype=int)
    absorbed_at = np.full(M, np.nan)
    active = np.ones(M, dtype=bool)

    for t in range(1, max_generations + 1):

        if not active.any():
            break

        p = X[active] / N
        X[active] = np.random.binomial(N, p)

        newly_absorbed = active & ((X == 0) | (X == N))
        absorbed_at[newly_absorbed] = t
        active[newly_absorbed] = False

    return X, absorbed_at
