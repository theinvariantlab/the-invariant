# The Invariant -- Genetic Drift
# Empirical verification of appendix results
# github.com/theinvariantlab/the-invariant

"""
Three experiments, each verifying a result from the appendix.

  1. Fixation probability = i/N           (Theorem 3.1, OST)
  2. Symmetry at equal split              (Corollary 3.2)
  3. Conditional fixation time vs p0      (Ewens 3.5, 3.11)

Each prints empirical vs theoretical side by side.
"""

import numpy as np
from drift_simulate import run_until_absorbed

np.random.seed(42)

M = 20_000   # number of independent populations per experiment
N = 100      # population size


# 1. Fixation probability
# -----------------------
# Theorem 3.1: starting from X_0 = i, red fixes with probability exactly i/N.
# We run M populations from each starting point and measure how often red wins.

print("=" * 60)
print("1. Fixation probability  P(red fixes | X_0 = i) = i/N")
print("=" * 60)
print(f"{'i':>6}  {'theory i/N':>12}  {'empirical':>12}  {'error':>10}")
print("-" * 60)

for i in [10, 25, 50, 75, 90]:
    final, _ = run_until_absorbed(N, M, p0=i/N)
    p_fix = np.mean(final == N)
    theory = i / N
    print(f"{i:>6}  {theory:>12.4f}  {p_fix:>12.4f}  {abs(p_fix - theory):>10.4f}")

print()


# 2. Symmetry at equal split
# --------------------------
# Corollary 3.2: at p0 = 0.5, red and blue are equally likely to fix.

print("=" * 60)
print("2. Symmetry at equal split  (Corollary 3.2)")
print("=" * 60)

final, _ = run_until_absorbed(N, M, p0=0.5)
p_red  = np.mean(final == N)
p_blue = np.mean(final == 0)

print(f"  P(red wins)  = {p_red:.4f}   (theory: 0.5000)")
print(f"  P(blue wins) = {p_blue:.4f}   (theory: 0.5000)")
print()


# 3. Conditional fixation time
# ----------------------------
# Ewens (3.5): unconditional mean absorption time is approximately
#              -N [p ln p + (1-p) ln(1-p)]
# Ewens (3.11): conditional fixation time is longer for rare starting frequencies.
#
# We measure E[T | red fixes] for several starting frequencies p0
# and compare against the unconditional formula as a reference.

print("=" * 60)
print("3. Conditional fixation time vs starting frequency")
print("   Reference: Ewens (3.5) unconditional E[T]")
print("=" * 60)
print(f"{'p0':>6}  {'E[T] Ewens 3.5':>16}  {'E[T|red fixes]':>16}  {'n fixed':>8}")
print("-" * 60)

def ewens_35(N, p):
    # Unconditional mean absorption time, diffusion approximation (Ewens 3.5)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -N * (p * np.log(p) + (1 - p) * np.log(1 - p))

for p0 in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    final, times = run_until_absorbed(N, M, p0=p0)
    fixed = final == N
    n_fixed = fixed.sum()
    if n_fixed == 0:
        print(f"{p0:>6.1f}  {'n/a':>16}  {'n/a':>16}  {0:>8}")
        continue
    mean_fix_time = np.nanmean(times[fixed])
    theory = ewens_35(N, p0)
    print(f"{p0:>6.1f}  {theory:>16.1f}  {mean_fix_time:>16.1f}  {n_fixed:>8}")

print()
print("The conditional fixation time is longer for rare alleles,")
print("shorter for common ones. The underdog, if it wins, wins slowly.")
print("The unconditional E[T] is lower because it averages over quick losses.")
