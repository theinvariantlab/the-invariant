# The Invariant -- Genetic Drift
# Figures for the appendix
# github.com/theinvariantlab/the-invariant

"""
Three figures, each corresponding to a result in the appendix.

  Fig 1 -- P(red fixes) vs starting frequency       (Theorem 3.1)
  Fig 2 -- Fraction still segregating over time      (Corollary 2.2)
  Fig 3 -- Conditional fixation time vs p0           (Ewens 3.5, 3.11)
"""

import numpy as np
import matplotlib.pyplot as plt
from drift_simulate import run, run_until_absorbed

# Visual identity from The Invariant
NAVY  = "#1C2B4A"
BLUE  = "#4A6FA5"
PALE  = "#DCE8F5"
GREY  = "#6B7A96"
RED   = "#C94040"
BG    = "#F7F9FC"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    GREY,
    "axes.labelcolor":   NAVY,
    "xtick.color":       GREY,
    "ytick.color":       GREY,
    "text.color":        NAVY,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        PALE,
    "grid.linewidth":    0.7,
})

np.random.seed(42)
M = 10_000
N = 100


# Fig 1: Fixation probability vs starting frequency
# -------------------------------------------------
# Each point is the empirical fraction of populations where red fixed,
# starting from a given frequency. Should lie on the diagonal y = x.

starting = np.arange(1, N) / N
p_fix = np.empty(len(starting))

for k, p0 in enumerate(starting):
    final, _ = run_until_absorbed(N, M // 5, p0=p0)
    p_fix[k] = np.mean(final == N)

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(starting, starting, color=GREY, lw=1.2, ls="--", label="i/N  (Theorem 3.1)")
ax1.scatter(starting, p_fix, s=12, color=BLUE, alpha=0.7, label="simulation")
ax1.set_xlabel("Starting frequency of red  p0 = i/N")
ax1.set_ylabel("P(red fixes)")
ax1.set_title("Fixation probability of red  --  Theorem 3.1", color=NAVY, fontsize=11)
ax1.legend(frameon=False, fontsize=9)
fig1.tight_layout()
fig1.savefig("fig1_fixation_probability.png", dpi=150)
print("Saved fig1_fixation_probability.png")


# Fig 2: Fraction still segregating over time
# -------------------------------------------
# Shows that absorption is certain (Corollary 2.2) and that
# larger populations take longer to absorb.

gens = 800

fig2, ax2 = plt.subplots(figsize=(7, 4.5))

for N_val, color in [(20, RED), (50, BLUE), (100, NAVY)]:
    _, absorbed_at = run_until_absorbed(N_val, M, p0=0.5, max_generations=gens)

    # At each generation t, count how many populations have not yet absorbed
    still_segregating = np.array([
        np.mean(np.isnan(absorbed_at) | (absorbed_at > t))
        for t in range(gens + 1)
    ])
    ax2.plot(range(gens + 1), still_segregating,
             color=color, lw=1.5, label=f"N = {N_val}")

ax2.set_xlabel("Generation")
ax2.set_ylabel("Fraction still segregating")
ax2.set_title("Absorption is certain  --  Corollary 2.2", color=NAVY, fontsize=11)
ax2.set_ylim(-0.02, 1.05)
ax2.legend(frameon=False, fontsize=9)
fig2.tight_layout()
fig2.savefig("fig2_absorption_curve.png", dpi=150)
print("Saved fig2_absorption_curve.png")


# Fig 3: Does the underdog take longer to win?
# --------------------------------------------
# E[T | red fixes] as a function of starting frequency p0.
# A rare allele that fixes takes much longer than a common one.
# The dashed curve is the unconditional E[T] from Ewens (3.5) as a reference.

def ewens_35(N, p):
    # Unconditional mean absorption time, diffusion approximation (Ewens 3.5)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -N * (p * np.log(p) + (1 - p) * np.log(1 - p))

p0_values = np.linspace(0.05, 0.95, 19)
emp_fix   = []

for p0 in p0_values:
    final, times = run_until_absorbed(N, M, p0=p0)
    fixed = final == N
    emp_fix.append(np.nanmean(times[fixed]) if fixed.any() else np.nan)

theory_unconditional = [ewens_35(N, p) for p in p0_values]

fig3, ax3 = plt.subplots(figsize=(7, 4.5))
ax3.plot(p0_values, theory_unconditional, color=GREY, lw=1.2, ls="--",
         label="E[T] unconditional  (Ewens 3.5)")
ax3.plot(p0_values, emp_fix, color=RED, lw=1.8, marker="o", ms=4,
         label="E[T | red fixes]  (simulation)")
ax3.set_xlabel("Starting frequency of red  p0")
ax3.set_ylabel("Generations until fixation")
ax3.set_title("The underdog pays in time", color=NAVY, fontsize=11)
ax3.legend(frameon=False, fontsize=9)

ax3.annotate("rare allele, long wait",
             xy=(0.1, emp_fix[1]), xytext=(0.18, emp_fix[1] + 15),
             color=GREY, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
ax3.annotate("common allele, short wait",
             xy=(0.9, emp_fix[-2]), xytext=(0.65, emp_fix[-2] + 20),
             color=GREY, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))

fig3.tight_layout()
fig3.savefig("fig3_underdog_time.png", dpi=150)
print("Saved fig3_underdog_time.png")
