import os

from densities import DENSITIES
from simulation import run_simulation, print_table
from plots import (
    plot_density_comparison,
    plot_R_boxplots,
    plot_D_hat_distribution,
    plot_L1_vs_n,
    plot_criterion_diagnostic,
)


K_REPS = 50
N_VALUES = [50, 100, 200]
PLOT_N = 200
SEED = 0
OUT_DIR = "outputs"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results_by_n = {}
    for n in N_VALUES:
        print(f"\nn = {n}")
        results_by_n[n] = run_simulation(n, k_reps=K_REPS, seed=SEED)

    print("\nSummary")
    print_table(results_by_n, N_VALUES)

    print("\nPlots")
    for name, sampler, pdf_true in DENSITIES:
        plot_density_comparison(name, sampler, pdf_true, n=PLOT_N,
                                seed=SEED, out_dir=OUT_DIR)
        plot_criterion_diagnostic(name, sampler, n=PLOT_N,
                                  seed=SEED, out_dir=OUT_DIR)
    plot_R_boxplots(results_by_n, N_VALUES, out_dir=OUT_DIR)
    plot_D_hat_distribution(results_by_n, N_VALUES, out_dir=OUT_DIR)
    plot_L1_vs_n(results_by_n, N_VALUES, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()