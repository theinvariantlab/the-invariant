import os
import numpy as np
import matplotlib.pyplot as plt

from ransac_data_gen import generate, TRUE_SLOPE, TRUE_INTERCEPT
from ransac_solvers import fit_ols, fit_ransac

PALETTE = {
    "bg":     "#F0F2F5",
    "navy":   "#1C2B4A",
    "steel":  "#4A6FA5",
    "muted":  "#6B7A96",
    "border": "#D4DCEA",
    "red":    "#C94040",
    "white":  "#FFFFFF",
}

SEED = 42
N_POINTS = 100
INLIER_FRAC = 0.65
OUTPUT_PATH = "outputs/comparison.png"


def run_comparison(seed=SEED):
    pts, true_mask = generate(N=N_POINTS, inlier_frac=INLIER_FRAC, seed=seed)
    ols_slope, ols_intercept = fit_ols(pts)
    ransac_slope, ransac_intercept, n_iters, ransac_mask = fit_ransac(pts, seed=seed)
    return dict(
        pts=pts,
        true_mask=true_mask,
        ols_slope=ols_slope,
        ols_intercept=ols_intercept,
        ransac_slope=ransac_slope,
        ransac_intercept=ransac_intercept,
        ransac_mask=ransac_mask,
        n_iters=n_iters,
    )


def print_results(r):
    n_in = r["true_mask"].sum()
    n_out = (~r["true_mask"]).sum()
    print("=" * 52)
    print(f"  N={N_POINTS}  |  wall contacts={n_in}  |  noise={n_out}")
    print("=" * 52)
    print(f"  True     slope={TRUE_SLOPE:.3f}   intercept={TRUE_INTERCEPT:.3f}")
    print(f"  OLS      slope={r['ols_slope']:.3f}   intercept={r['ols_intercept']:.3f}")
    print(f"  RANSAC   slope={r['ransac_slope']:.3f}   intercept={r['ransac_intercept']:.3f}"
          f"   ({r['n_iters']} iters, {r['ransac_mask'].sum()} consensus)")
    print("=" * 52)


def plot_results(r, output_path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pts = r["pts"]
    mask = r["true_mask"]
    xs = np.linspace(0, 100, 300)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["white"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.grid(True, color=PALETTE["border"], linewidth=0.6, zorder=0)

    ax.scatter(pts[~mask, 0], pts[~mask, 1],
               c=PALETTE["red"], s=18, alpha=0.55, zorder=2, label="sensor noise")
    ax.scatter(pts[mask, 0], pts[mask, 1],
               c=PALETTE["steel"], s=18, alpha=0.75, zorder=2, label="wall contact")

    ax.plot(xs, TRUE_SLOPE * xs + TRUE_INTERCEPT,
            color=PALETTE["red"], linewidth=1.5, linestyle="--", zorder=3, label="true wall")
    ax.plot(xs, r["ols_slope"] * xs + r["ols_intercept"],
            color=PALETTE["muted"], linewidth=2.0, zorder=4,
            label=f"OLS  (slope={r['ols_slope']:.2f})")
    ax.plot(xs, r["ransac_slope"] * xs + r["ransac_intercept"],
            color=PALETTE["navy"], linewidth=2.0, zorder=5,
            label=f"RANSAC  (slope={r['ransac_slope']:.2f})")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("x", fontsize=11, color=PALETTE["muted"])
    ax.set_ylabel("y", fontsize=11, color=PALETTE["muted"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)

    leg = ax.legend(frameon=True, fontsize=9, loc="upper left")
    leg.get_frame().set_edgecolor(PALETTE["border"])
    leg.get_frame().set_facecolor(PALETTE["white"])

    ax.set_title(
        f"OLS vs RANSAC — N={N_POINTS}, {mask.sum()} wall contacts ({100*mask.mean():.0f}%)",
        fontsize=12, color=PALETTE["navy"], pad=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"\n  -> {output_path}")
    plt.close()


def main():
    r = run_comparison()
    print_results(r)
    plot_results(r)


if __name__ == "__main__":
    main()
