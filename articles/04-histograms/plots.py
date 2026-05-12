import os
import numpy as np
import matplotlib.pyplot as plt

from densities import DENSITIES
from criterion import select_bins, optimal_D, l1_distance


COLORS = {
    "oracle":    "#1C2B4A",
    "hat":       "#4A6FA5",
    "histogram": "#C94040",
    "density":   "#4A6FA5",
    "bg":        "#F0F2F5",
    "navy":      "#1C2B4A",
    "muted":     "#6B7A96",
    "grid":      "#D4DCEA",
}


def _style(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    ax.grid(True, color=COLORS["grid"], lw=0.5)


def plot_density_comparison(name, sampler, pdf_true, n=200, seed=42, out_dir="."):
    rng = np.random.default_rng(seed)
    data = sampler(n, rng)
    D_hat = select_bins(data)
    D_oracle = optimal_D(pdf_true, data)

    xs = np.linspace(0.0, 1.0, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor=COLORS["bg"])
    fig.suptitle(f"{name}  (n={n})", fontsize=12, color=COLORS["navy"],
                 fontfamily="monospace")

    panels = [
        (D_oracle, f"D_oracle = {D_oracle}"),
        (D_hat,    f"D_hat = {D_hat}"),
    ]
    for ax, (D, label) in zip(axes, panels):
        _style(ax)
        ax.hist(data, bins=D, range=(0.0, 1.0), density=True,
                color=COLORS["histogram"], alpha=0.45, label=f"histogram: D={D}")
        ax.plot(xs, pdf_true(xs), color=COLORS["density"], lw=1.8, label="density")
        ax.set_title(label, color=COLORS["navy"], fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f"histogram_{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def plot_R_boxplots(results_by_n, n_values, out_dir="."):
    fig, axes = plt.subplots(1, len(n_values), figsize=(4 * len(n_values), 4),
                             facecolor=COLORS["bg"], sharey=True)
    if len(n_values) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        _style(ax)
        data_per_density = []
        labels = []
        for name, _, _ in DENSITIES:
            R = results_by_n[n][name]["R_all"]
            R = R[~np.isnan(R)]
            data_per_density.append(R)
            labels.append(name)

        bp = ax.boxplot(data_per_density, labels=labels, widths=0.5,
                        patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS["hat"])
            patch.set_alpha(0.4)
            patch.set_edgecolor(COLORS["navy"])
        for med in bp["medians"]:
            med.set_color(COLORS["histogram"])
            med.set_linewidth(1.5)
        for whisker in bp["whiskers"]:
            whisker.set_color(COLORS["muted"])
        for cap in bp["caps"]:
            cap.set_color(COLORS["muted"])
        for flier in bp["fliers"]:
            flier.set_marker("o")
            flier.set_markersize(3)
            flier.set_markerfacecolor(COLORS["muted"])
            flier.set_markeredgecolor(COLORS["muted"])

        ax.axhline(1.0, color=COLORS["oracle"], lw=0.8, linestyle="--", alpha=0.5)
        ax.set_title(f"n = {n}", color=COLORS["navy"], fontsize=10)
        ax.set_ylabel("R = L1(D_hat) / L1(D_oracle)", color=COLORS["muted"], fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "R_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def plot_D_hat_distribution(results_by_n, n_values, out_dir="."):
    fig, axes = plt.subplots(len(DENSITIES), len(n_values),
                             figsize=(3 * len(n_values), 2 * len(DENSITIES)),
                             facecolor=COLORS["bg"], squeeze=False)

    for i, (name, _, _) in enumerate(DENSITIES):
        for j, n in enumerate(n_values):
            ax = axes[i, j]
            _style(ax)
            r = results_by_n[n][name]
            D_hat_all = r["D_hat_all"]
            D_oracle_modal = r["D_oracle"]

            D_max = max(D_hat_all.max(), D_oracle_modal) + 1
            bins = np.arange(0.5, D_max + 1.5, 1)
            ax.hist(D_hat_all, bins=bins, color=COLORS["hat"], alpha=0.6,
                    edgecolor=COLORS["navy"], linewidth=0.5)
            ax.axvline(D_oracle_modal, color=COLORS["histogram"], lw=1.5,
                       linestyle="--", label=f"D_oracle = {D_oracle_modal}")
            ax.set_title(f"{name}, n={n}", color=COLORS["navy"], fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            ax.set_xlabel("D_hat", color=COLORS["muted"], fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "D_hat_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def plot_L1_vs_n(results_by_n, n_values, out_dir="."):
    fig, axes = plt.subplots(1, len(DENSITIES),
                             figsize=(3 * len(DENSITIES), 3.5),
                             facecolor=COLORS["bg"], sharey=False)

    for ax, (name, _, _) in zip(axes, DENSITIES):
        _style(ax)
        l1_hat_medians = []
        l1_oracle_medians = []
        for n in n_values:
            r = results_by_n[n][name]
            l1_hat_medians.append(np.median(r["l1_hat_all"]))
            l1_oracle_medians.append(np.median(r["l1_opt_all"]))

        ax.plot(n_values, l1_hat_medians, marker="o", color=COLORS["hat"],
                label="L1(D_hat)", lw=1.5)
        ax.plot(n_values, l1_oracle_medians, marker="s", color=COLORS["oracle"],
                label="L1(D_oracle)", lw=1.5, linestyle="--")
        ax.set_title(name, color=COLORS["navy"], fontsize=10)
        ax.set_xlabel("n", color=COLORS["muted"], fontsize=9)
        ax.set_xticks(n_values)
        ax.set_ylabel("median L1 error", color=COLORS["muted"], fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "L1_vs_n.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def plot_criterion_diagnostic(name, sampler, n=200, seed=42, out_dir="."):
    rng = np.random.default_rng(seed)
    data = sampler(n, rng)

    d_min, d_max = data.min(), data.max()
    u = (data - d_min) / (d_max - d_min)

    D_max = int(np.floor(n / np.log(n)))
    Ds = np.arange(1, D_max + 1)
    log_liks = []
    pens = []
    for D in Ds:
        counts, _ = np.histogram(u, bins=D, range=(0.0, 1.0))
        nz = counts[counts > 0]
        log_liks.append(np.sum(nz * np.log(D * nz / n)))
        pens.append((D - 1) + (np.log(D) ** 2.5 if D > 1 else 0.0))
    log_liks = np.array(log_liks)
    pens = np.array(pens)
    crit = log_liks - pens
    D_hat = Ds[np.argmax(crit)]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=COLORS["bg"])
    _style(ax)
    ax.plot(Ds, log_liks, marker="o", color=COLORS["hat"], label="L(D)", lw=1.5)
    ax.plot(Ds, pens, marker="s", color=COLORS["muted"], label="pen(D)", lw=1.5)
    ax.plot(Ds, crit, marker="^", color=COLORS["histogram"],
            label="L(D) - pen(D)", lw=1.8)
    ax.axvline(D_hat, color=COLORS["oracle"], lw=0.8, linestyle="--", alpha=0.6)
    ax.annotate(f"D_hat = {D_hat}", xy=(D_hat, crit.max()),
                xytext=(5, 0), textcoords="offset points",
                fontsize=9, color=COLORS["navy"])
    ax.set_title(f"criterion vs D, {name} (n={n})",
                 color=COLORS["navy"], fontsize=10)
    ax.set_xlabel("D", color=COLORS["muted"], fontsize=9)
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f"criterion_{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")