import numpy as np

from densities import DENSITIES
from criterion import select_bins, optimal_D, l1_distance


def run_simulation(n, k_reps=50, seed=0):
    rng = np.random.default_rng(seed)
    results = {}

    for name, sampler, pdf_true in DENSITIES:
        D_oracle_list = []
        D_hat_list = []
        R_list = []
        l1_hat_list = []
        l1_opt_list = []

        for _ in range(k_reps):
            data = sampler(n, rng)
            D_hat = select_bins(data)
            D_oracle = optimal_D(pdf_true, data)

            l1_hat = l1_distance(pdf_true, D_hat, data)
            l1_opt = l1_distance(pdf_true, D_oracle, data)
            R = l1_hat / l1_opt if l1_opt > 1e-6 else np.nan

            D_oracle_list.append(D_oracle)
            D_hat_list.append(D_hat)
            R_list.append(R)
            l1_hat_list.append(l1_hat)
            l1_opt_list.append(l1_opt)

        R_arr = np.array(R_list)
        n_degenerate = int(np.isnan(R_arr).sum())
        R_valid = R_arr[~np.isnan(R_arr)]

        if len(R_valid) > 0:
            R_med = float(np.median(R_valid))
            R_q25, R_q75 = np.percentile(R_valid, [25, 75])
            R_bar = float(R_valid.mean())
        else:
            R_med = R_q25 = R_q75 = R_bar = float("nan")

        D_oracle_modal = int(np.bincount(D_oracle_list).argmax())
        D_hat_modal = int(np.bincount(D_hat_list).argmax())

        results[name] = {
            "D_oracle": D_oracle_modal,
            "D_hat": D_hat_modal,
            "R_med": R_med,
            "R_iqr": float(R_q75 - R_q25),
            "R_bar": R_bar,
            "R_all": R_arr,
            "D_hat_all": np.array(D_hat_list),
            "D_oracle_all": np.array(D_oracle_list),
            "l1_hat_all": np.array(l1_hat_list),
            "l1_opt_all": np.array(l1_opt_list),
            "n_degenerate": n_degenerate,
        }
        suffix = f"  ({n_degenerate} degenerate)" if n_degenerate else ""
        print(f"  {name}  D_oracle={D_oracle_modal:2d}  D_hat={D_hat_modal:2d}  "
              f"R_med={R_med:.3f}  IQR={R_q75 - R_q25:.3f}  R_bar={R_bar:.3f}{suffix}")

    return results


def print_table(results_by_n, n_values):
    col = "  D_oracle  D_hat   R_med    IQR   R_bar"
    header = f"{'':8s}" + "".join(f"{'n=' + str(n):>{len(col)}s}" for n in n_values)
    subheader = f"{'density':8s}" + col * len(n_values)
    print(header)
    print(subheader)
    for name, _, _ in DENSITIES:
        row = f"{name:8s}"
        for n_val in n_values:
            r = results_by_n[n_val][name]
            row += (f"  {r['D_oracle']:8d}  {r['D_hat']:5d}"
                    f"  {r['R_med']:.3f}  {r['R_iqr']:.3f}  {r['R_bar']:.3f}")
        print(row)