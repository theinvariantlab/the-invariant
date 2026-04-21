"""
wave_recovery.py
----------------
Numerical recovery of surface wave profiles from seabed pressure measurements.

Based on:
  - Clamond & Constantin, "Recovery of steady periodic wave profiles from
    pressure measurements at the bed", 2012.
  - Clamond, "New exact relations for easy recovery of steady wave profiles
    from bottom pressure measurements", 2013.

The key result: reconstruction error is non-monotonic in N, the Fourier
truncation order. Error improves up to an optimal N, then diverges as
high-frequency terms amplify noise through the exponential factor exp(nkd).

Data:
  pressure_data.csv -- one pressure value per line, evenly spaced in x.
  Original data was in finite_HD_040_LD_0501.mat (Clamond, 2018).
"""

import warnings
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# normalised units: g = d = 1
# to use dimensional values, set G and D accordingly -- see thesis §1
G = 1.0
D = 1.0


# --- data ---

def load_pressure_data(filepath, dx):
    """
    Load seabed pressure measurements from a CSV file.

    Each row is one pressure value at the bed, evenly spaced by dx.
    dx is the spatial step size (normalised units).
    """
    pb = np.loadtxt(filepath)
    assert pb.ndim == 1, f"Expected 1D pressure array, got shape {pb.shape}"
    return pb, float(dx)


# --- spectral ---

def compute_fourier_coefficients(pb):
    """
    Compute Fourier coefficients pn of the bed pressure via FFT.

    p(x) ~= gd + sum_{|n|>0} pn exp(inkx)    (Eq. 6.1)
    """
    return np.fft.fft(pb) / len(pb)


def compute_beta(z, pn, k, N):
    """
    Evaluate the holomorphic function B(z) at a point z in the fluid domain.

    B(z) = gd + sum_{|n|>0} pn exp(inkz) / exp(nkd)    (Eq. 6.2)

    The sum runs over both positive and negative n. For negative n,
    exp(nkd) = exp(-|n|kd), so the denominator shrinks rather than grows --
    negative frequencies amplify rather than attenuate. At large N this
    asymmetry drives reconstruction instability.
    """
    ns = np.arange(1, N + 1)
    N_total = len(pn)
    pos = np.sum(pn[ns] * np.exp(1j * ns * k * z) / np.exp(ns * k * D))
    neg = np.sum(pn[N_total - ns] * np.exp(-1j * ns * k * z) / np.exp(-ns * k * D))
    return G * D + pos + neg


def compute_omega(z, pn, k, N, a):
    """
    Evaluate the holomorphic function D(z) at a point z.

    D(z) = sum_{|n|>0} pn/(ink) * [exp(inkz) - exp(-nka)] / exp(nkd)  (Eq. 6.3)

    Like B(z), the sum runs over both positive and negative n. The negative
    frequency terms use exp(-nkd) in the denominator, which grows with n
    and is the source of instability at high N.
    """
    ns = np.arange(1, N + 1)
    N_total = len(pn)
    pos = np.sum(
        pn[ns] / (1j * ns * k)
        * (np.exp(1j * ns * k * z) - np.exp(-ns * k * a))
        / np.exp(ns * k * D)
    )
    neg = np.sum(
        pn[N_total - ns] / (-1j * ns * k)
        * (np.exp(-1j * ns * k * z) - np.exp(ns * k * a))
        / np.exp(-ns * k * D)
    )
    return pos + neg


# --- recovery ---

def solve_wave_amplitudes(pn, k, N):
    """
    Solve the two implicit equations for crest amplitude a and trough depth b.

        a = g^{-1} [Re(B(ia)) - d]          (Eq. 5.13)
        b = d - g^{-1} Re(B(pi/k - ib))     (Eq. 5.14)

    Returns a, b, and a convergence flag.
    Residuals are checked explicitly -- at high N the solver can return
    a result that looks plausible but has drifted.
    """
    tol = 1e-10

    def eq_crest(a):
        return a + D - compute_beta(1j * a[0], pn, k, N).real / G

    def eq_trough(b):
        return b - D + compute_beta(np.pi / k - 1j * b[0], pn, k, N).real / G

    a_sol, info_a, ier_a, _ = fsolve(eq_crest, [0.3], full_output=True)
    b_sol, info_b, ier_b, _ = fsolve(eq_trough, [0.2], full_output=True)

    residual_a = np.abs(info_a["fvec"]).max()
    residual_b = np.abs(info_b["fvec"]).max()

    converged = (
        ier_a == 1 and ier_b == 1
        and residual_a < tol
        and residual_b < tol
    )

    if not converged:
        warnings.warn(
            f"N={N}: solver did not converge cleanly. "
            f"Residuals: a={residual_a:.2e}, b={residual_b:.2e}"
        )

    return float(a_sol[0]), float(b_sol[0]), converged


def compute_bernoulli_constant(a, b, pn, k, N):
    """
    Compute the Bernoulli constant B from wave amplitudes a and b.

    B = (1/2)g(a - b) - (a + b)^{-1} Im(D(pi/k - ib))    (Eq. 5.15)
    """
    omega_val = compute_omega(np.pi / k - 1j * b, pn, k, N, a)
    return 0.5 * G * (a - b) - omega_val.imag / (a + b)


def eta_equation(eta, x, pn, k, N, a, B):
    """
    Implicit equation for surface elevation at a single point x.

    eta*g - B + sqrt((B - ga)^2 - 2g*Im(D(x + i*eta))) = 0    (Eq. 6.4)

    eta appears inside D, so this must be solved pointwise with fsolve.
    """
    omega_s = compute_omega(x + 1j * eta, pn, k, N, a)
    return eta * G - B + np.sqrt((B - G * a) ** 2 - 2 * G * omega_s.imag)


def recover_surface(pn, k, N, dx):
    """
    Recover the surface elevation eta at each measurement point.

    Solves the implicit equation pointwise, starting from eta_initial = 0.1.
    Returns eta, delta_a, delta_b -- errors on crest and trough amplitudes.
    """
    a, b, converged = solve_wave_amplitudes(pn, k, N)
    B = compute_bernoulli_constant(a, b, pn, k, N)

    n_points = len(pn)
    eta = np.empty(n_points)

    for i in range(n_points):
        x = i * dx
        eta[i] = fsolve(eta_equation, 0.1, args=(x, pn, k, N, a, B))[0]

    delta_a = abs(a - np.nanmax(eta))
    delta_b = abs(b - abs(np.nanmin(eta)))

    return eta, delta_a, delta_b


def sweep_fourier_order(pb, dx, orders):
    """
    Run surface recovery for each value of N and collect error metrics.

    This is the central experiment: reconstruction error is non-monotonic.
    It improves as N increases up to an optimal order, then diverges as
    exp(nkd) amplifies noise in the high-frequency Fourier coefficients.
    """
    N_total = len(pb)
    k = 2 * np.pi / (N_total * dx)
    pn = compute_fourier_coefficients(pb)

    results = {"N": [], "delta_a": [], "delta_b": [], "eta": []}

    for N in orders:
        if N >= N_total // 2:
            warnings.warn(f"N={N} exceeds Nyquist limit for {N_total} points; skipping.")
            continue

        eta, delta_a, delta_b = recover_surface(pn, k, N, dx)
        results["N"].append(N)
        results["delta_a"].append(delta_a)
        results["delta_b"].append(delta_b)
        results["eta"].append(eta)

        print(f"N={N:>4d}   delta_a = {delta_a:.2e}   delta_b = {delta_b:.2e}")

    return results


# --- plotting ---

def plot_error_curve(results):
    """Plot reconstruction error vs Fourier order N."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(results["N"], results["delta_a"], "o-", color="#4A6FA5", label="delta_a (crest)")
    ax.semilogy(results["N"], results["delta_b"], "s--", color="#C94040", label="delta_b (trough)")
    ax.set_xlabel("Fourier order N", fontfamily="monospace")
    ax.set_ylabel("Reconstruction error", fontfamily="monospace")
    ax.set_title("Error is non-monotonic in N", fontfamily="monospace")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig("error_vs_N.png", dpi=150)
    plt.show()


def plot_reconstructions(results, dx):
    """Plot recovered surface profiles for each N."""
    n_plots = len(results["N"])
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    for ax, N, eta, da, db in zip(
        axes, results["N"], results["eta"], results["delta_a"], results["delta_b"]
    ):
        xs = np.arange(len(eta)) * dx
        ax.plot(xs, eta, color="#4A6FA5", label="wave approximation")
        ax.set_ylabel("eta", fontfamily="monospace")
        ax.set_title(f"N={N}   da={da:.2e}   db={db:.2e}", fontfamily="monospace", fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("x", fontfamily="monospace")
    fig.suptitle("Surface recovery at increasing Fourier order", fontfamily="monospace")
    fig.tight_layout()
    plt.savefig("reconstructions.png", dpi=150)
    plt.show()


# --- run ---

if __name__ == "__main__":
    # pressure_data.csv: one normalised pressure value per line, evenly spaced.
    # Original source: finite_HD_040_LD_0501.mat (column 26), Clamond 2018.
    DATA_FILE = "pressure_data.csv"
    DX = 0.00488   # spatial step (normalised units)

    pb, dx = load_pressure_data(DATA_FILE, DX)

    # orders chosen to span the stable -> unstable transition
    FOURIER_ORDERS = [5, 10, 20, 40, 100]

    print("Sweeping Fourier order N...\n")
    results = sweep_fourier_order(pb, dx, FOURIER_ORDERS)

    plot_error_curve(results)
    plot_reconstructions(results, dx)
