# Birgé-Rozenholc Histogram Selection

This repository implements the penalized log-likelihood rule for histogram bin
selection as described by Birgé and Rozenholc (2006). This is a modern Python
implementation used to update my 2018 research work at the Université Nice
Sophia Antipolis.
The original work was submitted as a statistics assignment in December 2018 at the Université de Nice Sophia Antipolis. The full manuscript is included in this repository. The code has been restructured and extended in 2026 as part of as part of [The Invariant](https://theinvariantlab.github.io/the-invariant).

## The rule

For data on `[0, 1]`, choose `D` to maximize

```
L(D) - pen(D)
```

where

```
L(D)   = sum_k N_k log(D N_k / n)
pen(D) = (D - 1) + log(D)^2.5
```

over `1 <= D <= n / log(n)`. `N_k` is the count in bin `k`.

## Run

```
pip install -r requirements.txt
python histogram_bins.py
```

Runs the five test densities at `n = 50, 100, 200` with 50 repetitions each,
prints the summary table, and writes one plot per density to `outputs/`.

## Reference

Birgé, L. and Rozenholc, Y. (2006). How many bins should be put in a regular
histogram. *ESAIM: Probability and Statistics*, 10, 24-45.