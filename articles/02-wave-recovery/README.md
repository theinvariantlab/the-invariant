# Wave Recovery

Code and data for the article [The seafloor knows the shape of the wave](https://theinvariantlab.substack.com).

## Files

- `wave_recovery.py` — runs the full recovery and plots results
- `pressure_data.csv` — 1024 seabed pressure measurements (normalised units, from Clamond 2018)

## Usage

```bash
pip install numpy scipy matplotlib
python wave_recovery.py
```

The script sweeps Fourier orders N = 5, 10, 20, 40, 100 and saves `error_vs_N.png` and `reconstructions.png`.
