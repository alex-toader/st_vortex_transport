"""
Route 20: Large-R continuum test — does σ_tr → R² at R >> a?

R=15 at α=0.3, L=160, r_m=40. The critical test for κ ~ O(1) scaling.

Step 1 results (2 k-pts, fork(2), 8.9 min):

  σ_tr comparison: R=3,5,7 (L=80) vs R=15 (L=160)
    k      R=3      R=5      R=7     R=15
   0.30    13.16    40.74    72.70   264.67
   0.50     6.43    14.16    22.23    80.07

  Power law fit σ_tr ∝ R^p:
    k=0.3: R^1.84 (all R), R^1.70 (R≥5)
    k=0.5: R^1.56 (all R), R^1.59 (R≥5)

  Sub-geometric scaling persists at R=15. Exponent does NOT increase
  toward 2.0 with larger R — stable at ~1.6-1.7 for R≥5.
  σ_tr ~ R² scaling argument does not hold even at R=15.

  Caveat: r_m/R = 2.67 (< 3, mild near-field). Would need r_m=50,
  L=180 for clean far-field at R=15. Near-field bias is ~5-10%,
  not enough to change R^1.7 into R^2.0.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/20_large_R.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from parallel_fdtd import compute_references, compute_scattering

K1, K2 = 1.0, 0.5
L = 160
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 40
R = 15
ALPHA = 0.3
N_POL = 2
N_WORKERS = 2

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

# Step 1: minimal test — 2 k values
k_vals = np.array([0.3, 0.5])

# Reference σ_tr from file 18 (L=80, r_m=20)
ref_data = {
    0.3: {3: 13.16, 5: 40.74, 7: 72.70},
    0.5: {3: 6.43,  5: 14.16, 7: 22.23},
}

if __name__ == '__main__':
    t0 = time.time()

    print(f"Route 20: R={R}, α={ALPHA}, L={L}, r_m={r_m}")
    print(f"  r_m/R = {r_m/R:.1f} (need ≥ 3 for far-field)")
    print(f"  k-grid: {k_vals}")
    print(f"  Workers: {N_WORKERS}")
    print()

    # References
    print(f"Computing {len(k_vals)} references...")
    t1 = time.time()
    refs = compute_references(k_vals, L, DW, DS, DT, sx, r_m,
                              thetas, phis, K1, K2, n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")

    # Scattering
    print(f"Scattering R={R}...")
    t1 = time.time()
    sigma_tr = compute_scattering(k_vals, refs, ALPHA, R, L, DW, DS, DT,
                                  r_m, thetas, phis, K1, K2,
                                  n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")
    print()

    # Compare with R=3,5,7 from file 18
    print("=" * 60)
    print("σ_tr comparison: R=3,5,7 (L=80) vs R=15 (L=160)")
    print("=" * 60)
    print(f"{'k':>5s}  {'R=3':>7s}  {'R=5':>7s}  {'R=7':>7s}  {'R=15':>7s}  "
          f"{'R²pred':>7s}  {'ratio':>6s}")
    print("-" * 55)
    for i, k0 in enumerate(k_vals):
        r3 = ref_data[k0][3]
        r5 = ref_data[k0][5]
        r7 = ref_data[k0][7]
        r15 = sigma_tr[i]
        r2_pred = r5 * (R / 5) ** 2  # geometric prediction from R=5
        ratio = r15 / r2_pred
        print(f"{k0:5.2f}  {r3:7.2f}  {r5:7.2f}  {r7:7.2f}  {r15:7.2f}  "
              f"{r2_pred:7.1f}  {ratio:6.3f}")

    # Fit σ_tr ∝ R^p across R=3,5,7,15
    print()
    print("Power law fit σ_tr ∝ R^p:")
    R_all = np.array([3, 5, 7, 15])
    for i, k0 in enumerate(k_vals):
        s_all = np.array([ref_data[k0][3], ref_data[k0][5],
                          ref_data[k0][7], sigma_tr[i]])
        p, _ = np.polyfit(np.log(R_all), np.log(s_all), 1)
        # Also fit R=5,7,15 only (exclude R=3 lattice regime)
        R_big = np.array([5, 7, 15])
        s_big = np.array([ref_data[k0][5], ref_data[k0][7], sigma_tr[i]])
        p_big, _ = np.polyfit(np.log(R_big), np.log(s_big), 1)
        print(f"  k={k0}: R^{p:.2f} (all), R^{p_big:.2f} (R≥5)")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
