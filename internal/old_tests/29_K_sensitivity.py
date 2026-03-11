"""
Route 29 (R1): K1/K2 sensitivity — are spectral results fine-tuned to K1=2K2?

Tests σ_tr at K1/K2 = 1.5, 2.0 (default), 3.0 to verify:
  1. Flat integrand (sin²·σ_tr ≈ const) persists at other K1/K2
  2. κ = O(1) is not a fine-tuning accident
  3. R^{3/2}-like scaling is robust

Uses 4 k-points (k=0.3, 0.7, 1.1, 1.5) at α=0.3, R=5, NN gauging.
K2 fixed at 0.5; K1 varies: 0.75, 1.0, 1.5. This gives K1/K2 = 1.5, 2.0, 3.0.

NOTE: K1=2K2 is the isotropy condition for this cubic lattice. At K1≠2K2,
the group velocity is direction-dependent (anisotropy). σ_tr measured at
x-incidence only — not angularly averaged. Comparison is between K1/K2
values, not absolute. κ formula uses sin²(k) which is independent of
K1/K2 — no c-rescaling needed.

Results (α=0.3, R=5, L=80, 4 k-pts, 115s):

  σ_tr(k) for each K1/K2:
    k     K1/K2=1.5  K1/K2=2.0  K1/K2=3.0
   0.30     35.11      40.74      47.28
   0.70      6.08       7.69       9.89
   1.10      2.88       3.78       5.11
   1.50      2.13       2.81       3.85

  Summary:
    K1/K2   c      aniso   CV(integrand)  κ(k≤1.5)
    1.5    1.658    9%       14.2%          0.751   FLAT
    2.0    1.732    0%        9.0%          0.949   FLAT
    3.0    1.871   14%        2.9%          1.231   FLAT

  CONCLUSION:
    Flat integrand persists at all K1/K2 tested (CV = 3-14%).
    Flatness IMPROVES at larger K1/K2 (more NNN dispersion relative to NN).
    κ varies 0.75-1.23 across K1/K2 = 1.5-3.0. Not fine-tuned.
    κ = O(1) is robust across spring constant ratios.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/29_K_sensitivity.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../src/1_foam'))
from parallel_fdtd import compute_references, compute_scattering

L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 5
ALPHA = 0.3
N_POL = 2
N_WORKERS = 4

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

k_vals = np.array([0.3, 0.7, 1.1, 1.5])

# K2 fixed, K1 varies
K2_fixed = 0.5
K1_vals = [0.75, 1.0, 1.5]  # K1/K2 = 1.5, 2.0, 3.0

if __name__ == '__main__':
    t0 = time.time()

    print("Route 29 (R1): K1/K2 sensitivity")
    print(f"  α={ALPHA}, R={R_LOOP}, L={L}")
    print(f"  K2={K2_fixed}, K1 = {K1_vals}")
    print(f"  K1/K2 = {[K1/K2_fixed for K1 in K1_vals]}")
    print(f"  k-grid: {list(k_vals)}")
    print()

    all_results = []
    for K1 in K1_vals:
        c = np.sqrt(K1 + 4 * K2_fixed)
        aniso = abs(K1 - 2 * K2_fixed) / (K1 + 4 * K2_fixed) * 100
        print(f"K1={K1}, K1/K2={K1/K2_fixed:.1f}, c={c:.3f}, anisotropy~{aniso:.0f}%:")

        # References depend on K1/K2 — must recompute
        t1 = time.time()
        refs = compute_references(k_vals, L, DW, DS, DT, sx, r_m,
                                  thetas, phis, K1, K2_fixed, n_workers=N_WORKERS)
        sigma_tr = compute_scattering(k_vals, refs, ALPHA, R_LOOP, L, DW, DS, DT,
                                      r_m, thetas, phis, K1, K2_fixed,
                                      n_workers=N_WORKERS)
        dt = time.time() - t1

        integrand = np.sin(k_vals)**2 * sigma_tr
        cv = np.std(integrand) / np.mean(integrand) * 100

        # κ (trapezoidal, k<=1.5)
        prefactor = N_POL * R_LOOP / (4 * np.pi**2)
        kappa = prefactor * np.trapz(integrand, k_vals)

        all_results.append({
            'K1': K1, 'ratio': K1/K2_fixed, 'c': c,
            'sigma_tr': sigma_tr.copy(), 'integrand': integrand.copy(),
            'cv': cv, 'kappa': kappa, 'dt': dt,
        })
        print(f"  done ({dt:.0f}s), CV={cv:.1f}%, κ={kappa:.3f}")

    # σ_tr table
    print()
    print("=" * 60)
    print("σ_tr(k) for each K1/K2")
    print("=" * 60)
    header = "      k"
    for r in all_results:
        label = "K1/K2=" + str(r['ratio'])
        header += f"  {label:>12s}"
    print(header)
    print(f"  {'-'*45}")
    for j, k0 in enumerate(k_vals):
        line = f"  {k0:5.2f}"
        for r in all_results:
            line += f"  {r['sigma_tr'][j]:12.3f}"
        print(line)

    # Integrand table
    print()
    print("sin²(k)·σ_tr (integrand)")
    header = "      k"
    for r in all_results:
        label = "K1/K2=" + str(r['ratio'])
        header += f"  {label:>12s}"
    print(header)
    print(f"  {'-'*45}")
    for j, k0 in enumerate(k_vals):
        line = f"  {k0:5.2f}"
        for r in all_results:
            line += f"  {r['integrand'][j]:12.4f}"
        print(line)

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  {'K1/K2':>6s}  {'c':>6s}  {'aniso':>6s}  {'CV':>6s}  {'κ(k≤1.5)':>9s}")
    print(f"  {'-'*45}")
    for r in all_results:
        aniso = abs(r['K1'] - 2 * K2_fixed) / (r['K1'] + 4 * K2_fixed) * 100
        flat = "FLAT" if r['cv'] < 15 else ""
        print(f"  {r['ratio']:6.1f}  {r['c']:6.3f}  {aniso:5.0f}%  {r['cv']:5.1f}%  {r['kappa']:9.3f}  {flat}")

    print()
    print("  NOTE: 4-pt grid (Δk=0.4). κ values are relative comparisons.")
    print("  For absolute κ, use 7-pt grid (file 22/28).")
    print("  sin²(k) weight is K1/K2-independent — no c-rescaling needed.")

    # Relative to default (K1/K2=2.0)
    ref_idx = next(i for i, r in enumerate(all_results) if abs(r['ratio'] - 2.0) < 0.01)
    ref_kappa = all_results[ref_idx]['kappa']
    print()
    print(f"  κ relative to K1/K2=2.0 ({ref_kappa:.3f}):")
    for r in all_results:
        ratio = r['kappa'] / ref_kappa
        print(f"    K1/K2={r['ratio']:.1f}: κ = {r['kappa']:.3f} ({ratio:.2f}×)")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
