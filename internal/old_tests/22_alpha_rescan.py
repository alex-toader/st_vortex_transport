"""
Route 24: α-rescan with full BZ k-grid (k=0.3-3.0, 13 pts).

Recomputes κ(α) with measured data across nearly full BZ.
No power law tail extrapolation — direct integration.
References shared across α. Uses parallel_fdtd fork(4).

BZ artifact concern: σ_tr rises at k > 1.7 (Route 23). May be physical
(Dirac disk microstructure) or BZ artifact (λ ≈ 3a at k=2.1).
Reports κ at three cutoffs (k≤1.5, k≤2.1, k≤3.0) to bracket systematic.

Results (R=5, L=80, 13 k-pts, 4 α values, fork(4), 16 min):

  κ(α) at different k-cutoffs:
    α     k≤1.5   k≤2.1   k≤3.0   spread
   0.25   0.625   0.898   1.211    94%
   0.30   0.944   1.341   1.752    86%
   0.40   1.545   2.188   2.760    79%
   0.50   1.798   2.552   3.185    77%

  κ=1 crossing:
    k≤1.5: α ≈ 0.309
    k≤2.1: α ≈ 0.261
    k≤3.0: below α=0.25 (κ(0.25)=1.211 already > 1)

  Spread 77-94% between cutoffs — dominant systematic uncertainty.
  σ_tr(k) has same pattern at all α: minimum at k≈1.7, then rises.
  Integrand sin²(k)·σ_tr monotonically decreasing at all α (no BZ warning).
  κ_data(α=0.3, k≤1.5) = 0.944 matches file 16 exactly.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/22_alpha_rescan.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from parallel_fdtd import compute_references, compute_scattering

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 5
N_POL = 2
N_WORKERS = 4

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

# Full BZ grid: 13 points from k=0.3 to k=3.0
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.4, 2.7, 3.0])

# Cutoffs for systematic uncertainty
K_CUTOFFS = [1.5, 2.1, 3.0]

# Reliable α only (β < -1 in old scan)
alpha_vals = [0.25, 0.30, 0.40, 0.50]

if __name__ == '__main__':
    t0 = time.time()

    print(f"Route 24: α-rescan with full BZ grid")
    print(f"  k-grid: {len(k_vals)} pts, k = {k_vals[0]:.1f} to {k_vals[-1]:.1f}")
    print(f"  α values: {alpha_vals}")
    print(f"  Cutoffs: {K_CUTOFFS}")
    print(f"  R={R_LOOP}, L={L}, workers={N_WORKERS}")
    print()

    # References (shared across all α)
    print(f"Computing {len(k_vals)} references...")
    t1 = time.time()
    refs = compute_references(k_vals, L, DW, DS, DT, sx, r_m,
                              thetas, phis, K1, K2, n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")
    print()

    # α scan
    prefactor = N_POL * R_LOOP / (4 * np.pi**2)

    all_results = []
    for alpha in alpha_vals:
        t1 = time.time()
        sigma_tr = compute_scattering(k_vals, refs, alpha, R_LOOP, L, DW, DS, DT,
                                      r_m, thetas, phis, K1, K2,
                                      n_workers=N_WORKERS)
        dt = time.time() - t1

        integrand = np.sin(k_vals)**2 * sigma_tr

        # κ at each cutoff
        kappas = {}
        for kc in K_CUTOFFS:
            mask = k_vals <= kc + 0.01
            kappas[kc] = prefactor * np.trapz(integrand[mask], k_vals[mask])

        # Check integrand monotonicity at high k
        warnings = []
        for j in range(len(k_vals)-1, max(0, len(k_vals)-4), -1):
            if integrand[j] > integrand[j-1]:
                warnings.append(k_vals[j])

        all_results.append({
            'alpha': alpha,
            'sigma_tr': sigma_tr.copy(),
            'integrand': integrand.copy(),
            'kappas': kappas,
            'dt': dt,
            'warnings': warnings,
        })
        print(f"  α={alpha:.2f}: done ({dt:.0f}s)" +
              (f"  WARNING: integrand rises at k={warnings}" if warnings else ""))

    # σ_tr table
    print()
    print("=" * 70)
    print("σ_tr(k) for each α")
    print("=" * 70)
    header = f"{'k':>5s}  {'sin²k':>6s}"
    for alpha in alpha_vals:
        header += f"  {'α='+str(alpha):>8s}"
    print(header)
    print("-" * (14 + 10 * len(alpha_vals)))

    for j, k0 in enumerate(k_vals):
        line = f"{k0:5.2f}  {np.sin(k0)**2:6.4f}"
        for r in all_results:
            line += f"  {r['sigma_tr'][j]:8.3f}"
        print(line)

    # Integrand table
    print()
    print("sin²(k)·σ_tr (integrand) for each α")
    header = f"{'k':>5s}"
    for alpha in alpha_vals:
        header += f"  {'α='+str(alpha):>8s}"
    print(header)
    print("-" * (5 + 10 * len(alpha_vals)))

    for j, k0 in enumerate(k_vals):
        line = f"{k0:5.2f}"
        for r in all_results:
            line += f"  {r['integrand'][j]:8.3f}"
        print(line)

    # κ at each cutoff
    print()
    print("=" * 70)
    print("κ(α) at different k-cutoffs (systematic uncertainty)")
    print("=" * 70)
    header = f"{'α':>5s}"
    for kc in K_CUTOFFS:
        header += f"  {'k≤'+str(kc):>8s}"
    header += f"  {'spread':>7s}"
    print(header)
    print("-" * (5 + 10 * len(K_CUTOFFS) + 9))

    for r in all_results:
        line = f"{r['alpha']:5.2f}"
        vals = []
        for kc in K_CUTOFFS:
            v = r['kappas'][kc]
            vals.append(v)
            line += f"  {v:8.3f}"
        spread = (max(vals) - min(vals)) / min(vals) * 100
        line += f"  {spread:+6.1f}%"
        print(line)

    # κ = 1 crossing at each cutoff
    print()
    for kc in K_CUTOFFS:
        kappas_at_cutoff = [r['kappas'][kc] for r in all_results]
        alphas_list = [r['alpha'] for r in all_results]
        for i in range(len(kappas_at_cutoff) - 1):
            if kappas_at_cutoff[i] <= 1.0 <= kappas_at_cutoff[i + 1]:
                frac = (1.0 - kappas_at_cutoff[i]) / \
                       (kappas_at_cutoff[i + 1] - kappas_at_cutoff[i])
                alpha_cross = alphas_list[i] + frac * (alphas_list[i + 1] - alphas_list[i])
                print(f"κ = 1.0 at α ≈ {alpha_cross:.3f} (cutoff k≤{kc})")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
