"""
Test F: Born limit — is flat integrand universal or strong-coupling only?

Measures σ_tr at α=0.05, 0.10, 0.30 on k=0.3-1.5 (7 pts).
If sin²(k)·σ_tr is flat at ALL α → lattice property (universal).
If flat only at large α → strong-coupling phenomenon.

Also tests Born scaling: σ_tr ∝ α² requires σ_tr/α² = const across α.

Results (R=5, L=80, fork(4), 148s):

  Flat integrand test — CV of sin²(k)·σ_tr:
    α=0.05: CV = 34.2% (integrand increases 2.6× from k=0.3 to k=1.5)
    α=0.10: CV = 15.3%
    α=0.30: CV = 7.5%  (flat)

  Flat integrand is NOT universal — appears only at strong coupling (α ≥ 0.2).

  Born scaling test — σ_tr/α² varies 44-373% across α (average 187%).
  kR = 1.5-7.5 — Born requires kR ≪ 1 AND α ≪ 1. Both fail here.

  Consistency: σ_tr(α=0.30, k=0.30) = 40.74, matches file 16 exactly.

  F7: α-exponent (σ_tr ∝ α^p):
    k     p(all)   p(0.05-0.10)
   0.30    2.84       3.37
   0.70    2.68       2.53
   1.30    2.30       2.15
   1.50    2.21       2.10

  At high k, p → 2.0 (approaching Born). At low k, p > 2 (non-perturbative).
  Even at α=0.05-0.10, p > 2.1 at all k — not in Born regime.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/26_test_born_limit.py
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

k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
alpha_vals = [0.05, 0.10, 0.30]

if __name__ == '__main__':
    t0 = time.time()

    print("Test F: Born limit + flat integrand universality")
    print(f"  R={R_LOOP}, L={L}, workers={N_WORKERS}")
    print(f"  k-grid: {len(k_vals)} pts, k = {k_vals[0]:.1f} to {k_vals[-1]:.1f}")
    print(f"  α values: {alpha_vals}")
    print()

    # Shared references
    print(f"Computing {len(k_vals)} references...")
    t1 = time.time()
    refs = compute_references(k_vals, L, DW, DS, DT, sx, r_m,
                              thetas, phis, K1, K2, n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")
    print()

    # Scattering at each α
    all_results = []
    for alpha in alpha_vals:
        t1 = time.time()
        sigma_tr = compute_scattering(k_vals, refs, alpha, R_LOOP, L, DW, DS, DT,
                                      r_m, thetas, phis, K1, K2,
                                      n_workers=N_WORKERS)
        dt = time.time() - t1
        integrand = np.sin(k_vals)**2 * sigma_tr
        cv = np.std(integrand) / np.mean(integrand) * 100
        all_results.append({
            'alpha': alpha,
            'sigma_tr': sigma_tr.copy(),
            'integrand': integrand.copy(),
            'cv': cv,
            'dt': dt,
        })
        print(f"  α={alpha:.2f}: done ({dt:.0f}s), CV(integrand) = {cv:.1f}%")

    # σ_tr table
    print()
    print("=" * 70)
    print("σ_tr(k) for each α")
    print("=" * 70)
    header = f"  {'k':>5s}"
    for alpha in alpha_vals:
        header += f"  {'α='+str(alpha):>10s}"
    print(header)
    print(f"  {'-'*40}")
    for j, k0 in enumerate(k_vals):
        line = f"  {k0:5.2f}"
        for r in all_results:
            line += f"  {r['sigma_tr'][j]:10.4f}"
        print(line)

    # Integrand table
    print()
    print("sin²(k)·σ_tr (integrand) for each α")
    header = f"  {'k':>5s}"
    for alpha in alpha_vals:
        header += f"  {'α='+str(alpha):>10s}"
    print(header)
    print(f"  {'-'*40}")
    for j, k0 in enumerate(k_vals):
        line = f"  {k0:5.2f}"
        for r in all_results:
            line += f"  {r['integrand'][j]:10.4f}"
        print(line)

    # CV summary
    print()
    print("=" * 70)
    print("Flat integrand test: CV of sin²(k)·σ_tr")
    print("=" * 70)
    for r in all_results:
        flat = "FLAT" if r['cv'] < 15 else "NOT FLAT"
        print(f"  α={r['alpha']:.2f}: CV = {r['cv']:.1f}%  ({flat})")

    print()
    if all_results[0]['cv'] > 20 and all_results[-1]['cv'] < 15:
        print("** Flat integrand is STRONG-COUPLING phenomenon. **")
        print("   Not universal — only appears at α ≥ 0.2.")
    elif all_results[0]['cv'] < 15:
        print("** Flat integrand is UNIVERSAL (appears even at α=0.05). **")
    else:
        print("** Flat integrand status: inconclusive. **")

    # Born scaling test: σ_tr / α²
    print()
    print("=" * 70)
    print("Born scaling test: σ_tr / α²")
    print("=" * 70)
    header = f"  {'k':>5s}"
    for alpha in alpha_vals:
        header += f"  {'α='+str(alpha):>10s}"
    header += f"  {'variation':>10s}"
    print(header)
    print(f"  {'-'*50}")
    for j, k0 in enumerate(k_vals):
        vals = [r['sigma_tr'][j] / r['alpha']**2 for r in all_results]
        variation = (max(vals) - min(vals)) / min(vals) * 100
        line = f"  {k0:5.2f}"
        for v in vals:
            line += f"  {v:10.1f}"
        line += f"  {variation:9.0f}%"
        print(line)

    print()
    print(f"  Note: kR = {k_vals[0]*R_LOOP:.1f}-{k_vals[-1]*R_LOOP:.1f}.")
    print(f"  Born requires kR << 1 AND α << 1. Both conditions fail here.")

    # Overall Born test
    born_violations = []
    for j in range(len(k_vals)):
        vals = [r['sigma_tr'][j] / r['alpha']**2 for r in all_results]
        born_violations.append((max(vals) - min(vals)) / min(vals) * 100)
    mean_violation = np.mean(born_violations)
    if mean_violation > 30:
        print(f"\n** NOT in Born regime: σ_tr/α² varies {mean_violation:.0f}% on average. **")
        print(f"   Born scaling (σ_tr ∝ α²) requires much smaller α or smaller R.")
    else:
        print(f"\n** Near Born regime: σ_tr/α² varies only {mean_violation:.0f}%. **")

    # Consistency check vs known data
    print()
    print("=" * 70)
    print("Consistency check")
    print("=" * 70)
    idx_030 = alpha_vals.index(0.30)
    s_030_k03 = all_results[idx_030]['sigma_tr'][0]  # k=0.3
    print(f"  σ_tr(α=0.30, k=0.30) = {s_030_k03:.2f}")
    print(f"  Expected from file 16: 40.74")
    diff_pct = abs(s_030_k03 - 40.74) / 40.74 * 100
    if diff_pct < 5:
        print(f"  Difference: {diff_pct:.1f}% — CONSISTENT")
    else:
        print(f"  Difference: {diff_pct:.1f}% — INCONSISTENT, investigate!")

    # F7: α-exponent fit — does σ_tr → α² as α → 0?
    print()
    print("=" * 70)
    print("F7: α-exponent (σ_tr ∝ α^p)")
    print("=" * 70)
    log_alpha = np.log([r['alpha'] for r in all_results])
    print(f"  {'k':>5s}  {'p (all)':>8s}  {'p (0.05-0.10)':>14s}")
    print(f"  {'-'*30}")
    for j, k0 in enumerate(k_vals):
        log_s = np.log([r['sigma_tr'][j] for r in all_results])
        p_all = np.polyfit(log_alpha, log_s, 1)[0]
        # Fit just the two smallest α
        p_small = np.polyfit(log_alpha[:2], log_s[:2], 1)[0]
        print(f"  {k0:5.2f}  {p_all:8.2f}  {p_small:14.2f}")
    print()
    print("  Born: p → 2.0 as α → 0. p < 2 = non-perturbative.")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
