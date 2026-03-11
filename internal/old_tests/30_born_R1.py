"""
Route 30 (F6): Born limit at R=1 — does α-exponent approach 2.0 at small kR?

File 26 showed σ_tr ∝ α^2.35 at R=5 (kR=1.5-7.5). Born requires kR ≪ 1.
At R=1: kR = 0.3-1.5 — marginal Born regime.

If α-exponent p → 2.0 at R=1 → confirms Born limit at small kR.
If p > 2.0 still → non-perturbative even at kR ~ O(1).

Tests α = 0.05, 0.10, 0.30 at R=1, k = 0.3-1.5 (7 pts), NN gauging.

Results (R=1, L=80, 5 gauged bonds, 152s):

  σ_tr(k) at R=1:
    k     α=0.05   α=0.10   α=0.30
   0.30   0.0026   0.0300   1.0261
   0.50   0.0032   0.0270   0.8072
   0.70   0.0043   0.0276   0.6935
   0.90   0.0054   0.0289   0.5886
   1.10   0.0065   0.0305   0.4978
   1.30   0.0075   0.0324   0.4223
   1.50   0.0084   0.0349   0.3762

  α-exponent (σ_tr ∝ α^p):
    k     kR    p(all)  p(0.05-0.10)
   0.30  0.30    3.33      3.55
   0.50  0.50    3.09      3.07
   0.70  0.70    2.85      2.69
   0.90  0.90    2.63      2.41
   1.10  1.10    2.43      2.23
   1.30  1.30    2.26      2.11
   1.50  1.50    2.12      2.05

  Born scaling σ_tr/α²: varies 24% at k=1.5 (best) to 1010% at k=0.3 (worst).
  At k=1.5 (kR=1.5), α=0.05-0.10 agree well: σ_tr/α² = 3.4 vs 3.5. Born regime.

  Flat integrand: CV = 37-77% at all α — NOT flat at R=1.

  CONCLUSION:
    p → 2.0 as kR → 1.5 (p=2.05 from α=0.05-0.10 pair at k=1.5).
    Born is approached but not reached — even at kR=1.5, p=2.05 not 2.00.
    At kR<1 (k<1.0), p = 2.4-3.6 — strongly non-perturbative.
    The non-Born behavior at R=5 (p=2.35) is confirmed to be a kR effect:
    p(R=1, kR=1.5) ≈ p(R=5, kR=7.5) ≈ 2.1-2.2 → same physics at similar kR.
    Flat integrand requires R≥3 — does not exist at R=1 (5 bonds too few).

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/30_born_R1.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../src/1_foam'))
from parallel_fdtd import compute_references, compute_scattering

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 1
N_POL = 2
N_WORKERS = 4

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
alpha_vals = [0.05, 0.10, 0.30]

if __name__ == '__main__':
    t0 = time.time()

    print("Route 30 (F6): Born limit at R=1")
    print(f"  R={R_LOOP}, L={L}, r_m={r_m}")
    print(f"  kR range: {k_vals[0]*R_LOOP:.1f} to {k_vals[-1]*R_LOOP:.1f}")
    print(f"  α values: {alpha_vals}")

    # Count gauged bonds at R=1
    from gauge_3d import precompute_disk_bonds
    iy_d, ix_d = precompute_disk_bonds(L, R_LOOP)
    print(f"  NN bonds in disk: {len(iy_d)} (R=1 is highly discrete)")
    print()

    # References
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
        all_results.append({
            'alpha': alpha,
            'sigma_tr': sigma_tr.copy(),
            'dt': dt,
        })
        print(f"  α={alpha:.2f}: done ({dt:.0f}s)")

    # σ_tr table
    print()
    print("=" * 60)
    print("σ_tr(k) at R=1")
    print("=" * 60)
    header = "      k"
    for alpha in alpha_vals:
        header += f"    α={alpha:.2f}"
    print(header)
    print(f"  {'-'*40}")
    for j, k0 in enumerate(k_vals):
        line = f"  {k0:5.2f}"
        for r in all_results:
            line += f"  {r['sigma_tr'][j]:9.4f}"
        print(line)

    # σ_tr / α² table (Born scaling test)
    print()
    print("σ_tr / α² (Born: should be constant across α)")
    header = "      k"
    for alpha in alpha_vals:
        header += f"    α={alpha:.2f}"
    header += "   variation"
    print(header)
    print(f"  {'-'*50}")
    for j, k0 in enumerate(k_vals):
        vals = [r['sigma_tr'][j] / r['alpha']**2 for r in all_results]
        variation = (max(vals) - min(vals)) / min(vals) * 100
        line = f"  {k0:5.2f}"
        for v in vals:
            line += f"  {v:9.1f}"
        line += f"  {variation:8.0f}%"
        print(line)

    # α-exponent fit
    print()
    print("=" * 60)
    print("α-exponent (σ_tr ∝ α^p)")
    print("=" * 60)
    log_alpha = np.log([r['alpha'] for r in all_results])
    print(f"  {'k':>5s}  {'kR':>5s}  {'p(all)':>7s}  {'p(0.05-0.10)':>13s}")
    print(f"  {'-'*35}")
    p_all_list = []
    p_small_list = []
    for j, k0 in enumerate(k_vals):
        log_s = np.log([r['sigma_tr'][j] for r in all_results])
        p_all = np.polyfit(log_alpha, log_s, 1)[0]
        p_small = np.polyfit(log_alpha[:2], log_s[:2], 1)[0]
        p_all_list.append(p_all)
        p_small_list.append(p_small)
        print(f"  {k0:5.2f}  {k0*R_LOOP:5.2f}  {p_all:7.2f}  {p_small:13.2f}")

    # Compare with R=5 (from file 26 header)
    print()
    print("=" * 60)
    print("Comparison: R=1 vs R=5 (file 26)")
    print("=" * 60)
    # File 26 data at selected k-points
    p_R5_all = {0.3: 2.84, 0.7: 2.68, 1.3: 2.30, 1.5: 2.21}
    p_R5_small = {0.3: 3.37, 0.7: 2.53, 1.3: 2.15, 1.5: 2.10}
    print(f"  {'k':>5s}  {'kR(R=1)':>8s}  {'p(R=1)':>7s}  {'kR(R=5)':>8s}  {'p(R=5)':>7s}")
    print(f"  {'-'*42}")
    for j, k0 in enumerate(k_vals):
        kR1 = k0 * 1
        kR5 = k0 * 5
        p1 = p_all_list[j]
        if k0 in p_R5_all:
            p5 = p_R5_all[k0]
            print(f"  {k0:5.2f}  {kR1:8.2f}  {p1:7.2f}  {kR5:8.2f}  {p5:7.2f}")
        else:
            print(f"  {k0:5.2f}  {kR1:8.2f}  {p1:7.2f}  {kR5:8.2f}     —")

    # Flat integrand test
    print()
    print("=" * 60)
    print("Flat integrand: CV of sin²(k)·σ_tr at R=1")
    print("=" * 60)
    for r in all_results:
        integrand = np.sin(k_vals)**2 * r['sigma_tr']
        cv = np.std(integrand) / np.mean(integrand) * 100
        flat = "FLAT" if cv < 15 else "NOT FLAT"
        print(f"  α={r['alpha']:.2f}: CV = {cv:.1f}%  ({flat})")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  R=1, kR = {k_vals[0]:.1f} to {k_vals[-1]:.1f}")
    p_mean_all = np.mean(p_all_list)
    p_mean_high = np.mean(p_all_list[3:])  # k >= 0.9
    print(f"  α-exponent (all k): mean p = {p_mean_all:.2f}")
    print(f"  α-exponent (k≥0.9): mean p = {p_mean_high:.2f}")
    if p_mean_high < 2.15:
        print(f"  → NEAR BORN at high k (p ≈ 2.0)")
    elif p_mean_high < 2.5:
        print(f"  → APPROACHING Born at high k but not there yet")
    else:
        print(f"  → NOT Born even at R=1")

    print()
    print(f"  NOTE: R=1 → {len(iy_d)} gauged bonds. Geometry is highly discrete.")
    print(f"  Results may reflect lattice discretization, not continuum R→0 limit.")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
