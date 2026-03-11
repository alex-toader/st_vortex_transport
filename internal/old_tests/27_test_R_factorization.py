"""
Test R3: R^{3/2} factorization at weak vs strong coupling.

Tests the empirical formula: σ_tr(k, R) ≈ A(α) × R^{3/2} / sin²(k)

The factorization separates:
  R^{3/2} = geometric (stationary phase, universal)
  1/sin²(k) = strong-coupling physics (requires α ≥ 0.2)

At α=0.3 (strong coupling, file 18):
  g(k) × sin²(k) has CV = 5.7% — FLAT
  g(k) × k^{0.5}  has CV = 79.4% — NOT flat

This test runs the same R-scan at α=0.05 (weak coupling).
If R^{3/2} holds but 1/sin²(k) breaks → confirms factorization.

Results (R=3,5,7,9, L=80, r_m=20, fork(4), 432s):

  σ_tr(k, R) at α=0.05:

    k       R=3      R=5      R=7      R=9
   0.30     0.05     0.24     0.52     0.98
   0.50     0.04     0.09     0.15     0.22
   0.70     0.03     0.06     0.10     0.15
   0.90     0.03     0.05     0.08     0.13
   1.10     0.02     0.05     0.08     0.12
   1.30     0.02     0.05     0.08     0.12
   1.50     0.03     0.05     0.08     0.13

  σ_tr(k, R) at α=0.30 (consistency, matches file 18 to <0.3%):

    k       R=3      R=5      R=7      R=9
   0.30    13.16    40.74    72.70   117.84
   0.50     6.43    14.16    22.23    34.54
   0.70     3.54     7.69    12.22    19.43
   0.90     2.28     5.05     8.16    13.34
   1.10     1.72     3.78     6.26    10.29
   1.30     1.43     3.14     5.25     8.56
   1.50     1.36     2.81     4.71     7.74

  R-exponent p(k):

    k     p(α=0.05)  p(α=0.30)
   0.30     2.62       1.99
   0.50     1.56       1.51
   0.70     1.43       1.53
   0.90     1.44       1.58
   1.10     1.43       1.61
   1.30     1.43       1.61
   1.50     1.43       1.56

  k-dependence (g = σ_tr / R^1.5, averaged):
    α=0.05: CV(g·sin²) = 33.1% — NOT FLAT
    α=0.30: CV(g·sin²) = 5.7%  — FLAT

  k-dependence (g = σ_tr / R^{p(k)} measured):
    α=0.05: CV(g·sin²) = 52.4% — NOT FLAT
    α=0.30: CV(g·sin²) = 19.1% — NOT FLAT (but p(k) varies 1.51-1.99)

  CONCLUSION:
    R^{3/2}: holds at BOTH α. p(k≥0.5) = 1.45 (α=0.05) and 1.57 (α=0.30).
      At weak coupling p is CLOSER to 1.5 (less near-field/transport corrections).
    1/sin²(k): ONLY at α=0.30 (CV=5.7%). Breaks completely at α=0.05 (CV=33%).
    Factorization CONFIRMED: R-part is geometric (universal), k-part is strong-coupling.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/27_test_R_factorization.py
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
N_POL = 2
N_WORKERS = 4

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
R_vals = [3, 5, 7, 9]
alpha_vals = [0.05, 0.30]  # weak + strong (strong = consistency check vs file 18)

if __name__ == '__main__':
    t0 = time.time()

    print("Test R3: R^{3/2} factorization at weak vs strong coupling")
    print(f"  L={L}, r_m={r_m}, workers={N_WORKERS}")
    print(f"  R values: {R_vals}")
    print(f"  k values: {list(k_vals)}")
    print(f"  α values: {alpha_vals}")
    print()

    # Shared references (α-independent)
    print(f"Computing {len(k_vals)} references...")
    t1 = time.time()
    refs = compute_references(k_vals, L, DW, DS, DT, sx, r_m,
                              thetas, phis, K1, K2, n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")
    print()

    # Scattering at each α × R
    all_data = {}  # (alpha, R) -> sigma_tr array
    for alpha in alpha_vals:
        print(f"α = {alpha:.2f}:")
        for R in R_vals:
            t1 = time.time()
            sigma_tr = compute_scattering(k_vals, refs, alpha, R, L, DW, DS, DT,
                                          r_m, thetas, phis, K1, K2,
                                          n_workers=N_WORKERS)
            dt = time.time() - t1
            all_data[(alpha, R)] = sigma_tr.copy()
            print(f"  R={R}: done ({dt:.0f}s)")
        print()

    # ── Raw σ_tr tables ──
    for alpha in alpha_vals:
        print("=" * 60)
        print(f"σ_tr(k, R) at α={alpha:.2f}")
        print("=" * 60)
        header = f"  {'k':>5s}"
        for R in R_vals:
            header += f"  {'R='+str(R):>8s}"
        print(header)
        print(f"  {'-'*40}")
        for j, k0 in enumerate(k_vals):
            line = f"  {k0:5.2f}"
            for R in R_vals:
                line += f"  {all_data[(alpha, R)][j]:8.2f}"
            print(line)
        print()

    # ── Analysis 1: R-exponent p(k) at each α ──
    print("=" * 60)
    print("R-exponent p(k): fit σ_tr ~ R^p at each k")
    print("=" * 60)
    logR = np.log(np.array(R_vals, dtype=float))
    for alpha in alpha_vals:
        print(f"\n  α = {alpha:.2f}:")
        print(f"  {'k':>5s}  {'p(all R)':>8s}  {'p(R=3,5)':>9s}")
        print(f"  {'-'*25}")
        for j, k0 in enumerate(k_vals):
            s = np.array([all_data[(alpha, R)][j] for R in R_vals])
            logS = np.log(s)
            p_all = np.polyfit(logR, logS, 1)[0]
            p_far = np.polyfit(logR[:2], logS[:2], 1)[0]
            print(f"  {k0:5.2f}  {p_all:8.2f}  {p_far:9.2f}")

    # ── Analysis 2a: k-dependence with fixed R^{1.5} ──
    print()
    print("=" * 60)
    print("k-dependence test (2a): g(k) = σ_tr / R^{1.5} fixed")
    print("=" * 60)
    R_arr = np.array(R_vals, dtype=float)
    for alpha in alpha_vals:
        print(f"\n  α = {alpha:.2f}:")
        sigma_matrix = np.array([all_data[(alpha, R)] for R in R_vals]).T
        g_vals = sigma_matrix / R_arr[np.newaxis, :] ** 1.5
        g_mean = np.mean(g_vals, axis=1)

        print(f"  {'k':>5s}  {'g(k)':>9s}  {'g*sin²(k)':>10s}  {'g*k^0.5':>9s}")
        print(f"  {'-'*40}")
        for j, k0 in enumerate(k_vals):
            g = g_mean[j]
            print(f"  {k0:5.2f}  {g:9.4f}  {g*np.sin(k0)**2:10.4f}  {g*k0**0.5:9.4f}")

        gs = g_mean * np.sin(k_vals) ** 2
        gk = g_mean * k_vals ** 0.5
        cv_sin2 = np.std(gs) / np.mean(gs) * 100
        cv_sqrt = np.std(gk) / np.mean(gk) * 100
        print(f"\n  CV of g*sin²(k): {cv_sin2:.1f}%  {'FLAT' if cv_sin2 < 15 else 'NOT FLAT'}")
        print(f"  CV of g*k^0.5:   {cv_sqrt:.1f}%  {'FLAT' if cv_sqrt < 15 else 'NOT FLAT'}")

    # ── Analysis 2b: k-dependence with measured p(k) ──
    # Uses p(k) from Analysis 1 to remove R-scaling residuals
    print()
    print("=" * 60)
    print("k-dependence test (2b): g(k) = σ_tr / R^{p(k)} measured")
    print("=" * 60)
    for alpha in alpha_vals:
        print(f"\n  α = {alpha:.2f}:")
        sigma_matrix = np.array([all_data[(alpha, R)] for R in R_vals]).T

        # Fit p(k) at each k
        p_measured = np.zeros(len(k_vals))
        for j in range(len(k_vals)):
            logS = np.log(sigma_matrix[j])
            p_measured[j] = np.polyfit(logR, logS, 1)[0]

        # Extract g(k) using measured p(k)
        g_corrected = np.zeros(len(k_vals))
        for j in range(len(k_vals)):
            g_j = sigma_matrix[j] / R_arr ** p_measured[j]
            g_corrected[j] = np.mean(g_j)

        print(f"  {'k':>5s}  {'p(k)':>6s}  {'g(k)':>9s}  {'g*sin²(k)':>10s}  {'g*k^0.5':>9s}")
        print(f"  {'-'*48}")
        for j, k0 in enumerate(k_vals):
            g = g_corrected[j]
            print(f"  {k0:5.2f}  {p_measured[j]:6.2f}  {g:9.4f}  "
                  f"{g*np.sin(k0)**2:10.4f}  {g*k0**0.5:9.4f}")

        gs = g_corrected * np.sin(k_vals) ** 2
        gk = g_corrected * k_vals ** 0.5
        cv_sin2 = np.std(gs) / np.mean(gs) * 100
        cv_sqrt = np.std(gk) / np.mean(gk) * 100
        print(f"\n  CV of g*sin²(k): {cv_sin2:.1f}%  {'FLAT' if cv_sin2 < 15 else 'NOT FLAT'}")
        print(f"  CV of g*k^0.5:   {cv_sqrt:.1f}%  {'FLAT' if cv_sqrt < 15 else 'NOT FLAT'}")
        print(f"  (p(k) range: {p_measured.min():.2f} to {p_measured.max():.2f})")

    # ── Analysis 3: Consistency check vs file 18 ──
    print()
    print("=" * 60)
    print("Consistency: α=0.30 vs file 18")
    print("=" * 60)
    ref_18 = {
        (0.30, 3): np.array([13.16, 6.43, 3.54, 2.28, 1.72, 1.43, 1.36]),
        (0.30, 5): np.array([40.74, 14.16, 7.69, 5.05, 3.78, 3.14, 2.81]),
        (0.30, 7): np.array([72.70, 22.23, 12.22, 8.16, 6.26, 5.25, 4.71]),
        (0.30, 9): np.array([117.84, 34.54, 19.43, 13.34, 10.29, 8.56, 7.74]),
    }
    max_diff = 0
    for R in R_vals:
        s_new = all_data[(0.30, R)]
        s_old = ref_18[(0.30, R)]
        diff_pct = np.max(np.abs(s_new - s_old) / s_old) * 100
        max_diff = max(max_diff, diff_pct)
        print(f"  R={R}: max diff = {diff_pct:.2f}%")
    if max_diff < 2:
        print(f"  CONSISTENT (max {max_diff:.2f}%)")
    else:
        print(f"  WARNING: max diff {max_diff:.2f}% — investigate!")

    # ── Summary ──
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for alpha in alpha_vals:
        sigma_matrix = np.array([all_data[(alpha, R)] for R in R_vals]).T

        # p(k) measured
        p_vals = []
        for j in range(len(k_vals)):
            s = sigma_matrix[j]
            p_vals.append(np.polyfit(logR, np.log(s), 1)[0])
        p_mean = np.mean(p_vals[1:])  # exclude k=0.3 (geometric)

        # Method A: fixed R^{1.5}
        g_fixed = np.mean(sigma_matrix / R_arr[np.newaxis, :] ** 1.5, axis=1)
        cv_fixed = np.std(g_fixed * np.sin(k_vals)**2) / np.mean(g_fixed * np.sin(k_vals)**2) * 100

        # Method B: measured p(k)
        g_corr = np.zeros(len(k_vals))
        for j in range(len(k_vals)):
            g_corr[j] = np.mean(sigma_matrix[j] / R_arr ** p_vals[j])
        cv_corr = np.std(g_corr * np.sin(k_vals)**2) / np.mean(g_corr * np.sin(k_vals)**2) * 100

        print(f"  α={alpha:.2f}: p(k≥0.5) = {p_mean:.2f}")
        print(f"    CV(g·sin²) fixed R^1.5: {cv_fixed:.1f}%")
        print(f"    CV(g·sin²) with p(k):   {cv_corr:.1f}%")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
