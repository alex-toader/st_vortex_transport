"""
Route 44 (F8): σ_tr vs N_bonds — arc fraction scaling.

Tests whether σ_tr scales as N (incoherent), N^{3/2} (stationary phase),
or N² (coherent) by measuring σ_tr on angular arcs of the Dirac disk.

NOTE on two kinds of exponent:
  p_arc (this file): how σ scales when removing bonds from a fixed-R disk.
  p_R (files 18/27): how σ scales with ring radius R (all bonds present).
  Both give p=1 for incoherent. Stationary phase gives p_arc=1, p_R=3/2.
  They probe different physics: p_arc tests bond independence,
  p_R tests geometric scaling of the active bond set.

Known result (from file 42, not re-run):
  Full ring (NN, α=0.30, R=5): 81 bonds.

NOTE: File 42's half-ring uses iy >= cy filter (46 bonds), NOT angular filter.
Angular half (angle < π) gives 41 bonds — different geometry because 5 diameter
bonds at ix<cx have angle=π exactly, excluded by strict inequality angle < π.
This file uses angular filter throughout for consistency. File 42 half shown
for comparison only, NOT included in the fit.

Arcs (angular filter on [0, 2π) angles from disk center):
  Q1 front (0, π/2):     21 bonds — faces incoming wave (+x direction)
  Q3 back (π, 3π/2):     20 bonds — opposite to incoming wave
  Third (0, 2π/3):       ~27 bonds
  Angular half (0, π):    41 bonds — consistent angular filter
  Two-thirds (0, 4π/3): ~54 bonds
  Three-quarter (0, 3π/2): ~61 bonds

Symmetry test: Q1 vs Q3. If σ(Q1)/σ(Q3) ≈ N(Q1)/N(Q3) → isotropic on disk
(bond position vs wave direction doesn't matter → incoherent).

Log-log fit on 6 arc points: log(σ) = p × log(N) + C.
Full ring excluded from fit (at N=81, σ/σ_full = 1 is trivially satisfied,
adds no constraint on slope). σ(N=81) predicted from fit as independent check.

  p = 1.0 → incoherent.  p = 1.5 → stationary phase.  p = 2.0 → coherent.

Results:

Bond counts: Q1=21, Q3=20, third=30, ang-half=41, two-thirds=57, three-quarter=61.

Symmetry test Q1 (front) vs Q3 (back) — ANISOTROPIC:
  Mean Q1/Q3 = 0.78 (drifts 0.90→0.74 from k=0.3 to k=1.5).
  σ(Q1) < σ(Q3) despite N(Q1) > N(Q3). Back-facing bonds scatter more.
  Cause: transport weighting (1-cosθ_s). Q1 bonds are near forward direction
  where (1-cosθ_s)→0, so they contribute less to σ_tr per bond.

Exponent fit (6 arcs, full ring excluded):
  p_arc = 0.792 ± 0.134, R² = 0.66–0.97. Sub-linear (p < 1).
  Full ring prediction from fit: pred/actual = 0.92–1.02.
  p < 1 because arcs from angle=0 accumulate forward bonds first (low σ/bond),
  then add lateral/back bonds (high σ/bond) → sub-linear growth.

CV of sin²(k)·σ_tr vs arc size:
  Q3=16.7%, Q1=12.3%, third=5.0%, ang-half=5.1%, two-thirds=5.6%,
  three-quarter=4.1%, full=7.5%.
  Flatness degrades at quarter-arcs (~20 bonds), robust at ≥30 bonds.

Conclusion: p_arc is NOT a clean coherence diagnostic because σ_per_bond
depends on bond position (transport weighting effect). Scattering can be
fully incoherent with position-dependent σ_j. The clean incoherence tests
remain file 42 (half/full=0.50) and file 43 (incoherent Born CV=2.7%).

σ_tr values (k = 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5):
  Q1 front (21):       8.20  4.43  2.67  1.78  1.35  1.12  1.02
  Q3 back (20):        9.12  5.52  3.48  2.35  1.82  1.54  1.38
  third (30):         12.34  5.28  2.97  2.03  1.51  1.23  1.13
  ang-half (41):      15.89  5.29  3.00  2.20  1.77  1.47  1.38
  two-thirds (57):    26.97 11.15  6.80  4.47  3.32  2.77  2.49
  three-quarter (61): 30.50 12.49  7.27  4.70  3.56  3.00  2.69
  full ring (81):     40.74 14.16  7.69  5.05  3.78  3.14  2.81

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/44_arc_scaling.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../src/1_foam'))
from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           compute_sphere_f2, integrate_sigma_3d,
                           estimate_n_steps_3d)
from gauge_3d import precompute_disk_bonds

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 5
ALPHA = 0.30

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# Known results from file 42 (not re-run)
sigma_full = np.array([40.7432, 14.1580, 7.6935, 5.0477, 3.7764, 3.1376, 2.8065])
N_FULL = 81

# File 42 half-ring (iy >= cy filter) — for comparison only, NOT in fit
sigma_half_f42 = np.array([18.3667, 6.2898, 3.6647, 2.5977, 2.0348, 1.6899, 1.5273])
N_HALF_F42 = 46

# Angular arcs: (label, angle_min, angle_max)
ARCS = [
    ("Q1 front (0,π/2)",       0,              np.pi / 2),
    ("Q3 back (π,3π/2)",       np.pi,          3 * np.pi / 2),
    ("third (0,2π/3)",         0,              2 * np.pi / 3),
    ("ang-half (0,π)",         0,              np.pi),
    ("two-thirds (0,4π/3)",    0,              4 * np.pi / 3),
    ("three-quarter (0,3π/2)", 0,              3 * np.pi / 2),
]


def make_arc_vortex(alpha, R_loop, L, angle_min, angle_max, K1=1.0, K2=0.5):
    """Peierls vortex on disk bonds with angle in [angle_min, angle_max)."""
    iy_disk, ix_disk = precompute_disk_bonds(L, R_loop)
    cx, cy = L // 2, L // 2
    cz = L // 2

    # Compute angles in [0, 2π)
    angles = np.arctan2((iy_disk - cy).astype(float),
                        (ix_disk - cx).astype(float))
    angles = angles % (2 * np.pi)

    # Select bonds in [angle_min, angle_max)
    # Center bond (cx,cy) has angle=0, naturally included when 0 is in range
    if angle_min < angle_max:
        mask = (angles >= angle_min) & (angles < angle_max)
    else:
        # Wraps around 2π (not used currently but supports future arcs)
        mask = (angles >= angle_min) | (angles < angle_max)

    iy_arc = iy_disk[mask]
    ix_arc = ix_disk[mask]
    n_bonds = len(iy_arc)

    phi = 2.0 * np.pi * (float(alpha) % 1.0)
    cm1 = np.cos(phi) - 1.0
    s_phi = np.sin(phi)
    if abs(s_phi) < 1e-12:
        s_phi = 0.0
    if abs(cm1) < 1e-12:
        cm1 = 0.0

    iz_lo = cz - 1
    iz_hi = cz

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if cm1 == 0.0 and s_phi == 0.0:
            return fx, fy, fz

        ux_hi = ux[iz_hi, iy_arc, ix_arc]
        uy_hi = uy[iz_hi, iy_arc, ix_arc]
        ux_lo = ux[iz_lo, iy_arc, ix_arc]
        uy_lo = uy[iz_lo, iy_arc, ix_arc]

        fx[iz_lo, iy_arc, ix_arc] += K1 * (cm1 * ux_hi - s_phi * uy_hi)
        fy[iz_lo, iy_arc, ix_arc] += K1 * (s_phi * ux_hi + cm1 * uy_hi)

        fx[iz_hi, iy_arc, ix_arc] += K1 * (cm1 * ux_lo + s_phi * uy_lo)
        fy[iz_hi, iy_arc, ix_arc] += K1 * (-s_phi * ux_lo + cm1 * uy_lo)

        return fx, fy, fz

    force_fn.n_bonds = n_bonds
    return force_fn


if __name__ == '__main__':
    t0 = time.time()

    print("Route 44 (F8): Arc fraction scaling — σ_tr vs N_bonds")
    print(f"  R={R_LOOP}, α={ALPHA}, L={L}")
    print(f"  k = {list(k_vals)}")
    print()

    # Verify disk geometry
    iy_disk, ix_disk = precompute_disk_bonds(L, R_LOOP)
    cx, cy = L // 2, L // 2
    center = (iy_disk == cy) & (ix_disk == cx)
    print(f"  Full disk: {len(iy_disk)} bonds")
    print(f"  Center bond (cx,cy): {int(center.sum())}"
          f" (at angle=0, included in arcs starting from 0)")
    print()

    # Show bond counts for each arc
    print("  Arc bond counts:")
    for label, a_min, a_max in ARCS:
        f = make_arc_vortex(ALPHA, R_LOOP, L, a_min, a_max, K1, K2)
        print(f"    {label}: {f.n_bonds} bonds")
    print(f"    full ring (file 42): {N_FULL} bonds")
    print(f"    half iy>=cy (file 42): {N_HALF_F42} bonds"
          f" (different filter — NOT in fit)")
    print()

    # Geometry cross-check: angular half vs file 42 half
    f_ang_half = make_arc_vortex(ALPHA, R_LOOP, L, 0, np.pi, K1, K2)
    print(f"  Geometry check: angular half = {f_ang_half.n_bonds} bonds,"
          f" file 42 half (iy>=cy) = {N_HALF_F42} bonds")
    print(f"  Difference = {N_HALF_F42 - f_ang_half.n_bonds} bonds"
          f" (diameter bonds at ix<cx have angle=π, excluded by angle<π)")
    print()

    gamma = make_damping_3d(L, DW, DS)
    iz_s, iy_s, ix_s = make_sphere_points(L, r_m, thetas, phis)

    # References (computed once)
    def force_plain(ux, uy, uz):
        return scalar_laplacian_3d(ux, uy, uz, K1, K2)

    print("Computing references...")
    t1 = time.time()
    refs = {}
    for k0 in k_vals:
        ux0, vx0 = make_wave_packet_3d(L, k0, DW + 5, sx, K1, K2)
        ns = estimate_n_steps_3d(k0, L, DW + 5, sx, r_m, DT, 50, K1, K2)
        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma, DT, ns,
                          rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
        refs[k0] = (ref, ux0, vx0, ns)
    print(f"  Done ({time.time()-t1:.0f}s)")

    # Run all arcs
    arc_results = []

    for label, a_min, a_max in ARCS:
        f_test = make_arc_vortex(ALPHA, R_LOOP, L, a_min, a_max, K1, K2)
        n = f_test.n_bonds
        print(f"\n{label}  ({n} bonds)")

        t1 = time.time()
        sigma_tr = np.zeros(len(k_vals))
        for j, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_test, ux0.copy(), vx0.copy(), gamma, DT, ns,
                            rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
            f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                   ref['ux'], ref['uy'], ref['uz'], r_m)
            _, st = integrate_sigma_3d(f2, thetas, phis)
            sigma_tr[j] = max(st, 0.0)
        dt = time.time() - t1

        integrand = np.sin(k_vals)**2 * sigma_tr
        cv = np.std(integrand) / np.mean(integrand) * 100

        arc_results.append({'label': label, 'n_bonds': n,
                            'sigma_tr': sigma_tr.copy(), 'cv': cv})

        print(f"  σ_tr: " + "  ".join(f"{s:.4f}" for s in sigma_tr))
        print(f"  sin²(k)·σ_tr CV = {cv:.1f}%  ({dt:.0f}s)")

    # ═══════════════════════════════════════════════════════════════════
    # Symmetry test: Q1 (front) vs Q3 (back)
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("Symmetry test: Q1 (front, faces wave) vs Q3 (back, away from wave)")
    print("  If σ(Q1)/σ(Q3) ≈ N(Q1)/N(Q3) → isotropic, incoherent")
    print("=" * 75)

    r_q1 = [r for r in arc_results if "Q1" in r['label']][0]
    r_q3 = [r for r in arc_results if "Q3" in r['label']][0]

    print(f"\n  {'k':>5s}  {'σ(Q1)':>8s}  {'σ(Q3)':>8s}  {'Q1/Q3':>7s}")
    ratios_sym = []
    for j, k0 in enumerate(k_vals):
        r = r_q1['sigma_tr'][j] / r_q3['sigma_tr'][j] if r_q3['sigma_tr'][j] > 0 else 0
        ratios_sym.append(r)
        print(f"  {k0:5.1f}  {r_q1['sigma_tr'][j]:8.4f}  {r_q3['sigma_tr'][j]:8.4f}"
              f"  {r:7.4f}")

    mean_sym = np.mean(ratios_sym)
    n_ratio = r_q1['n_bonds'] / r_q3['n_bonds']
    print(f"\n  Mean Q1/Q3 = {mean_sym:.4f}")
    print(f"  N(Q1)/N(Q3) = {n_ratio:.3f}")
    if abs(mean_sym - n_ratio) < 0.15:
        print(f"  → ISOTROPIC: ratio matches N-ratio — incoherent confirmed")
    elif abs(mean_sym - 1.0) < 0.15:
        print(f"  → σ(Q1) ≈ σ(Q3) despite N(Q1)≠N(Q3) — position-dependent")
    else:
        print(f"  → Anisotropic: ratio differs from N-ratio and from 1.0")

    # ═══════════════════════════════════════════════════════════════════
    # Scaling analysis: log(σ) vs log(N)
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("Scaling analysis: log(σ) vs log(N)")
    print("  p_arc = 1.0 → incoherent.  p_arc = 1.5 → stationary phase."
          "  p_arc = 2.0 → coherent.")
    print("  NOTE: p_arc (bond removal at fixed R) ≠ p_R (varying ring radius).")
    print("=" * 75)

    # Collect all data: arcs + full ring
    all_N = [r['n_bonds'] for r in arc_results]
    all_sigma = [r['sigma_tr'] for r in arc_results]
    all_labels = [r['label'] for r in arc_results]

    all_N.append(N_FULL)
    all_sigma.append(sigma_full)
    all_labels.append("full ring (f42)")

    # Sort by N
    order = np.argsort(all_N)
    all_N = [all_N[i] for i in order]
    all_sigma = [all_sigma[i] for i in order]
    all_labels = [all_labels[i] for i in order]

    # Print ratio table
    print(f"\n  {'label':>25s}  {'N':>4s}  {'N/81':>6s}", end="")
    for k0 in k_vals:
        print(f"  {'σ/σ_full('+f'{k0:.1f}'+')':>12s}", end="")
    print()

    for idx in range(len(all_N)):
        N = all_N[idx]
        s = all_sigma[idx]
        ratio = s / sigma_full
        print(f"  {all_labels[idx]:>25s}  {N:4d}  {N/81:6.3f}", end="")
        for r in ratio:
            print(f"  {r:12.4f}", end="")
        print()

    # File 42 half for comparison (NOT in fit — different geometry)
    ratio_f42 = sigma_half_f42 / sigma_full
    print(f"  {'half iy>=cy (f42, ref)':>25s}  {N_HALF_F42:4d}"
          f"  {N_HALF_F42/81:6.3f}", end="")
    for r in ratio_f42:
        print(f"  {r:12.4f}", end="")
    print("  ← NOT in fit")

    # Log-log fit: 6 arc points only (full ring excluded — trivially satisfied)
    fit_N = np.array([r['n_bonds'] for r in arc_results], dtype=float)
    fit_sigma = [r['sigma_tr'] for r in arc_results]
    log_N = np.log(fit_N)
    log_N_full = np.log(81.0)

    print()
    print("=" * 75)
    print("Exponent p: log(σ) = p × log(N) + C  (6 arc points, full ring excluded)")
    print("  Full ring (N=81) predicted from fit as independent check.")
    print("=" * 75)

    print(f"\n  {'k':>5s}  {'p':>7s}  {'R²':>7s}  {'σ(81) pred':>11s}"
          f"  {'σ(81) actual':>12s}  {'pred/actual':>11s}")
    p_vals = []
    for j, k0 in enumerate(k_vals):
        log_s = np.log(np.array([fit_sigma[i][j] for i in range(len(fit_N))]))
        p, C = np.polyfit(log_N, log_s, 1)
        p_vals.append(p)

        # Goodness: R²
        y_pred = p * log_N + C
        ss_res = np.sum((log_s - y_pred)**2)
        ss_tot = np.sum((log_s - np.mean(log_s))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Predict σ at N=81 (independent check)
        sigma_pred = np.exp(p * log_N_full + C)
        print(f"  {k0:5.1f}  {p:7.3f}  {R2:7.4f}  {sigma_pred:11.4f}"
              f"  {sigma_full[j]:12.4f}  {sigma_pred/sigma_full[j]:11.4f}")

    mean_p = np.mean(p_vals)
    std_p = np.std(p_vals)
    print(f"\n  Mean p = {mean_p:.3f} ± {std_p:.3f}")

    if mean_p < 1.25:
        verdict = "INCOHERENT (p ≈ 1)"
    elif mean_p < 1.75:
        verdict = "STATIONARY PHASE (p ≈ 1.5)"
    else:
        verdict = "COHERENT (p ≈ 2)"
    print(f"  → {verdict}")

    # ═══════════════════════════════════════════════════════════════════
    # CV of sin²(k)·σ_tr at each arc size
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("CV of sin²(k)·σ_tr at each arc size")
    print("  If CV constant across N → flatness is per-bond, not collective")
    print("=" * 75)
    print(f"  {'label':>25s}  {'N':>4s}  {'CV%':>6s}")
    for idx in range(len(all_N)):
        s = all_sigma[idx]
        integrand = np.sin(k_vals)**2 * s
        cv = np.std(integrand) / np.mean(integrand) * 100
        print(f"  {all_labels[idx]:>25s}  {all_N[idx]:4d}  {cv:5.1f}%")

    # File 42 half for reference
    integrand_f42 = np.sin(k_vals)**2 * sigma_half_f42
    cv_f42 = np.std(integrand_f42) / np.mean(integrand_f42) * 100
    print(f"  {'half iy>=cy (f42, ref)':>25s}  {N_HALF_F42:4d}  {cv_f42:5.1f}%"
          f"  ← ref only")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
