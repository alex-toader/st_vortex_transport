"""
Route 25: α-scan with NNN gauging (k=0.3-3.0, 13 pts).

Compares κ(α) with NN-only vs NN+NNN gauging.
Same k-grid as Route 24. NN scan fork(4), NNN scan serial. 34 min total.

Results (R=5, L=80, 13 k-pts, 4 α values):

  NNN gauging multiplies σ_tr by 1.3× (k=0.3) to 3.6× (k=1.5-1.7).
  Effect peaks at k≈1.5-1.7, then decreases toward BZ edge.

  σ_tr: NN vs NNN (α=0.30):
    k     σ_NN   σ_NNN   Δ%
   0.30  40.74   54.26   +33%
   0.70   7.69   18.08  +135%
   1.30   3.14   10.79  +244%
   1.50   2.81    9.98  +256%
   2.10   3.34    9.81  +194%
   3.00  22.45   31.35   +40%

  κ(α): NN vs NNN at cutoff k≤1.5:
    α     κ_NN   κ_NNN   Δ%
   0.25   0.625   1.782  +185%
   0.30   0.944   2.490  +164%
   0.40   1.545   3.722  +141%
   0.50   1.798   4.205  +134%

  κ(α): NN vs NNN at cutoff k≤2.1:
    α     κ_NN   κ_NNN   Δ%
   0.25   0.898   2.783  +210%
   0.30   1.341   3.845  +187%
   0.40   2.188   5.650  +158%
   0.50   2.552   6.339  +148%

  NNN gauging increases κ by 2.3-3.1× across all α and cutoffs.
  No κ=1 crossing found — κ_NNN > 1 at all tested α (even α=0.25).

  Timing: NN scan uses fork(4), NNN scan is serial (parallel_fdtd
  doesn't support gauge_nnn). Per-α timing not directly comparable.

  Prerequisite: Test C (24_test_alpha_zero.py) confirms σ_tr=0 at α=0
  for all k — high-k data is not a pipeline artifact.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/23_nnn_alpha_scan.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from parallel_fdtd import compute_references, compute_scattering
from gauge_3d import make_vortex_force
from elastic_3d import make_damping_3d, run_fdtd_3d
from scattering_3d import (make_sphere_points, compute_sphere_f2,
                           integrate_sigma_3d)

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

k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.4, 2.7, 3.0])
K_CUTOFFS = [1.5, 2.1, 3.0]
alpha_vals = [0.25, 0.30, 0.40, 0.50]

if __name__ == '__main__':
    t0 = time.time()

    print(f"Route 25: NNN α-scan")
    print(f"  k-grid: {len(k_vals)} pts, k = {k_vals[0]:.1f} to {k_vals[-1]:.1f}")
    print(f"  α values: {alpha_vals}")
    print(f"  R={R_LOOP}, L={L}, workers={N_WORKERS}")
    print()

    # References (shared, no gauging needed)
    print(f"Computing {len(k_vals)} references...")
    t1 = time.time()
    refs = compute_references(k_vals, L, DW, DS, DT, sx, r_m,
                              thetas, phis, K1, K2, n_workers=N_WORKERS)
    t_ref = time.time() - t1
    print(f"  Done ({t_ref:.0f}s)")
    print()

    # NN scan (reuse parallel_fdtd)
    prefactor = N_POL * R_LOOP / (4 * np.pi**2)

    print("── NN-only scan ──")
    nn_results = []
    for alpha in alpha_vals:
        t1 = time.time()
        sigma_tr = compute_scattering(k_vals, refs, alpha, R_LOOP, L, DW, DS, DT,
                                      r_m, thetas, phis, K1, K2,
                                      n_workers=N_WORKERS)
        dt = time.time() - t1
        nn_results.append({'alpha': alpha, 'sigma_tr': sigma_tr.copy(), 'dt': dt})
        print(f"  α={alpha:.2f}: done ({dt:.0f}s)")

    # NNN scan (manual loop — parallel_fdtd doesn't support gauge_nnn)
    # NOTE: NNN runs serial, NN runs fork(4). Per-α timing not comparable.
    print()
    print("── NN+NNN scan (serial — no fork support) ──")
    iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
    gamma = make_damping_3d(L, DW, DS)

    nnn_results = []
    for alpha in alpha_vals:
        t1 = time.time()
        f_nnn = make_vortex_force(alpha, R_LOOP, L, K1, K2, gauge_nnn=True)
        sigma_tr = np.zeros(len(k_vals))
        for i, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_nnn, ux0.copy(), vx0.copy(), gamma, DT, ns,
                            rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
            f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                   ref['ux'], ref['uy'], ref['uz'], r_m)
            _, st = integrate_sigma_3d(f2, thetas, phis)
            sigma_tr[i] = st
        dt = time.time() - t1
        nnn_results.append({'alpha': alpha, 'sigma_tr': sigma_tr.copy(), 'dt': dt})
        print(f"  α={alpha:.2f}: done ({dt:.0f}s)")

    # σ_tr comparison table
    print()
    print("=" * 80)
    print("σ_tr: NN vs NNN (α=0.30)")
    print("=" * 80)
    idx30 = alpha_vals.index(0.30)
    print(f"{'k':>5s}  {'σ_NN':>8s}  {'σ_NNN':>8s}  {'ratio':>6s}  {'Δ%':>6s}")
    print("-" * 40)
    for j, k0 in enumerate(k_vals):
        snn = nn_results[idx30]['sigma_tr'][j]
        snnn = nnn_results[idx30]['sigma_tr'][j]
        ratio = snnn / snn
        diff = (snnn - snn) / snn * 100
        print(f"{k0:5.2f}  {snn:8.3f}  {snnn:8.3f}  {ratio:6.3f}  {diff:+5.0f}%")

    # κ comparison at cutoffs
    print()
    print("=" * 80)
    print("κ(α): NN vs NNN at different cutoffs")
    print("=" * 80)

    for kc in K_CUTOFFS:
        mask = k_vals <= kc + 0.01
        print(f"\n  Cutoff k≤{kc}:")
        print(f"  {'α':>5s}  {'κ_NN':>7s}  {'κ_NNN':>7s}  {'ratio':>6s}  {'Δ%':>6s}")
        print(f"  {'-'*40}")
        for i, alpha in enumerate(alpha_vals):
            I_nn = np.sin(k_vals[mask])**2 * nn_results[i]['sigma_tr'][mask]
            I_nnn = np.sin(k_vals[mask])**2 * nnn_results[i]['sigma_tr'][mask]
            k_nn = prefactor * np.trapz(I_nn, k_vals[mask])
            k_nnn = prefactor * np.trapz(I_nnn, k_vals[mask])
            ratio = k_nnn / k_nn
            diff = (k_nnn - k_nn) / k_nn * 100
            print(f"  {alpha:5.2f}  {k_nn:7.3f}  {k_nnn:7.3f}  {ratio:6.3f}  {diff:+5.0f}%")

    # κ=1 crossing with NNN
    print()
    for kc in K_CUTOFFS:
        mask = k_vals <= kc + 0.01
        kappas = []
        for i, alpha in enumerate(alpha_vals):
            I = np.sin(k_vals[mask])**2 * nnn_results[i]['sigma_tr'][mask]
            kappas.append(prefactor * np.trapz(I, k_vals[mask]))
        found_crossing = False
        for i in range(len(kappas) - 1):
            if kappas[i] <= 1.0 <= kappas[i + 1]:
                frac = (1.0 - kappas[i]) / (kappas[i + 1] - kappas[i])
                ac = alpha_vals[i] + frac * (alpha_vals[i + 1] - alpha_vals[i])
                print(f"NNN: κ = 1.0 at α ≈ {ac:.3f} (cutoff k≤{kc})")
                found_crossing = True
        if not found_crossing:
            if min(kappas) > 1.0:
                imin = int(np.argmin(kappas))
                print(f"NNN: κ > 1 at all tested α (cutoff k≤{kc}), "
                      f"min κ={kappas[imin]:.3f} at α={alpha_vals[imin]}")
            else:
                print(f"NNN: κ < 1 at all tested α (cutoff k≤{kc}), "
                      f"max κ={max(kappas):.3f}")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
