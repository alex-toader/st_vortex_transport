"""
Route 28: NNN fine α-scan — κ_NNN at k≤1.0 and k≤1.5.

Combined W2 + Q investigation:
  W2: κ_NNN at conservative k≤1.0 cutoff (was "zero compute" but missing data)
  Q:  Fine α scan with NNN gauging, find α_cross(κ=1)

Uses standard 7-point k-grid (k=0.3-1.5), α = 0.10-0.30.
NNN gauging (serial — parallel_fdtd doesn't support gauge_nnn).

Results (R=5, L=80, r_m=20, 681s):

  σ_tr NNN (all α, all k):

    k       α=0.10   α=0.15   α=0.20   α=0.25   α=0.30
   0.30      3.645   10.495   22.017   37.313   54.255
   0.50      2.445    5.928   11.183   17.938   25.436
   0.70      1.865    4.409    8.142   12.868   18.081
   0.90      1.526    3.551    6.475   10.157   14.234
   1.10      1.346    3.113    5.639    8.790   12.243
   1.30      1.243    2.835    5.066    7.807   10.786
   1.50      1.181    2.678    4.753    7.272    9.978

  σ_tr NN (all α, all k):

    k       α=0.10   α=0.15   α=0.20   α=0.25   α=0.30
   0.30      2.471    8.747   18.676   30.051   40.743
   0.50      0.673    2.342    5.422    9.607   14.158
   0.70      0.371    1.185    2.726    4.990    7.694
   0.90      0.272    0.798    1.770    3.228    5.048
   1.10      0.235    0.639    1.357    2.424    3.776
   1.30      0.228    0.586    1.185    2.051    3.138
   1.50      0.231    0.569    1.109    1.868    2.806

  NNN/NN ratio:

    k       α=0.10   α=0.15   α=0.20   α=0.25   α=0.30
   0.30       1.48     1.20     1.18     1.24     1.33
   0.50       3.64     2.53     2.06     1.87     1.80
   0.70       5.02     3.72     2.99     2.58     2.35
   0.90       5.61     4.45     3.66     3.15     2.82
   1.10       5.74     4.87     4.16     3.63     3.24
   1.30       5.45     4.84     4.28     3.81     3.44
   1.50       5.11     4.71     4.28     3.89     3.56

  κ(α) — NN vs NNN:

    Cutoff k≤0.9 (= k≤1.0 on this grid):
      α     κ_NN   κ_NNN  ratio
     0.10   0.025   0.099  3.93
     0.15   0.084   0.240  2.86
     0.20   0.189   0.451  2.38
     0.25   0.333   0.720  2.16
     0.30   0.495   1.018  2.05

    Cutoff k≤1.5:
      α     κ_NN   κ_NNN  ratio
     0.10   0.056   0.266  4.78
     0.15   0.164   0.621  3.79
     0.20   0.355   1.136  3.20
     0.25   0.625   1.782  2.85
     0.30   0.944   2.490  2.64

  κ = 1 crossing:
    NN  (k≤0.9): κ < 1 at all α (max κ=0.495)
    NNN (k≤0.9): κ = 1 at α ≈ 0.297
    NN  (k≤1.5): κ < 1 at all α (max κ=0.944)
    NNN (k≤1.5): κ = 1 at α ≈ 0.187

  Consistency vs file 23:
    α=0.25: κ_NNN = 1.782 (file 23: 1.782, diff 0.0%) — OK
    α=0.30: κ_NNN = 2.490 (file 23: 2.490, diff 0.0%) — OK

  Flat integrand CV (sin²(k)·σ_tr):
      α    CV_NN  CV_NNN
     0.10  15.3%   34.9%
     0.15  15.9%   31.0%
     0.20  15.5%   27.7%
     0.25  11.9%   25.5%
     0.30   7.5%   24.2%

  KEY FINDINGS:
  1. κ_NNN = 1 at α ≈ 0.30 (conservative k≤0.9) or α ≈ 0.19 (k≤1.5).
  2. Flat integrand is NN-SPECIFIC. With NNN, CV = 24-35% at all α.
     NNN adds more scattering at high k (ratio 3-6×) than at low k (1.2-1.5×),
     destroying the sin²(k)·σ_tr flatness.
  3. Consistency with file 23: 0.0% difference — perfect.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/28_nnn_fine_scan.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../src/1_foam'))
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

k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
alpha_vals = [0.10, 0.15, 0.20, 0.25, 0.30]
K_CUTOFFS = [0.9, 1.5]

if __name__ == '__main__':
    t0 = time.time()

    print("Route 28: NNN fine α-scan (W2 + Q)")
    print(f"  k-grid: {len(k_vals)} pts, k = {k_vals[0]:.1f} to {k_vals[-1]:.1f}")
    print(f"  α values: {alpha_vals}")
    print(f"  R={R_LOOP}, L={L}, r_m={r_m}")
    print(f"  Cutoffs: k≤{K_CUTOFFS}")
    print()

    # References (shared, parallel)
    print(f"Computing {len(k_vals)} references...")
    t1 = time.time()
    refs = compute_references(k_vals, L, DW, DS, DT, sx, r_m,
                              thetas, phis, K1, K2, n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")
    print()

    prefactor = N_POL * R_LOOP / (4 * np.pi**2)

    # NN scan (parallel, for comparison)
    print("── NN-only scan (parallel) ──")
    nn_results = []
    for alpha in alpha_vals:
        t1 = time.time()
        sigma_tr = compute_scattering(k_vals, refs, alpha, R_LOOP, L, DW, DS, DT,
                                      r_m, thetas, phis, K1, K2,
                                      n_workers=N_WORKERS)
        dt = time.time() - t1
        nn_results.append({'alpha': alpha, 'sigma_tr': sigma_tr.copy(), 'dt': dt})
        print(f"  α={alpha:.2f}: done ({dt:.0f}s)")

    # NNN scan (serial)
    print()
    print("── NNN scan (serial) ──")
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

    # ── Raw σ_tr tables ──
    print()
    print("=" * 70)
    print("σ_tr NNN (all α, all k)")
    print("=" * 70)
    header = f"  {'k':>5s}"
    for alpha in alpha_vals:
        header += f"  {'α='+f'{alpha:.2f}':>10s}"
    print(header)
    print(f"  {'-'*60}")
    for j, k0 in enumerate(k_vals):
        line = f"  {k0:5.2f}"
        for r in nnn_results:
            line += f"  {r['sigma_tr'][j]:10.3f}"
        print(line)

    print()
    print("σ_tr NN (all α, all k)")
    print("=" * 70)
    header = f"  {'k':>5s}"
    for alpha in alpha_vals:
        header += f"  {'α='+f'{alpha:.2f}':>10s}"
    print(header)
    print(f"  {'-'*60}")
    for j, k0 in enumerate(k_vals):
        line = f"  {k0:5.2f}"
        for r in nn_results:
            line += f"  {r['sigma_tr'][j]:10.3f}"
        print(line)

    # NNN/NN ratio table
    print()
    print("NNN/NN ratio (all α, all k)")
    print("=" * 70)
    header = f"  {'k':>5s}"
    for alpha in alpha_vals:
        header += f"  {'α='+f'{alpha:.2f}':>10s}"
    print(header)
    print(f"  {'-'*60}")
    for j, k0 in enumerate(k_vals):
        line = f"  {k0:5.2f}"
        for i in range(len(alpha_vals)):
            ratio = nnn_results[i]['sigma_tr'][j] / nn_results[i]['sigma_tr'][j]
            line += f"  {ratio:10.2f}"
        print(line)

    # ── κ at each cutoff ──
    print()
    print("=" * 70)
    print("κ(α) at different cutoffs — NN vs NNN")
    print("=" * 70)

    print(f"  NOTE: k≤0.9 and k≤1.0 are identical on this grid (no k=1.0 point).")
    for kc in K_CUTOFFS:
        mask = k_vals <= kc + 0.01
        print(f"\n  Cutoff k≤{kc}:")
        print(f"  {'α':>5s}  {'κ_NN':>7s}  {'κ_NNN':>7s}  {'ratio':>6s}")
        print(f"  {'-'*35}")
        for i, alpha in enumerate(alpha_vals):
            I_nn = np.sin(k_vals[mask])**2 * nn_results[i]['sigma_tr'][mask]
            I_nnn = np.sin(k_vals[mask])**2 * nnn_results[i]['sigma_tr'][mask]
            k_nn = prefactor * np.trapz(I_nn, k_vals[mask])
            k_nnn = prefactor * np.trapz(I_nnn, k_vals[mask])
            ratio = k_nnn / k_nn
            print(f"  {alpha:5.2f}  {k_nn:7.3f}  {k_nnn:7.3f}  {ratio:6.2f}")

    # ── κ=1 crossing ──
    print()
    print("=" * 70)
    print("κ = 1 crossing (linear interpolation)")
    print("=" * 70)

    for kc in K_CUTOFFS:
        mask = k_vals <= kc + 0.01
        kappas_nn = []
        kappas_nnn = []
        for i, alpha in enumerate(alpha_vals):
            I_nn = np.sin(k_vals[mask])**2 * nn_results[i]['sigma_tr'][mask]
            I_nnn = np.sin(k_vals[mask])**2 * nnn_results[i]['sigma_tr'][mask]
            kappas_nn.append(prefactor * np.trapz(I_nn, k_vals[mask]))
            kappas_nnn.append(prefactor * np.trapz(I_nnn, k_vals[mask]))

        for label, kappas in [("NN", kappas_nn), ("NNN", kappas_nnn)]:
            found = False
            for i in range(len(kappas) - 1):
                if kappas[i] <= 1.0 <= kappas[i + 1]:
                    frac = (1.0 - kappas[i]) / (kappas[i + 1] - kappas[i])
                    ac = alpha_vals[i] + frac * (alpha_vals[i + 1] - alpha_vals[i])
                    print(f"  {label} (k≤{kc}): κ = 1 at α ≈ {ac:.3f}")
                    found = True
            if not found:
                if min(kappas) > 1.0:
                    imin = int(np.argmin(kappas))
                    # Extrapolate below smallest α
                    slope = (kappas[1] - kappas[0]) / (alpha_vals[1] - alpha_vals[0])
                    if slope > 0:
                        ac_ext = alpha_vals[0] - (kappas[0] - 1.0) / slope
                        print(f"  {label} (k≤{kc}): κ > 1 at all α "
                              f"(min κ={kappas[imin]:.3f} at α={alpha_vals[imin]}). "
                              f"Extrapolated crossing: α ≈ {ac_ext:.3f}")
                    else:
                        print(f"  {label} (k≤{kc}): κ > 1 at all α "
                              f"(min κ={kappas[imin]:.3f} at α={alpha_vals[imin]})")
                else:
                    imax = int(np.argmax(kappas))
                    print(f"  {label} (k≤{kc}): κ < 1 at all α "
                          f"(max κ={kappas[imax]:.3f} at α={alpha_vals[imax]})")

    # ── Consistency check vs file 23 ──
    print()
    print("=" * 70)
    print("Consistency vs file 23 (α=0.25 and 0.30 at k≤1.5)")
    print("=" * 70)
    ref_23 = {0.25: 1.782, 0.30: 2.490}
    mask15 = k_vals <= 1.51
    for alpha, kref in ref_23.items():
        idx = alpha_vals.index(alpha)
        I = np.sin(k_vals[mask15])**2 * nnn_results[idx]['sigma_tr'][mask15]
        k_new = prefactor * np.trapz(I, k_vals[mask15])
        diff = abs(k_new - kref) / kref * 100
        status = "OK" if diff < 5 else "INVESTIGATE"
        print(f"  α={alpha}: κ_NNN = {k_new:.3f} (file 23: {kref:.3f}, diff {diff:.1f}%) — {status}")
    print()
    print(f"  NOTE: file 23 uses 13 k-pts (0.3-3.0), this uses 7 k-pts (0.3-1.5).")
    print(f"  Trapezoidal integration on different grids may differ slightly.")

    # ── Flat integrand at each α (NN vs NNN) ──
    print()
    print("=" * 70)
    print("Flat integrand test: CV of sin²(k)·σ_tr — NN vs NNN")
    print("=" * 70)
    print(f"  {'α':>5s}  {'CV_NN':>7s}  {'CV_NNN':>8s}")
    print(f"  {'-'*25}")
    for i in range(len(alpha_vals)):
        I_nn = np.sin(k_vals)**2 * nn_results[i]['sigma_tr']
        I_nnn = np.sin(k_vals)**2 * nnn_results[i]['sigma_tr']
        cv_nn = np.std(I_nn) / np.mean(I_nn) * 100
        cv_nnn = np.std(I_nnn) / np.mean(I_nnn) * 100
        print(f"  {alpha_vals[i]:5.2f}  {cv_nn:6.1f}%  {cv_nnn:7.1f}%")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
