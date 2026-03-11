"""
Route 41 (F20): Non-Born ratio from displacement coupling data.

Reruns displacement coupling at K_eff = -0.5, -1.0, -1.3 to get full
σ_tr(k) arrays (7 k-points). Then computes the non-Born ratio:

  R_NB(K_a, K_b, k) = [σ(K_a,k)/σ(K_b,k)] / (K_a/K_b)²

If R_NB ≈ 1 for all k → Born regime, σ ~ K².
If R_NB varies with k → non-Born modifies the spectral shape.

From file 37 endpoints (k=0.3, 1.5):
  K=-0.5 vs K=-1.3: R_NB = 2.46 → 1.30 (decreases with k)
  K=-1.0 vs K=-1.3: R_NB = 1.34 → 1.10 (nearly Born)
Non-Born enhances low-k more than high-k → explains why weak coupling
has higher CV (spectral distortion).

Results:


Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/41_non_born_ratio.py
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

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# Displacement coupling K_eff values to test
K_effs = [-0.5, -1.0, -1.3]


def make_displacement_force(K_eff, R_loop, L, K1=1.0, K2=0.5):
    """Pure displacement coupling on Dirac disk z-bonds.

    Identical to file 37 make_displacement_force.
    Force on lo site += K_eff * u_hi (and symmetric).
    """
    cz = L // 2
    iy_disk, ix_disk = precompute_disk_bonds(L, R_loop)

    iz_lo = cz - 1
    iz_hi = cz

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if K_eff == 0.0:
            return fx, fy, fz

        ux_hi = ux[iz_hi, iy_disk, ix_disk]
        uy_hi = uy[iz_hi, iy_disk, ix_disk]
        ux_lo = ux[iz_lo, iy_disk, ix_disk]
        uy_lo = uy[iz_lo, iy_disk, ix_disk]

        fx[iz_lo, iy_disk, ix_disk] += K_eff * ux_hi
        fy[iz_lo, iy_disk, ix_disk] += K_eff * uy_hi

        fx[iz_hi, iy_disk, ix_disk] += K_eff * ux_lo
        fy[iz_hi, iy_disk, ix_disk] += K_eff * uy_lo

        return fx, fy, fz

    return force_fn


if __name__ == '__main__':
    t0 = time.time()

    print("Route 41 (F20): Non-Born ratio from displacement coupling")
    print(f"  K_eff = {K_effs}")
    print(f"  R={R_LOOP}, L={L}, k = {list(k_vals)}")
    print()

    gamma = make_damping_3d(L, DW, DS)
    iz_s, iy_s, ix_s = make_sphere_points(L, r_m, thetas, phis)

    # References
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

    # Run 3 displacement cases
    sigma_data = {}

    for K_eff in K_effs:
        f_test = make_displacement_force(K_eff, R_LOOP, L, K1, K2)
        print(f"\nK_eff = {K_eff}")

        t1 = time.time()
        sigma_tr = np.zeros(len(k_vals))
        for j, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_test, ux0.copy(), vx0.copy(), gamma, DT, ns,
                            rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
            f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                   ref['ux'], ref['uy'], ref['uz'], r_m)
            _, st = integrate_sigma_3d(f2, thetas, phis)
            assert st > -1e-6, f"σ_tr={st} at k={k0}, K_eff={K_eff}"
            sigma_tr[j] = max(st, 0.0)
        dt = time.time() - t1

        integrand = np.sin(k_vals)**2 * sigma_tr
        cv = np.std(integrand) / np.mean(integrand) * 100

        sigma_data[K_eff] = sigma_tr.copy()

        print(f"  σ_tr: " + "  ".join(f"{s:.4f}" for s in sigma_tr))
        print(f"  CV = {cv:.1f}%  ({dt:.0f}s)")

    # Non-Born ratio analysis
    print()
    print("=" * 75)
    print("Non-Born ratio: R_NB(k) = [σ(K_a)/σ(K_b)] / (K_a/K_b)²")
    print("  R_NB = 1 → Born (σ ~ K²).  R_NB ≠ 1 → non-Born correction.")
    print("=" * 75)

    pairs = [(-0.5, -1.3), (-1.0, -1.3), (-0.5, -1.0)]
    for K_a, K_b in pairs:
        born_ratio = (K_a / K_b)**2
        s_a = sigma_data[K_a]
        s_b = sigma_data[K_b]
        actual_ratio = s_a / s_b
        R_NB = actual_ratio / born_ratio

        print(f"\n  K={K_a:.1f} vs K={K_b:.1f}:  Born ratio = {born_ratio:.4f}")
        print(f"  {'k':>5s}  {'σ(K_a)':>8s}  {'σ(K_b)':>8s}  {'actual':>8s}"
              f"  {'Born':>8s}  {'R_NB':>6s}")
        for j, k0 in enumerate(k_vals):
            print(f"  {k0:5.1f}  {s_a[j]:8.4f}  {s_b[j]:8.4f}"
                  f"  {actual_ratio[j]:8.4f}  {born_ratio:8.4f}"
                  f"  {R_NB[j]:6.3f}")

        cv_rnb = np.std(R_NB) / np.mean(R_NB) * 100
        print(f"  R_NB: mean={np.mean(R_NB):.3f}, CV={cv_rnb:.1f}%,"
              f" range={R_NB.min():.3f}-{R_NB.max():.3f}")

    # Born scaling check: σ/K² should be K-independent at each k
    print()
    print("=" * 75)
    print("Born scaling: σ(k)/K_eff² should be independent of K_eff")
    print("=" * 75)
    print(f"  {'k':>5s}", end="")
    for K_eff in K_effs:
        print(f"  {'σ/K²('+str(K_eff)+')':>12s}", end="")
    print(f"  {'CV%':>6s}  {'Born?':>6s}")

    for j, k0 in enumerate(k_vals):
        vals = [sigma_data[K][j] / K**2 for K in K_effs]
        cv_born = np.std(vals) / np.mean(vals) * 100
        born_ok = "YES" if cv_born < 10 else ("~" if cv_born < 20 else "NO")
        print(f"  {k0:5.1f}", end="")
        for v in vals:
            print(f"  {v:12.4f}", end="")
        print(f"  {cv_born:5.1f}%  {born_ok:>6s}")

    # Spectral shape comparison: σ(k)×sin²(k) normalized
    print()
    print("=" * 75)
    print("Spectral shape: sin²(k)·σ(k) normalized to k=0.3 value")
    print("  If shape independent of K_eff → non-Born is multiplicative")
    print("=" * 75)
    print(f"  {'k':>5s}", end="")
    for K_eff in K_effs:
        print(f"  {'K='+str(K_eff):>10s}", end="")
    print()
    for j, k0 in enumerate(k_vals):
        print(f"  {k0:5.1f}", end="")
        for K_eff in K_effs:
            integrand_j = np.sin(k0)**2 * sigma_data[K_eff][j]
            integrand_0 = np.sin(k_vals[0])**2 * sigma_data[K_eff][0]
            print(f"  {integrand_j/integrand_0:10.4f}", end="")
        print()

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
