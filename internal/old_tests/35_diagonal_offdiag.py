"""
Route 35: Diagonal vs off-diagonal decomposition of Peierls correction.

Decomposes R(2πα) - I into:
  diagonal:    cm1·I  = (cos(2πα)-1)·I    — scalar-like (same component)
  off-diagonal: s·J  = sin(2πα)·[[0,-1],[1,0]]  — rotation/mixing

Physical argument (reviewer):
  - Diagonal cm1·I acts like altered spring constant (couples to strain, ∝k)
    → expect CV >> 10% (like mass sphere)
  - Off-diagonal s·J mixes ux↔uy (couples to displacement, no k factor)
    → expect CV ~ 7% (like full Peierls)

Tests α=0.3, R=5, NN only, k=0.3-1.5 (7 pts), standard grid.

Results (L=80, r_m=20, 299s):

  cos(2πα)-1 = -1.309,  sin(2πα) = 0.951,  |cm1|/|s| = 1.38

  sin²(k)·σ_tr:
    mode         CV%     verdict    σ_tr(0.3)  σ_tr(1.5)  ratio
    full          7.5%   FLAT         40.74      2.81     14.5
    diagonal      8.8%   FLAT         45.23      3.00     15.1
    offdiag      23.7%   NOT FLAT      4.11      0.57      7.2

  Superposition (σ_diag + σ_offdiag vs σ_full):
    k    full    diag   offdiag    sum     Δ%
   0.3  40.74   45.23    4.11    49.34   +21%
   0.5  14.16   16.27    1.24    17.51   +24%
   0.7   7.69    8.95    0.75     9.71   +26%
   0.9   5.05    5.83    0.59     6.42   +27%
   1.1   3.78    4.29    0.54     4.82   +28%
   1.3   3.14    3.46    0.55     4.00   +28%
   1.5   2.81    3.00    0.57     3.57   +27%

  CONCLUSION:

    Reviewer's hypothesis was WRONG. The DIAGONAL part (cm1·I) is flat
    (CV=8.8%), not the off-diagonal rotation part (CV=23.7%).

    The diagonal Peierls correction is NOT the same as a mass sphere:
    - Mass sphere: changes K1→K1' on bond, modifies K1*(u_nbr - u_site)
      → both coupling AND self-energy change → strain coupling → NOT flat
    - Diagonal Peierls: adds K1*cm1*u_neighbor only (no self-energy change)
      → effective coupling K1*(1+cm1) = K1*cos(2πα) but self-energy unchanged
      → displacement coupling (force ∝ u_neighbor, not ∝ Δu) → FLAT

    At α=0.3: K1*cos(2πα) = -0.31 (effective coupling is NEGATIVE).
    Mass sphere tests were K1_inside ≥ 0.

    The diagonal part dominates: σ_tr(diag)/σ_tr(offdiag) = 5-11×,
    despite |cm1|/|s_phi| = 1.38 (only 38% stronger amplitude).
    The off-diagonal rotation is a weak scatterer.

    Cross-term 2·Re⟨f_diag|f_offdiag⟩ is NEGATIVE and large (~25%).
    The two components destructively interfere.

    The flat integrand mechanism is NOT about polarization rotation/mixing.
    It's about the asymmetric neighbor coupling structure of Peierls gauge:
    force ∝ K*u_neighbor (displacement) rather than K*Δu (strain).

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/35_diagonal_offdiag.py
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


def make_decomposed_force(alpha, R_loop, L, mode, K1=1.0, K2=0.5):
    """Force function with decomposed Peierls correction.

    mode = "full"     : standard Peierls (cm1 + s_phi terms)
    mode = "diagonal" : only cm1·I part (scalar-like)
    mode = "offdiag"  : only s_phi·J part (rotation/mixing)
    """
    assert mode in ("full", "diagonal", "offdiag")

    cz = L // 2
    iy_disk, ix_disk = precompute_disk_bonds(L, R_loop)

    phi = 2.0 * np.pi * (float(alpha) % 1.0)
    cm1 = np.cos(phi) - 1.0
    s_phi = np.sin(phi)
    if abs(s_phi) < 1e-12:
        s_phi = 0.0
    if abs(cm1) < 1e-12:
        cm1 = 0.0

    iz_lo = cz - 1
    iz_hi = cz

    use_diag = mode in ("full", "diagonal")
    use_offdiag = mode in ("full", "offdiag")

    n_bonds = len(iy_disk)

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if cm1 == 0.0 and s_phi == 0.0:
            return fx, fy, fz

        ux_hi = ux[iz_hi, iy_disk, ix_disk]
        uy_hi = uy[iz_hi, iy_disk, ix_disk]
        ux_lo = ux[iz_lo, iy_disk, ix_disk]
        uy_lo = uy[iz_lo, iy_disk, ix_disk]

        # Lower site: (R - I) @ u_upper
        #   diagonal:    cm1*ux_hi,  cm1*uy_hi
        #   off-diagonal: -s_phi*uy_hi,  s_phi*ux_hi
        if use_diag:
            fx[iz_lo, iy_disk, ix_disk] += K1 * cm1 * ux_hi
            fy[iz_lo, iy_disk, ix_disk] += K1 * cm1 * uy_hi
        if use_offdiag:
            fx[iz_lo, iy_disk, ix_disk] += K1 * (-s_phi) * uy_hi
            fy[iz_lo, iy_disk, ix_disk] += K1 * s_phi * ux_hi

        # Upper site: (R^T - I) @ u_lower
        #   diagonal:    cm1*ux_lo,  cm1*uy_lo
        #   off-diagonal: s_phi*uy_lo,  -s_phi*ux_lo
        if use_diag:
            fx[iz_hi, iy_disk, ix_disk] += K1 * cm1 * ux_lo
            fy[iz_hi, iy_disk, ix_disk] += K1 * cm1 * uy_lo
        if use_offdiag:
            fx[iz_hi, iy_disk, ix_disk] += K1 * s_phi * uy_lo
            fy[iz_hi, iy_disk, ix_disk] += K1 * (-s_phi) * ux_lo

        return fx, fy, fz

    force_fn.n_bonds = n_bonds
    force_fn.mode = mode
    return force_fn


if __name__ == '__main__':
    t0 = time.time()

    phi = 2.0 * np.pi * ALPHA
    cm1 = np.cos(phi) - 1.0
    s_phi = np.sin(phi)

    print("Route 35: Diagonal vs off-diagonal Peierls decomposition")
    print(f"  α={ALPHA}, R={R_LOOP}, L={L}")
    print(f"  cos(2πα)-1 = {cm1:.6f}")
    print(f"  sin(2πα)   = {s_phi:.6f}")
    print(f"  |cm1|/|s|  = {abs(cm1)/abs(s_phi):.3f}")
    print(f"  k = {list(k_vals)}")
    print()

    gamma = make_damping_3d(L, DW, DS)
    iz_s, iy_s, ix_s = make_sphere_points(L, r_m, thetas, phis)

    # References (plain lattice)
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

    # Three modes
    modes = ["full", "diagonal", "offdiag"]
    all_results = {}

    for mode in modes:
        f_dec = make_decomposed_force(ALPHA, R_LOOP, L, mode, K1, K2)
        print(f"\nMode: {mode}  ({f_dec.n_bonds} bonds)")

        t1 = time.time()
        sigma_tr = np.zeros(len(k_vals))
        for j, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_dec, ux0.copy(), vx0.copy(), gamma, DT, ns,
                            rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
            f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                   ref['ux'], ref['uy'], ref['uz'], r_m)
            _, st = integrate_sigma_3d(f2, thetas, phis)
            assert st > 0, f"σ_tr={st} at k={k0}, mode={mode}"
            sigma_tr[j] = st

        integrand = np.sin(k_vals)**2 * sigma_tr
        cv = np.std(integrand) / np.mean(integrand) * 100
        dt = time.time() - t1

        all_results[mode] = {
            'sigma_tr': sigma_tr.copy(),
            'integrand': integrand.copy(),
            'cv': cv,
            'dt': dt,
        }

        print(f"  σ_tr: " + "  ".join(f"{s:.4f}" for s in sigma_tr))
        print(f"  sin²(k)·σ_tr: " + "  ".join(f"{v:.4f}" for v in integrand))
        print(f"  CV = {cv:.1f}%  ({dt:.0f}s)")

    # Superposition check: diagonal + offdiag ≈ full?
    print()
    print("=" * 70)
    print("Superposition check: σ_tr(diag) + σ_tr(offdiag) vs σ_tr(full)")
    print("=" * 70)
    s_full = all_results['full']['sigma_tr']
    s_diag = all_results['diagonal']['sigma_tr']
    s_offdiag = all_results['offdiag']['sigma_tr']
    s_sum = s_diag + s_offdiag
    print(f"  {'k':>5s}  {'full':>8s}  {'diag':>8s}  {'offdiag':>8s}  "
          f"{'sum':>8s}  {'Δ%':>6s}")
    print(f"  {'-'*50}")
    for j, k0 in enumerate(k_vals):
        diff_pct = (s_sum[j] - s_full[j]) / s_full[j] * 100
        print(f"  {k0:5.1f}  {s_full[j]:8.4f}  {s_diag[j]:8.4f}  "
              f"{s_offdiag[j]:8.4f}  {s_sum[j]:8.4f}  {diff_pct:+6.1f}%")
    print()
    print("  NOTE: σ_tr is NOT linear in perturbation (involves |scattered|²),")
    print("  so σ_tr(diag) + σ_tr(offdiag) ≠ σ_tr(full) is expected.")
    print("  Cross term 2·Re⟨f_diag | f_offdiag⟩ is the difference.")

    # Summary
    print()
    print("=" * 70)
    print("Summary: flat integrand mechanism")
    print("=" * 70)
    print(f"  {'mode':>12s}  {'CV%':>6s}  {'verdict':>10s}  {'σ_tr(0.3)':>10s}  "
          f"{'σ_tr(1.5)':>10s}  {'ratio':>6s}")
    print(f"  {'-'*60}")
    for mode in modes:
        r = all_results[mode]
        cv = r['cv']
        s03 = r['sigma_tr'][0]
        s15 = r['sigma_tr'][-1]
        ratio = s03 / s15
        if cv < 10:
            verdict = "FLAT"
        elif cv < 20:
            verdict = "marginal"
        else:
            verdict = "NOT FLAT"
        print(f"  {mode:>12s}  {cv:5.1f}%  {verdict:>10s}  {s03:10.4f}  "
              f"{s15:10.4f}  {ratio:6.2f}")

    # Reference from file 28 / 34
    print()
    print("  Reference (from previous files):")
    print("    full Peierls ring (file 28):  CV = 7.5%   FLAT")
    print("    mass sphere (file 34):        CV = 89-152% NOT FLAT")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
