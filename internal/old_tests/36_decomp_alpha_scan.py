"""
Route 36 (F3b): Diagonal/off-diagonal decomposition at multiple α.

File 35 showed at α=0.30: diagonal is FLAT (CV=8.8%), offdiag NOT FLAT (CV=23.7%).
Question: does diagonal remain flat at weak coupling (α=0.05, 0.10)?

Tests α = 0.05, 0.10, 0.20, 0.30 (four values), modes = full, diagonal, offdiag.
NN only, R=5, k=0.3-1.5 (7 pts), standard grid.

Results (L=80, r_m=20, 1071s):

  Peierls amplitudes:
    α=0.05: cm1=-0.049, s=0.309, |cm1|/|s|=0.16
    α=0.10: cm1=-0.191, s=0.588, |cm1|/|s|=0.33
    α=0.20: cm1=-0.691, s=0.951, |cm1|/|s|=0.73
    α=0.30: cm1=-1.309, s=0.951, |cm1|/|s|=1.38

  CV(sin²(k)·σ_tr):
     α     full     diag   offdiag   cross/full
    0.05   34.2%   61.9%    47.8%    -27% ± 24%
    0.10   15.3%   50.8%    38.8%    -50% ± 21%
    0.20   15.5%   25.2%    23.7%    -51% ± 5%
    0.30    7.5%    8.8%    23.7%    -26% ± 2%

  σ_diag/σ_offdiag at k=0.5:
    α=0.05: 0.9,  α=0.10: 2.7,  α=0.20: 5.9,  α=0.30: 13.1

  NOTE: σ_offdiag identical at α=0.20 and α=0.30 because
  sin(2π×0.2) = sin(2π×0.3) = 0.951.

  CONCLUSION:

    File 35's finding ("diagonal is flat") is ONLY true at α≥0.30.
    At weak coupling, diagonal is the WORST component (CV=62% at α=0.05).

    The mechanism has two regimes:

    1. Strong coupling (α ≥ 0.25, |cm1| > |s_phi|):
       Diagonal dominates (σ_diag/σ_offdiag > 6). Diagonal is inherently
       flat (displacement coupling → 1/sin²(k)). Off-diagonal is small and
       non-flat but doesn't contaminate the full result much.
       → CV(full) ≈ CV(diag) ≈ 8%.

    2. Weak coupling (α < 0.20, |cm1| < |s_phi|):
       Off-diagonal dominates or is comparable. Neither component is flat.
       The cross-term (interference) produces partial cancellation that
       makes full flatter than either component alone.
       But not flat enough: CV(full) = 15-34%.

    The transition at |cm1|/|s| ≈ 1 (α ≈ 0.25) is the threshold:
    when diagonal displacement coupling dominates the scattering amplitude,
    its natural 1/sin²(k) spectrum produces the flat integrand.

    This explains WHY strong coupling is required for flatness:
    not because of unitarity or non-perturbative physics, but because
    the diagonal component must dominate the off-diagonal component.
    The threshold is geometric: |cos(2πα)-1| > |sin(2πα)|.

    For paper §4.3: the flat integrand is a strong-coupling emergent
    property where the diagonal Peierls correction (displacement coupling)
    dominates the off-diagonal (rotation/mixing). At α ≥ 0.25, the
    displacement coupling naturally produces σ_tr ∝ 1/sin²(k).

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/36_decomp_alpha_scan.py
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

alpha_vals = [0.05, 0.10, 0.20, 0.30]
modes = ["full", "diagonal", "offdiag"]


def make_decomposed_force(alpha, R_loop, L, mode, K1=1.0, K2=0.5):
    """Force function with decomposed Peierls correction.

    mode = "full"     : standard Peierls (cm1 + s_phi terms)
    mode = "diagonal" : only cm1·I part (scalar-like)
    mode = "offdiag"  : only s_phi·J part (rotation/mixing)

    Copied from file 35 to avoid numeric-named module import.
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

        if use_diag:
            fx[iz_lo, iy_disk, ix_disk] += K1 * cm1 * ux_hi
            fy[iz_lo, iy_disk, ix_disk] += K1 * cm1 * uy_hi
        if use_offdiag:
            fx[iz_lo, iy_disk, ix_disk] += K1 * (-s_phi) * uy_hi
            fy[iz_lo, iy_disk, ix_disk] += K1 * s_phi * ux_hi

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

    print("Route 36 (F3b): Decomposition at multiple α")
    print(f"  R={R_LOOP}, L={L}")
    print(f"  α = {alpha_vals}")
    print(f"  k = {list(k_vals)}")
    print()

    # Peierls amplitudes
    for alpha in alpha_vals:
        phi = 2.0 * np.pi * alpha
        cm1 = np.cos(phi) - 1.0
        s_phi = np.sin(phi)
        print(f"  α={alpha:.2f}: cm1={cm1:.4f}, s={s_phi:.4f}, "
              f"|cm1|/|s|={abs(cm1)/abs(s_phi):.3f}")
    print()

    gamma = make_damping_3d(L, DW, DS)
    iz_s, iy_s, ix_s = make_sphere_points(L, r_m, thetas, phis)

    # References (plain lattice, shared across all α)
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

    # Run all (alpha, mode) combinations
    all_results = {}  # key = (alpha, mode)

    for alpha in alpha_vals:
        print(f"\n{'='*70}")
        print(f"α = {alpha:.2f}")
        print(f"{'='*70}")

        for mode in modes:
            f_dec = make_decomposed_force(alpha, R_LOOP, L, mode, K1, K2)

            t1 = time.time()
            sigma_tr = np.zeros(len(k_vals))
            for j, k0 in enumerate(k_vals):
                ref, ux0, vx0, ns = refs[k0]
                d = run_fdtd_3d(f_dec, ux0.copy(), vx0.copy(), gamma, DT, ns,
                                rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
                f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                       ref['ux'], ref['uy'], ref['uz'], r_m)
                _, st = integrate_sigma_3d(f2, thetas, phis)
                assert st > -1e-6, \
                    f"σ_tr={st} at k={k0}, α={alpha}, mode={mode}"
                sigma_tr[j] = max(st, 0.0)
            dt = time.time() - t1

            integrand = np.sin(k_vals)**2 * sigma_tr
            if np.mean(integrand) > 1e-10:
                cv = np.std(integrand) / np.mean(integrand) * 100
            else:
                cv = 0.0

            all_results[(alpha, mode)] = {
                'sigma_tr': sigma_tr.copy(),
                'integrand': integrand.copy(),
                'cv': cv,
                'dt': dt,
            }

            print(f"  {mode:>10s}: CV={cv:5.1f}%  "
                  f"σ_tr=[{sigma_tr[0]:.3f}..{sigma_tr[-1]:.3f}]  ({dt:.0f}s)")

        # SNR check at first alpha (weakest coupling)
        if alpha == alpha_vals[0]:
            print(f"  SNR check (vs noise floor ~1e-10):")
            for mode in modes:
                s_min = all_results[(alpha, mode)]['sigma_tr'].min()
                print(f"    {mode:>10s}: min σ_tr = {s_min:.3e}")

        # Cross-term analysis
        s_full = all_results[(alpha, 'full')]['sigma_tr']
        s_diag = all_results[(alpha, 'diagonal')]['sigma_tr']
        s_offdiag = all_results[(alpha, 'offdiag')]['sigma_tr']
        cross = s_full - s_diag - s_offdiag
        cross_frac = cross / s_full
        print(f"  cross/full: mean={np.mean(cross_frac):.3f} ± "
              f"{np.std(cross_frac):.3f}")

    # Grand summary
    print()
    print("=" * 70)
    print("Grand summary: CV(sin²(k)·σ_tr) by α and mode")
    print("=" * 70)
    print(f"  {'α':>6s}  {'full':>8s}  {'diagonal':>10s}  {'offdiag':>10s}  "
          f"{'cross/full':>10s}")
    print(f"  {'-'*50}")
    for alpha in alpha_vals:
        cv_f = all_results[(alpha, 'full')]['cv']
        cv_d = all_results[(alpha, 'diagonal')]['cv']
        cv_o = all_results[(alpha, 'offdiag')]['cv']
        s_f = all_results[(alpha, 'full')]['sigma_tr']
        s_d = all_results[(alpha, 'diagonal')]['sigma_tr']
        s_o = all_results[(alpha, 'offdiag')]['sigma_tr']
        cross = s_f - s_d - s_o
        cf = np.mean(cross / s_f)
        print(f"  {alpha:6.2f}  {cv_f:7.1f}%  {cv_d:9.1f}%  {cv_o:9.1f}%  "
              f"{cf:+9.1f}%")

    # Verdict
    print()
    for alpha in alpha_vals:
        cv_d = all_results[(alpha, 'diagonal')]['cv']
        cv_f = all_results[(alpha, 'full')]['cv']
        flat_d = "FLAT" if cv_d < 10 else ("marginal" if cv_d < 20 else "NOT FLAT")
        flat_f = "FLAT" if cv_f < 10 else ("marginal" if cv_f < 20 else "NOT FLAT")
        print(f"  α={alpha:.2f}: full={flat_f} (CV={cv_f:.1f}%), "
              f"diagonal={flat_d} (CV={cv_d:.1f}%)")

    # α-scaling of components
    print()
    print("=" * 70)
    print("Component scaling with α")
    print("=" * 70)
    print(f"  {'α':>6s}  {'σ_full(0.5)':>12s}  {'σ_diag(0.5)':>12s}  "
          f"{'σ_offdiag(0.5)':>14s}  {'diag/offdiag':>12s}")
    print(f"  {'-'*60}")
    ik05 = 1  # index of k=0.5
    for alpha in alpha_vals:
        sf = all_results[(alpha, 'full')]['sigma_tr'][ik05]
        sd = all_results[(alpha, 'diagonal')]['sigma_tr'][ik05]
        so = all_results[(alpha, 'offdiag')]['sigma_tr'][ik05]
        ratio = sd / so if so > 1e-10 else float('inf')
        print(f"  {alpha:6.2f}  {sf:12.4f}  {sd:12.4f}  {so:14.4f}  {ratio:12.1f}")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
