"""
Route 50 (I8→I5 hybrid): Diagonal-only (cm1·I) on single bond.

File 49 showed: full Peierls on 1 bond gives CV(sin²·σ)=70.8% (α=0.30).
File 35 showed: diagonal-only on full ring gives CV=8.8% (α=0.30).
Question: is per-bond displacement coupling Born-flat?

Method: force function with ONLY diagonal part (cm1 terms, no s_phi)
on 1 z-bond at (cx, cy). Standard FDTD, 7 k-points.

Diagonal Peierls on 1 bond:
  lower site: fx += K1 * cm1 * ux_hi,  fy += K1 * cm1 * uy_hi
  upper site: fx += K1 * cm1 * ux_lo,  fy += K1 * cm1 * uy_lo
  (no s_phi cross-terms)

Born prediction: displacement coupling is K_eff = K1*cm1 modifying
  spring constant on 1 bond. σ_Born ∝ K_eff² × Z_avg(k)/sin²k.
  sin²·σ ∝ Z_avg(k), CV = 5.9% (same as file 49).

Expected outcomes and implications:
  CV(diag, α=0.30) ≈ 6%  → displacement coupling is Born per bond
                             s_phi·J is source of ALL non-Born behavior
                             flat integrand = Born-level mechanism
  CV(diag, α=0.30) >> 6% → even cm1 channel is non-Born per bond
                             flat integrand requires non-Born even for diagonal
                             §4.3 needs different framing

Note: at α=0, cm1=0 → force_fn = force_plain → σ_tr=0 exactly by
construction. No separate run needed.

Results (229s, 2 × 7 FDTD runs):

  CV(sin²·σ_tr, diagonal-only):
    α=0.10:  70.0%
    α=0.30:  69.0%
    Born:     5.9%

  Displacement coupling is strongly NON-BORN per bond (CV≈69% vs Born 5.9%).
  Rotation (s_phi·J) adds almost nothing per bond (full Peierls: 70.8%).

  σ_tr(diag) INCREASES with k at both α (ratio 0.64-0.66×), same as full
  Peierls. Per-bond non-Born behavior comes from displacement, not rotation.

  σ(0.30)/σ(0.10) diagonal-only: mean=35.5, Born=47.0, ratio/Born=0.76.
  Closer to Born than full Peierls (0.55) — rotation adds extra suppression.

  Key comparison:
    Diag, 1 bond:   CV = 69.0%  (α=0.30)
    Full, 1 bond:   CV = 70.8%  (file 49, α=0.30)
    Diag, ring 81:  CV =  8.8%  (file 35, α=0.30)
    Full, ring 81:  CV =  7.4%  (file 47, α=0.30)

  CONCLUSION: flat integrand is a COLLECTIVE ring property.
  Per-bond displacement is non-Born (CV=69%), ring interference
  reduces this 8× to CV=8.8%. Two separate mechanisms:
    A. Per bond: cm1 channel is non-Born (this file)
    B. Ring interference: 81 bonds average + destructive cross-terms
       flatten from 69% to 8.8% (file 35 diagonal ring)
  Rotation (s_phi) is minor per bond and minor on ring.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/50_diagonal_single_bond.py
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

c = np.sqrt(K1 + 4 * K2)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
alpha_vals = [0.10, 0.30]


def cv(x):
    return np.std(x) / np.abs(np.mean(x)) * 100


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


def make_diagonal_only_force(alpha, L, K1, K2):
    """Force function with ONLY diagonal Peierls (cm1·I) on 1 z-bond.

    Applies cm1 = cos(2πα)-1 displacement coupling on both (ux, uy)
    at the single bond (cx, cy, cz-1) <-> (cx, cy, cz). No s_phi terms.
    """
    cx = cy = cz = L // 2
    iz_lo = cz - 1
    iz_hi = cz

    phi = 2.0 * np.pi * alpha
    cm1 = np.cos(phi) - 1.0

    # Verify this is the same bond as R_loop=0 in make_vortex_force
    # gauge_3d uses: cz = L//2, iz_lo = cz-1, iz_hi = cz
    # precompute_disk_bonds returns (iy, ix) for bonds at that z-level
    iy_disk, ix_disk = precompute_disk_bonds(L, 0)
    assert len(iy_disk) == 1, f"R_loop=0 gives {len(iy_disk)} bonds"
    assert iy_disk[0] == cy and ix_disk[0] == cx, \
        f"Bond at ({ix_disk[0]},{iy_disk[0]}), expected ({cx},{cy})"

    # Convention matches gauge_3d.py lines 136-158 exactly:
    #   iz_lo = cz - 1,  iz_hi = cz
    #   lower gets K1 * cm1 * u_upper  (diagonal part of R-I)
    #   upper gets K1 * cm1 * u_lower  (diagonal part of R^T-I)
    # We drop s_phi terms (off-diagonal rotation coupling).

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        # Diagonal only: cm1 * u_neighbor (displacement coupling)
        # Lower site gets K1 * cm1 * u_upper
        fx[iz_lo, cy, cx] += K1 * cm1 * ux[iz_hi, cy, cx]
        fy[iz_lo, cy, cx] += K1 * cm1 * uy[iz_hi, cy, cx]

        # Upper site gets K1 * cm1 * u_lower
        fx[iz_hi, cy, cx] += K1 * cm1 * ux[iz_lo, cy, cx]
        fy[iz_hi, cy, cx] += K1 * cm1 * uy[iz_lo, cy, cx]

        return fx, fy, fz

    return force_fn


if __name__ == '__main__':
    thetas = np.linspace(0, np.pi, 13)
    phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    x_start = DW + 5
    iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
    gamma_pml = make_damping_3d(L, DW, DS)

    t0 = time.time()
    cx = cy = cz = L // 2
    iz_lo = cz - 1
    iz_hi = cz
    print("Route 50: Diagonal-only (cm1·I) on single bond")
    print(f"  L={L}, r_m={r_m}")
    print(f"  z-bond: iz_lo={iz_lo} -> iz_hi={iz_hi} at (iy={cy}, ix={cx})")
    print(f"  k = {list(k_vals)}")
    print(f"  α = {alpha_vals}")

    # cm1 values — s_phi shown but DROPPED from force
    for alpha in alpha_vals:
        cm1 = np.cos(2 * np.pi * alpha) - 1.0
        s_phi = np.sin(2 * np.pi * alpha)
        print(f"  α={alpha}: cm1={cm1:.4f}, s_phi={s_phi:.4f} (DROPPED)")
    print(f"  Note: α=0 → cm1=0 → force=force_plain → σ_tr=0 (by construction)")

    # References
    print(f"\nComputing {len(k_vals)} references...")
    refs = {}
    for k0 in k_vals:
        ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
        ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                          DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        refs[k0] = (ref, ux0, vx0, ns)
    print(f"  Done ({time.time()-t0:.0f}s)")

    # Born prediction
    Z_avg = 8 * np.pi * (1 + np.sin(k_vals) / k_vals)
    cv_born = cv(Z_avg)
    print(f"\nBorn prediction: CV(sin²·σ) = CV(Z_avg) = {cv_born:.1f}%")
    print(f"  sin²·σ_Born(1 bond, displacement) = K_eff² × Z_avg")
    print(f"  K_eff cancels in CV → CV = {cv_born:.1f}% at any α")

    results = {}
    for alpha in alpha_vals:
        t1 = time.time()
        f_diag = make_diagonal_only_force(alpha, L, K1, K2)

        sigma_tr = np.zeros(len(k_vals))
        for i, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_diag, ux0.copy(), vx0.copy(), gamma_pml,
                            DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
            f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                   ref['ux'], ref['uy'], ref['uz'], r_m)
            _, st = integrate_sigma_3d(f2, thetas, phis)
            sigma_tr[i] = st

        dt_run = time.time() - t1
        integrand = np.sin(k_vals)**2 * sigma_tr
        cv_val = cv(integrand)

        # Shape ratio: σ_FDTD / σ_Born_shape, normalized
        born_shape = Z_avg / np.sin(k_vals)**2
        shape_ratio = sigma_tr / born_shape
        shape_ratio_n = shape_ratio / shape_ratio[0]
        cv_shape = cv(shape_ratio_n)

        print(f"\n{'='*60}")
        print(f"  α = {alpha} — DIAGONAL ONLY (cm1·I)  ({dt_run:.0f}s)")
        print(f"{'='*60}")

        cm1 = np.cos(2 * np.pi * alpha) - 1.0
        print(f"  cm1 = {cm1:.4f} (K_eff = K1*cm1 = {K1*cm1:.4f})")

        print(f"\n  {'k':>5s}  {'σ_tr':>10s}  {'sin²·σ':>8s}  {'σ/Born':>10s}")
        print(f"  {'-'*38}")
        for i in range(len(k_vals)):
            print(f"  {k_vals[i]:5.2f}  {sigma_tr[i]:10.6f}  "
                  f"{integrand[i]:8.6f}  {shape_ratio_n[i]:10.4f}")

        print(f"\n  CV(sin²·σ_tr) = {cv_val:.1f}%")
        print(f"  CV(σ/Born_shape) = {cv_shape:.1f}%")

        if cv_val < 10:
            verdict_cv = "BORN-FLAT (CV < 10%)"
        elif cv_val < 30:
            verdict_cv = "Mild non-Born"
        else:
            verdict_cv = "Strong non-Born"
        print(f"  → {verdict_cv}")

        if cv_shape < 5:
            verdict_shape = "Born shape holds per bond"
        elif cv_shape < 15:
            verdict_shape = "Mild spectral deviation"
        else:
            verdict_shape = "Non-Born MODIFIES spectral shape"
        print(f"  → {verdict_shape}")

        # σ_tr trend
        print(f"  σ_tr range: {sigma_tr[0]:.6f} → {sigma_tr[-1]:.6f}"
              f"  (ratio {sigma_tr[0]/sigma_tr[-1]:.2f}×)")
        if sigma_tr[0] > sigma_tr[-1]:
            print(f"  σ_tr DECREASES with k (Born-like)")
        else:
            print(f"  σ_tr INCREASES with k (non-Born)")

        results[alpha] = {
            'sigma_tr': sigma_tr.copy(),
            'cv': cv_val,
            'cv_shape': cv_shape,
            'integrand': integrand.copy(),
        }

    # Comparison: diagonal-only vs full Peierls (file 49)
    print(f"\n{'='*60}")
    print(f"  COMPARISON — single bond")
    print(f"{'='*60}")

    # Reference values from 49_single_bond_cv.py (full Peierls, 1 bond, Mar 2026)
    file49_cv_full_030 = 70.8
    file49_cv_shape_030 = 75.5

    print(f"\n  {'':>18s}  {'Diag α=0.10':>12s}  {'Diag α=0.30':>12s}"
          f"  {'Full α=0.30':>12s}  {'Born':>8s}")
    print(f"  {'-'*68}")
    print(f"  {'CV(sin²·σ)':>18s}  {results[0.10]['cv']:11.1f}%"
          f"  {results[0.30]['cv']:11.1f}%"
          f"  {file49_cv_full_030:11.1f}%  {cv_born:7.1f}%")
    print(f"  {'CV(σ/Born)':>18s}  {results[0.10]['cv_shape']:11.1f}%"
          f"  {results[0.30]['cv_shape']:11.1f}%"
          f"  {file49_cv_shape_030:11.1f}%  {'0.0':>7s}%")

    # Interpret: does rotation (s_phi) cause the non-Born per-bond behavior?
    cv_diag_030 = results[0.30]['cv']
    cv_full_030 = file49_cv_full_030
    print(f"\n  At α=0.30:")
    print(f"    Diagonal-only CV = {cv_diag_030:.1f}%")
    print(f"    Full Peierls CV  = {cv_full_030:.1f}% (file 49)")
    print(f"    Born minimum     = {cv_born:.1f}%")

    if cv_diag_030 < 15:
        print(f"\n  DISPLACEMENT COUPLING IS BORN-FLAT PER BOND.")
        print(f"  CV={cv_full_030:.0f}% comes entirely from rotation (s_phi·J).")
        print(f"  Confirms file 35 ring result at per-bond level.")
    elif cv_diag_030 < 40:
        print(f"\n  Displacement coupling has moderate non-Born deviation.")
        print(f"  Both cm1 and s_phi contribute to non-Born per bond.")
    else:
        print(f"\n  Displacement coupling is strongly non-Born per bond.")
        print(f"  Non-Born per bond is NOT just rotation.")

    # α-ratio for diagonal-only
    ratio = results[0.30]['sigma_tr'] / results[0.10]['sigma_tr']
    cm1_030 = np.cos(2 * np.pi * 0.30) - 1.0
    cm1_010 = np.cos(2 * np.pi * 0.10) - 1.0
    born_ratio = cm1_030**2 / cm1_010**2
    mean_ratio = np.mean(ratio)
    print(f"\n  σ(0.30)/σ(0.10) diagonal-only:")
    print(f"    Mean ratio = {mean_ratio:.1f}")
    print(f"    Born prediction = cm1(0.30)²/cm1(0.10)² = {born_ratio:.1f}")
    print(f"    Ratio/Born = {mean_ratio/born_ratio:.2f}")
    if abs(mean_ratio/born_ratio - 1) < 0.2:
        print(f"    → Born scaling HOLDS for displacement coupling")
    else:
        print(f"    → Born scaling FAILS for displacement coupling")

    # Cross-check vs file 35 (diagonal on ring, α=0.30, CV=8.8%)
    file35_cv_diag_ring = 8.8  # from 35_diagonal_offdiag.py (Mar 2026)
    print(f"\n  Cross-check vs file 35 (diagonal on ring):")
    print(f"    Diagonal single bond CV = {cv_diag_030:.1f}%")
    print(f"    Diagonal full ring CV   = {file35_cv_diag_ring}% (file 35, α=0.30)")
    if cv_diag_030 < 15:
        print(f"    Consistent: per-bond displacement is Born-flat")
        print(f"    Ring flatness ({file35_cv_diag_ring}%) = per-bond Born-flat"
              f" + ring interference")
        print(f"    The two facts are ADDITIVE, not the same mechanism")

    # Main conclusion
    print(f"\n{'='*60}")
    print(f"  CONCLUSION")
    print(f"{'='*60}")
    print(f"  Diagonal-only (cm1·I) on 1 bond:")
    print(f"    CV(sin²·σ, α=0.10) = {results[0.10]['cv']:.1f}%")
    print(f"    CV(sin²·σ, α=0.30) = {results[0.30]['cv']:.1f}%")
    print(f"    Born prediction     = {cv_born:.1f}%")

    if results[0.30]['cv'] < 15 and results[0.10]['cv'] < 15:
        print(f"\n  Displacement coupling (cm1·I) is Born-flat per bond at BOTH α.")
        print(f"  Two separate mechanisms contribute to ring flatness:")
        print(f"    A. Per bond: cm1 channel is Born-flat (this file)")
        print(f"    B. Per bond: s_phi channel causes non-Born CV≈71% (file 49)")
        print(f"    C. On ring: rotation (s_phi) partly cancels between bonds")
        print(f"       → diagonal dominates ring (file 35: CV=8.8%)")
        print(f"    D. Ring interference flattens further (file 48: CV=7.4%)")
        print(f"  A and C+D are ADDITIVE effects, not the same mechanism.")
    elif results[0.30]['cv'] < 15:
        print(f"\n  Born-flat at α=0.30, non-Born at α=0.10.")
        print(f"  Non-Born displacement coupling at weak coupling only.")
    else:
        print(f"\n  Displacement coupling is NON-BORN per bond.")
        print(f"  Non-Born per bond affects both cm1 and s_phi channels.")

    t_total = time.time() - t0
    print(f"\nTotal: {t_total:.0f}s ({t_total/60:.1f} min)")
