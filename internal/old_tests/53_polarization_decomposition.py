"""
Route 53 (I16): Polarization decomposition of σ_bond.

Question: Is σ_bond ≈ const(k) from polarization compensation?
  σ_bond(k) = σ_xx(k) + σ_xy(k) + σ_xz(k)
  where σ_xx = same-pol (x→x), σ_xy = cross-pol (x→y), σ_xz = (x→z, should be 0).

File 52: scalar T-matrix FAILS (|DK·G|≪1, Born at all α).
File 49: cross-pol fraction grows 1%→63% from k=0.3→1.5.
Hypothesis: σ_xx decreases with k (Born-like), σ_xy increases with k,
  sum ≈ const. This would explain σ_bond ≈ const as vectorial effect.

Method: same as file 49 (single bond, R_loop=0, 7 k-pts).
  Instead of compute_sphere_f2 (sums all components), compute separately:
  f2_xx = r_m² × <sc_ux²> / <inc²>   (same polarization)
  f2_xy = r_m² × <sc_uy²> / <inc²>   (cross polarization)
  f2_xz = r_m² × <sc_uz²> / <inc²>   (uz channel, should be ~0)
  Then integrate each to get σ_tr_xx, σ_tr_xy, σ_tr_xz.
  Two α values: 0.30 (σ_bond≈const) and 0.10 (σ_bond NOT const).

Results (4 min, 2×7 FDTD):

  POLARIZATION COMPENSATION HYPOTHESIS FAILS.
  σ_xx (same-pol) INCREASES with k at both α — opposite to Born prediction.
  σ_xy (cross-pol) also increases. Both grow → no compensation possible.

  α=0.30:
    σ_xx: 0.0595→0.0804 (1.35× increase). Born predicts 13.6× DECREASE.
    σ_xy: 0.0003→0.0158 (57.9×). Cross-pol fraction: 0.5%→16.5%.
    σ_tot: 0.0598→0.0962 (1.61×). CV(sin²·σ_tot)=70.8%.

  α=0.10:
    σ_xx: 0.0013→0.0016 (1.16× increase). Same reversal as α=0.30.
    σ_xy: 0.0001→0.0055 (50.0×). Cross-pol fraction: 7.6%→77.9%.
    σ_tot: 0.0015→0.0071 (4.87×). CV(sin²·σ_tot)=96.3%.

  Key finding: σ_xx is nearly constant (CV≈62-66%) at BOTH α.
  This is α-INDEPENDENT — same as diagonal-only CV≈69% in file 50.
  The non-Born per-bond effect is in same-pol channel: Born gives 1/sin²(k)
  (13.6× decrease), FDTD gives slight increase (1.2-1.4×).

  Difference α=0.30 vs α=0.10 in σ_tot comes entirely from σ_xy weight:
    α=0.30: xy is 0.5-16.5% of total → small perturbation → CV≈71%
    α=0.10: xy is 7.6-77.9% of total → dominates at high k → CV≈96%

  σ_xz = 0.0000 at all k (uz decoupled, file 8 sanity).
  Sanity vs file 51: max rel diff 0.1%. PASS.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/53_polarization_decomposition.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../src/1_foam'))
from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           integrate_sigma_3d, estimate_n_steps_3d)
from gauge_3d import make_vortex_force, precompute_disk_bonds

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20

k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
alpha_vals = [0.30, 0.10]


def cv(x):
    return np.std(x) / np.abs(np.mean(x)) * 100


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


def compute_sphere_f2_pol(def_ux, def_uy, def_uz,
                          ref_ux, ref_uy, ref_uz, r_m):
    """Like compute_sphere_f2 but returns per-polarization f2.

    Returns f2_xx, f2_xy, f2_xz (each shape N_pts).
    """
    sc_ux = def_ux - ref_ux
    sc_uy = def_uy - ref_uy
    sc_uz = def_uz - ref_uz
    inc2 = np.mean(ref_ux**2 + ref_uy**2 + ref_uz**2, axis=0)
    inc2_floor = max(1e-30, 1e-12 * np.max(inc2))
    inc2 = np.maximum(inc2, inc2_floor)
    f2_xx = r_m**2 * np.mean(sc_ux**2, axis=0) / inc2
    f2_xy = r_m**2 * np.mean(sc_uy**2, axis=0) / inc2
    f2_xz = r_m**2 * np.mean(sc_uz**2, axis=0) / inc2
    return f2_xx, f2_xy, f2_xz


if __name__ == '__main__':
    thetas = np.linspace(0, np.pi, 13)
    phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    x_start = DW + 5
    iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
    gamma_pml = make_damping_3d(L, DW, DS)

    t0 = time.time()
    print("Route 53: Polarization decomposition of σ_bond")
    print(f"  Single bond (R_loop=0), L={L}, r_m={r_m}")
    print(f"  k = {list(k_vals)}")
    print(f"  α = {alpha_vals}")

    # Sanity: 1 bond
    iy_test, ix_test = precompute_disk_bonds(L, 0)
    assert len(iy_test) == 1
    print(f"  1 bond at ({ix_test[0]},{iy_test[0]}). OK.")

    # References (shared across α)
    print(f"\nComputing {len(k_vals)} references...")
    refs = {}
    for k0 in k_vals:
        ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
        ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                          DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        refs[k0] = (ref, ux0, vx0, ns)
    print(f"  Done ({time.time()-t0:.0f}s)")

    # Sanity: incident is x-polarized → ref_uy, ref_uz should be ~0
    ref_test = refs[k_vals[3]][0]  # k=0.9
    ratio_uy = np.max(np.abs(ref_test['uy'])) / np.max(np.abs(ref_test['ux']))
    ratio_uz = np.max(np.abs(ref_test['uz'])) / np.max(np.abs(ref_test['ux']))
    print(f"  Sanity: max|ref_uy|/max|ref_ux| = {ratio_uy:.2e}")
    print(f"  Sanity: max|ref_uz|/max|ref_ux| = {ratio_uz:.2e}")

    # Born prediction for same-pol: σ_Born ∝ Z_avg/sin²(k), all same-pol
    Z_avg = 8 * np.pi * (1 + np.sin(k_vals) / k_vals)
    born_shape = Z_avg / np.sin(k_vals)**2
    born_ratio_first_last = born_shape[0] / born_shape[-1]

    # ═══════════════════════════════════════════════════════════════
    # Loop over α values
    # ═══════════════════════════════════════════════════════════════
    all_results = {}
    for alpha in alpha_vals:
        t1 = time.time()
        f_def = make_vortex_force(alpha, 0, L, K1, K2)

        sigma_xx = np.zeros(len(k_vals))
        sigma_xy = np.zeros(len(k_vals))
        sigma_xz = np.zeros(len(k_vals))
        sigma_tot = np.zeros(len(k_vals))

        for i, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                            DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
            f2_xx, f2_xy, f2_xz = compute_sphere_f2_pol(
                d['ux'], d['uy'], d['uz'],
                ref['ux'], ref['uy'], ref['uz'], r_m)

            _, st_xx = integrate_sigma_3d(f2_xx, thetas, phis)
            _, st_xy = integrate_sigma_3d(f2_xy, thetas, phis)
            _, st_xz = integrate_sigma_3d(f2_xz, thetas, phis)

            sigma_xx[i] = st_xx
            sigma_xy[i] = st_xy
            sigma_xz[i] = st_xz
            sigma_tot[i] = st_xx + st_xy + st_xz

        dt = time.time() - t1

        # ═══════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print(f"  α = {alpha}  ({dt:.0f}s)")
        print(f"{'='*60}")

        print(f"\n  {'k':>5s}  {'σ_xx':>8s}  {'σ_xy':>8s}  {'σ_xz':>8s}"
              f"  {'σ_tot':>8s}  {'xy/tot':>7s}")
        print(f"  {'-'*50}")
        for i in range(len(k_vals)):
            frac = sigma_xy[i] / sigma_tot[i] * 100
            print(f"  {k_vals[i]:5.2f}  {sigma_xx[i]:8.4f}  {sigma_xy[i]:8.4f}"
                  f"  {sigma_xz[i]:8.4f}  {sigma_tot[i]:8.4f}  {frac:6.1f}%")

        # Internal consistency
        print(f"\n  Internal consistency (sum of components vs σ_tot):")
        for i in range(len(k_vals)):
            s = sigma_xx[i] + sigma_xy[i] + sigma_xz[i]
            print(f"    k={k_vals[i]:.1f}: sum={s:.6f} tot={sigma_tot[i]:.6f}"
                  f" diff={abs(s - sigma_tot[i]) / sigma_tot[i] * 100:.4f}%")

        # Sanity vs file 51/49
        if alpha == 0.30:
            sigma_ref = np.array([0.0598, 0.0543, 0.0571, 0.0624,
                                  0.0707, 0.0807, 0.0962])
            label = "file 51"
        else:
            sigma_ref = None
            label = None

        if sigma_ref is not None:
            rel_diff = np.abs(sigma_tot - sigma_ref) / sigma_ref * 100
            print(f"\n  Sanity vs {label} σ_bond:")
            print(f"    max rel diff: {np.max(rel_diff):.1f}%")
            if np.max(rel_diff) < 5:
                print(f"    PASS")
            else:
                print(f"    WARNING: discrepancy > 5%")

        # CVs
        cv_xx = cv(np.sin(k_vals)**2 * sigma_xx)
        cv_xy = cv(np.sin(k_vals)**2 * sigma_xy)
        cv_tot = cv(np.sin(k_vals)**2 * sigma_tot)

        print(f"\n  CV(sin²·σ_xx) = {cv_xx:.1f}%  (same-pol)")
        print(f"  CV(sin²·σ_xy) = {cv_xy:.1f}%  (cross-pol)")
        print(f"  CV(sin²·σ_tot) = {cv_tot:.1f}%  (total)")

        # Ranges
        print(f"\n  σ_xx: {sigma_xx[0]:.4f} → {sigma_xx[-1]:.4f}"
              f" (ratio {sigma_xx[0]/sigma_xx[-1]:.2f}×)")
        print(f"  σ_xy: {sigma_xy[0]:.6f} → {sigma_xy[-1]:.4f}"
              f" (ratio {sigma_xy[-1]/sigma_xy[0]:.1f}×)")
        print(f"  σ_tot: {sigma_tot[0]:.4f} → {sigma_tot[-1]:.4f}"
              f" (ratio {sigma_tot[-1]/sigma_tot[0]:.2f}×)")
        print(f"  Born σ_xx ratio (k=0.3)/(k=1.5) = {born_ratio_first_last:.1f}")

        # Compensation test: σ_tot flatter than both components?
        if cv_tot < cv_xx and cv_tot < cv_xy:
            print(f"\n  ==> COMPENSATION: σ_tot (CV={cv_tot:.1f}%) flatter"
                  f" than σ_xx ({cv_xx:.1f}%) and σ_xy ({cv_xy:.1f}%)")
        else:
            print(f"\n  ==> NO COMPENSATION: σ_tot not flatter than both"
                  f" components")
            print(f"      CV: xx={cv_xx:.1f}%, xy={cv_xy:.1f}%,"
                  f" tot={cv_tot:.1f}%")

        all_results[alpha] = {
            'sigma_xx': sigma_xx.copy(),
            'sigma_xy': sigma_xy.copy(),
            'sigma_xz': sigma_xz.copy(),
            'sigma_tot': sigma_tot.copy(),
            'cv_xx': cv_xx,
            'cv_xy': cv_xy,
            'cv_tot': cv_tot,
        }

    # ═══════════════════════════════════════════════════════════════
    # Comparison across α
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")

    print(f"\n  {'':>12s}  {'α=0.30':>8s}  {'α=0.10':>8s}")
    print(f"  {'-'*35}")
    print(f"  {'CV(sin²·xx)':>12s}  {all_results[0.30]['cv_xx']:7.1f}%"
          f"  {all_results[0.10]['cv_xx']:7.1f}%")
    print(f"  {'CV(sin²·xy)':>12s}  {all_results[0.30]['cv_xy']:7.1f}%"
          f"  {all_results[0.10]['cv_xy']:7.1f}%")
    print(f"  {'CV(sin²·tot)':>12s}  {all_results[0.30]['cv_tot']:7.1f}%"
          f"  {all_results[0.10]['cv_tot']:7.1f}%")

    # Cross-pol fraction at k=0.3 and k=1.5
    for alpha in alpha_vals:
        r = all_results[alpha]
        frac_lo = r['sigma_xy'][0] / r['sigma_tot'][0] * 100
        frac_hi = r['sigma_xy'][-1] / r['sigma_tot'][-1] * 100
        print(f"\n  α={alpha}: xy/tot at k=0.3: {frac_lo:.1f}%,"
              f" k=1.5: {frac_hi:.1f}%")

    # Conclusion
    print(f"\n{'='*60}")
    print(f"  CONCLUSION")
    print(f"{'='*60}")

    r030 = all_results[0.30]
    r010 = all_results[0.10]

    comp_030 = r030['cv_tot'] < r030['cv_xx'] and r030['cv_tot'] < r030['cv_xy']
    comp_010 = r010['cv_tot'] < r010['cv_xx'] and r010['cv_tot'] < r010['cv_xy']

    if comp_030 and not comp_010:
        print(f"  Compensation at α=0.30 but NOT at α=0.10.")
        print(f"  Polarization compensation explains σ_bond ≈ const"
              f" AND the α threshold.")
    elif comp_030 and comp_010:
        print(f"  Compensation at BOTH α values.")
        print(f"  Mechanism is universal, not α-dependent.")
    elif not comp_030:
        print(f"  NO compensation even at α=0.30.")
        print(f"  Polarization compensation hypothesis FAILS.")
    else:
        print(f"  Compensation only at α=0.10 — unexpected.")

    t_total = time.time() - t0
    print(f"\nTotal: {t_total:.0f}s ({t_total/60:.1f} min)")
