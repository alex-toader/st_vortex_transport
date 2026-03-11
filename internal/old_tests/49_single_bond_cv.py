"""
Route 49 (I7): Single bond CV at α=0.10 vs α=0.30.

File 48 shows per-quarter CV drops from 37.3% (α=0.10) to 15.5% (α=0.30).
Claim: non-Born enhancement at strong coupling makes individual bonds flatter.
Test: measure CV(sin²·σ_tr) for a SINGLE z-bond at both α.

Method: make_vortex_force with R_loop=0 → disk with 1 bond at (cx, cy).
Standard FDTD pipeline, 7 k-points.

If CV(α=0.10) >> CV(α=0.30) ≈ 71%: non-Born per-bond flattening confirmed.
If CV(α=0.10) ≈ CV(α=0.30): α-dependence is inter-bond, not per-bond.

Born prediction: σ_tr(Born, 1 bond) ∝ K_eff² × Z_avg(k) / sin²(k).
  sin²·σ ∝ Z_avg(k) = 8π(1+sin(k)/k), CV(Z_avg) ≈ 5.7%.
  Born shape is α-independent (K_eff cancels in CV).
  Deviation from 5.7% = non-Born effect.

Results (244s, 2 × 7 FDTD runs):

  Sanity: R_loop=0 gives 1 bond at (40,40). CV(α=0.30) = 70.8% ≈ 71% (F4). PASS.

  CV(sin²·σ_tr):
    α=0.10:  96.3%
    α=0.30:  70.8%
    Born:     5.9%

  Per-bond integrand is α-DEPENDENT. CV(0.10) > CV(0.30).
  Non-Born enhancement at strong coupling makes individual bonds flatter.
  Confirms file 48 point 6.

  But both are FAR from Born (5.9%). Non-Born modifies per-bond spectral
  shape massively: σ_tr INCREASES with k (ratio 0.2× at α=0.10, 0.6× at
  α=0.30 — i.e., σ grows from k=0.3 to k=1.5). Born predicts σ ∝ 1/sin²k
  (decreasing). The non-Born reversal is stronger at weak coupling.

  σ(0.30)/σ(0.10) per bond:
    k=0.3: 41.1,  k=0.9: 24.8,  k=1.5: 13.6  (mean 26.0)
    Born for full Peierls: sin²(0.3π)/sin²(0.1π) = 6.9
      (NOT cm1²/cm1² = 47.0, which applies only to diagonal-only)
    FDTD/Born = 3.79 — per-bond OVERSHOOTS Born α-ratio
    Full ring:  17.0/6.9 = 2.48 (file 47)
    → ring interference reduces excess from 3.8× to 2.5× Born

  CV(σ/Born_shape):
    α=0.10: 100.9%  (shape completely different from Born)
    α=0.30:  75.5%  (shape very different from Born)
    Non-Born MODIFIES spectral shape per bond at both α.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/49_single_bond_cv.py
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
from gauge_3d import make_vortex_force, precompute_disk_bonds

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


if __name__ == '__main__':
    thetas = np.linspace(0, np.pi, 13)
    phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    x_start = DW + 5
    iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
    gamma_pml = make_damping_3d(L, DW, DS)

    # References (shared)
    t0 = time.time()
    print("Route 49: Single bond CV(sin²·σ_tr)")
    print(f"  R_loop=0 (1 bond), L={L}, r_m={r_m}")
    print(f"  k = {list(k_vals)}")
    print(f"  α = {alpha_vals}")

    # Sanity: verify R_loop=0 gives exactly 1 bond
    iy_test, ix_test = precompute_disk_bonds(L, 0)
    assert len(iy_test) == 1, f"R_loop=0 gives {len(iy_test)} bonds, expected 1"
    print(f"  R_loop=0 sanity: {len(iy_test)} bond at ({ix_test[0]},{iy_test[0]}). PASS.")

    print(f"\nComputing {len(k_vals)} references...")
    refs = {}
    for k0 in k_vals:
        ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
        ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                          DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        refs[k0] = (ref, ux0, vx0, ns)
    print(f"  Done ({time.time()-t0:.0f}s)")

    # Born prediction: sin²·σ_Born = sin²·(Z_avg/sin²) = Z_avg exactly.
    # CV(Born integrand) = CV(Z_avg) — theoretical minimum for single
    # displacement-coupled bond (no non-Born correction).
    Z_avg = 8 * np.pi * (1 + np.sin(k_vals) / k_vals)
    cv_born = cv(Z_avg)
    print(f"\nBorn prediction: CV(sin²·σ) = CV(Z_avg) = {cv_born:.1f}%")
    print(f"  (theoretical minimum — no non-Born correction)")
    print(f"  Z_avg = {['%.1f' % z for z in Z_avg]}")

    results = {}
    for alpha in alpha_vals:
        t1 = time.time()
        f_def = make_vortex_force(alpha, 0, L, K1, K2)  # R_loop=0 → 1 bond

        sigma_tr = np.zeros(len(k_vals))
        for i, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                            DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
            f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                   ref['ux'], ref['uy'], ref['uz'], r_m)
            _, st = integrate_sigma_3d(f2, thetas, phis)
            sigma_tr[i] = st

        dt = time.time() - t1
        integrand = np.sin(k_vals)**2 * sigma_tr
        cv_val = cv(integrand)

        # Non-Born ratio: σ_FDTD / (Z_avg/sin²k) normalized
        shape_ratio = sigma_tr / (Z_avg / np.sin(k_vals)**2)
        shape_ratio_n = shape_ratio / shape_ratio[0]
        cv_shape = cv(shape_ratio_n)

        print(f"\n{'='*60}")
        print(f"  α = {alpha}  ({dt:.0f}s)")
        print(f"{'='*60}")

        cm1 = np.cos(2 * np.pi * alpha) - 1
        s_phi = np.sin(2 * np.pi * alpha)
        print(f"  cm1 = {cm1:.4f}, s_phi = {s_phi:.4f}")

        print(f"\n  {'k':>5s}  {'σ_tr':>10s}  {'sin²·σ':>8s}  {'σ/Born_shape':>12s}")
        print(f"  {'-'*40}")
        for i in range(len(k_vals)):
            print(f"  {k_vals[i]:5.2f}  {sigma_tr[i]:10.4f}  {integrand[i]:8.4f}"
                  f"  {shape_ratio_n[i]:12.4f}")

        print(f"\n  CV(sin²·σ_tr) = {cv_val:.1f}%")
        print(f"  CV(σ/Born_shape) = {cv_shape:.1f}% (deviation from Born)")

        if cv_shape < 5:
            verdict = "Born shape holds per bond (non-Born is multiplicative)"
        elif cv_shape < 15:
            verdict = "Mild spectral deviation per bond"
        else:
            verdict = "Non-Born MODIFIES spectral shape per bond"
        print(f"  → {verdict}")

        print(f"  σ_tr range: {sigma_tr[0]:.4f} to {sigma_tr[-1]:.4f}"
              f" (ratio {sigma_tr[0]/sigma_tr[-1]:.1f}×)")

        results[alpha] = {
            'sigma_tr': sigma_tr.copy(),
            'cv': cv_val,
            'cv_shape': cv_shape,
            'integrand': integrand.copy(),
        }

    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")

    print(f"\n  {'':>10s}  {'α=0.10':>8s}  {'α=0.30':>8s}  {'Born':>8s}")
    print(f"  {'-'*40}")
    print(f"  {'CV(sin²·σ)':>10s}  {results[0.10]['cv']:7.1f}%"
          f"  {results[0.30]['cv']:7.1f}%  {cv_born:7.1f}%")
    print(f"  {'CV(σ/Born)':>10s}  {results[0.10]['cv_shape']:7.1f}%"
          f"  {results[0.30]['cv_shape']:7.1f}%  {'0.0':>7s}%")

    # α-ratio
    ratio = results[0.30]['sigma_tr'] / results[0.10]['sigma_tr']
    born_ratio = (np.cos(2*np.pi*0.30)-1)**2 / (np.cos(2*np.pi*0.10)-1)**2
    print(f"\n  σ(0.30)/σ(0.10) at each k:")
    print(f"  {'k':>5s}  {'ratio':>8s}")
    print(f"  {'-'*15}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {ratio[i]:8.2f}")
    mean_ratio = np.mean(ratio)
    print(f"  {'mean':>5s}  {mean_ratio:8.2f}  (Born pred: {born_ratio:.1f})")
    print(f"  CV of ratio: {cv(ratio):.1f}%")

    # Interpret: is non-Born enhancement per-bond or collective?
    ratio_frac = mean_ratio / born_ratio
    if abs(ratio_frac - 1) < 0.2:
        print(f"  → ratio/Born = {ratio_frac:.2f} — NON-BORN IS PER-BOND"
              f" (single bond matches Born prediction)")
    elif ratio_frac < 0.5:
        print(f"  → ratio/Born = {ratio_frac:.2f} — NON-BORN IS COLLECTIVE"
              f" (ring suppresses ratio below Born)")
    else:
        print(f"  → ratio/Born = {ratio_frac:.2f} — INTERMEDIATE"
              f" (partial per-bond, partial collective)")

    # Ring comparison (file 47: mean σ(0.30)/σ(0.10) = 17.0, Born pred = 47.0)
    print(f"\n  Ring comparison (file 47):")
    print(f"    Single bond: ratio = {mean_ratio:.1f}, Born = {born_ratio:.1f},"
          f" frac = {ratio_frac:.2f}")
    print(f"    Full ring:   ratio = 17.0,  Born = 47.0,"
          f" frac = 0.36")
    if ratio_frac > 0.8:
        print(f"    → Single bond closer to Born than ring → ring interference"
              f" modifies α-scaling")
    else:
        print(f"    → Both suppressed → non-Born enhancement per-bond + collective")

    # Sanity check vs file 35 (CV=71% at α=0.30)
    cv_030_measured = results[0.30]['cv']
    cv_030_expected = 71.0
    print(f"\n  Sanity vs tracker (F4, file 35 era): CV(α=0.30) ="
          f" {cv_030_measured:.1f}% (expected ~{cv_030_expected:.0f}%)")
    if abs(cv_030_measured - cv_030_expected) > 10:
        print(f"  WARNING: discrepancy > 10% — check single bond setup")
    else:
        print(f"  PASS")

    # Main conclusion
    cv_010 = results[0.10]['cv']
    cv_030 = results[0.30]['cv']
    print(f"\n{'='*60}")
    print(f"  CONCLUSION")
    print(f"{'='*60}")
    print(f"  CV(sin²·σ, α=0.10) = {cv_010:.1f}%")
    print(f"  CV(sin²·σ, α=0.30) = {cv_030:.1f}%")
    print(f"  Born minimum:         {cv_born:.1f}%")

    if cv_010 > cv_030 * 1.3:
        print(f"\n  CV(0.10) > CV(0.30) — per-bond integrand is α-DEPENDENT.")
        print(f"  Non-Born enhancement at strong coupling makes individual")
        print(f"  bonds flatter. Confirms file 48 point 6.")
    elif abs(cv_010 - cv_030) / cv_030 < 0.2:
        print(f"\n  CV(0.10) ≈ CV(0.30) — per-bond integrand is α-INDEPENDENT.")
        print(f"  α-dependence in file 48 is inter-bond (interference), not per-bond.")
    else:
        print(f"\n  Intermediate: CV ratio = {cv_010/cv_030:.2f}")

    t_total = time.time() - t0
    print(f"\nTotal: {t_total:.0f}s ({t_total/60:.1f} min)")
