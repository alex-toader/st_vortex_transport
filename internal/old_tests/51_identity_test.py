"""
Route 51 (I11+I12): σ_per_bond × I_tr identity test + functional form.

I11: Does σ_per_bond(k) × I_tr(k, R=5) ≈ const × Z_avg(k)?
  If yes → flat integrand is algebraic identity (non-Born × coherent = incoherent Born)
  If no  → compensation is emergent, no simple closed form

I12: What is the functional form of σ_per_bond(k)?
  Fit candidates: sin²(k), k², 1-cos(k), Z_avg/I_tr, etc.

Additional:
  N_eff(k) = σ_ring / σ_per_bond: effective coherent bond count vs k.
  R_bond(k) vs R_ring(k): is non-Born enhancement same per-bond and on ring?

Data:
  σ_ring(α=0.30, R=5): file 18 [40.74, 14.16, 7.69, 5.05, 3.78, 3.14, 2.81]
  I_tr(k, R=5): computed analytically (file 43 method)
  σ_per_bond: FDTD with R_loop=0 (file 49 setup), α=0.30, 7 k-pts

Results (153s):

  Sanity: CV(sin²·σ_bond) = 70.8% (file 49: 70.8%). ratio k=0.3/1.5 = 0.62. PASS.
  Sanity: I_tr ratio k=0.3/1.5 = 58.2× (file 43: ~58×). PASS.

  σ_per_bond(k, α=0.30):
    k:    0.30     0.50     0.70     0.90     1.10     1.30     1.50
    σ:  0.0598   0.0543   0.0571   0.0624   0.0707   0.0807   0.0962

  C1 (I11): σ_bond × I_tr ≈ Z_avg?
    CV(product/Z_avg) = 145.2%. Product drops 36.2×, Z_avg varies 1.19×.
    IDENTITY FAILS. Compensation is NOT algebraic.

  C2: N_eff(k) = σ_ring / σ_bond
    N_eff: 681 → 261 → 135 → 81 → 53 → 39 → 29
    Drops 23.3× (k=0.3 to 1.5). NOT incoherent (N_bonds=81).
    N_eff vs I_tr/Z_avg: CV=21.3%. Follows Born form factor approximately,
    with systematic 2× drift at high k.

  C3 (I12): Functional form of σ_bond(k)
    Best fit: CONST (CV=20.3%). Power law k^{0.29}.
    σ_bond ≈ nearly k-independent. Grows only 1.6× over full BZ.
    Born would give 1/sin²(k) (varies 11.5×). Completely different.

  C4: Non-Born R(k) per-bond vs ring
    R_bond grows 21.8×, R_ring grows 45.8×. CV(Rb/Rr)=26.6%.
    Ring interference AMPLIFIES non-Born enhancement by ~2× at high k.

  Mechanism decomposition:
    sin²·σ_ring = sin²·σ_bond × N_eff = (grows 18.3×) × (drops 23.3×)
    Net: varies 1.27× = flat integrand (CV=7.4%).
    The 18.3× growth is mostly sin²(k) itself (11.4×); σ_bond adds only 1.6×.

  Central finding: σ_bond ≈ const(k) is the non-Born per-bond property
  that makes flat integrand possible. WHY σ_bond ≈ const is the open question.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/51_identity_test.py
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
ALPHA = 0.30

k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# Ring FDTD data from file 18 at α=0.30, R=5 (81 bonds)
sigma_ring = np.array([40.74, 14.16, 7.69, 5.05, 3.78, 3.14, 2.81])

# I_tr angular grid (from file 43)
N_THETA = 150
N_PHI = 300


def cv(x):
    return np.std(x) / np.abs(np.mean(x)) * 100


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


if __name__ == '__main__':
    t0 = time.time()
    print("Route 51 (I11+I12): Identity test + functional form")
    print(f"  alpha={ALPHA}, L={L}, r_m={r_m}")
    print(f"  k = {list(k_vals)}")

    # =================================================================
    # Part A: FDTD for sigma_per_bond(k)
    # =================================================================
    print(f"\n{'='*65}")
    print(f"Part A: Single-bond FDTD (R_loop=0, alpha={ALPHA})")
    print(f"{'='*65}")

    thetas_sph = np.linspace(0, np.pi, 13)
    phis_sph = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    x_start = DW + 5
    iz, iy, ix = make_sphere_points(L, r_m, thetas_sph, phis_sph)
    gamma_pml = make_damping_3d(L, DW, DS)

    iy_test, ix_test = precompute_disk_bonds(L, 0)
    assert len(iy_test) == 1, f"R_loop=0: {len(iy_test)} bonds"
    print(f"  R_loop=0: 1 bond at ({ix_test[0]},{iy_test[0]}). OK.")

    print(f"  Computing references...")
    refs = {}
    for k0 in k_vals:
        ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
        ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                          DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        refs[k0] = (ref, ux0, vx0, ns)

    f_def = make_vortex_force(ALPHA, 0, L, K1, K2)
    sigma_bond = np.zeros(len(k_vals))
    for i, k0 in enumerate(k_vals):
        ref, ux0, vx0, ns = refs[k0]
        d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                        DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        _, st = integrate_sigma_3d(f2, thetas_sph, phis_sph)
        sigma_bond[i] = st

    t_fdtd = time.time() - t0
    print(f"  Done ({t_fdtd:.0f}s)")
    print(f"\n  sigma_per_bond(k):")
    print(f"  {'k':>5s}  {'sigma':>12s}  {'sin2*sigma':>12s}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {sigma_bond[i]:12.6f}"
              f"  {np.sin(k_vals[i])**2 * sigma_bond[i]:12.6f}")

    integ_bond = np.sin(k_vals)**2 * sigma_bond
    cv_bond = cv(integ_bond)
    ratio_bond = sigma_bond[0] / sigma_bond[-1]
    print(f"\n  CV(sin2*sigma_bond) = {cv_bond:.1f}%")
    print(f"  sigma ratio k=0.3/k=1.5 = {ratio_bond:.3f}")
    print(f"\n  Sanity vs file 49 (alpha=0.30):")
    print(f"    CV: {cv_bond:.1f}% (expected ~70.8%)")
    print(f"    ratio: {ratio_bond:.2f} (expected ~0.6)")
    assert abs(cv_bond - 70.8) < 10, f"CV={cv_bond:.1f}% too far from file 49 (70.8%)"

    # =================================================================
    # Part B: Analytic I_tr(k, R=5)
    # =================================================================
    print(f"\n{'='*65}")
    print(f"Part B: Analytic I_tr(k, R=5)")
    print(f"{'='*65}")

    th_grid = np.linspace(0, np.pi, N_THETA + 1)
    ph_grid = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    TH, PH = np.meshgrid(th_grid, ph_grid, indexing='ij')
    theta_f = TH.ravel()
    phi_f = PH.ravel()

    sin_t = np.sin(theta_f)
    cos_t = np.cos(theta_f)
    cos_p = np.cos(phi_f)
    sin_p = np.sin(phi_f)

    cos_scat = sin_t * cos_p
    w_tr = 1 - cos_scat

    dtheta = np.pi / N_THETA
    dphi = 2 * np.pi / N_PHI
    theta_w = np.ones(N_THETA + 1) * dtheta
    theta_w[0] = dtheta / 2
    theta_w[-1] = dtheta / 2
    w_quad = np.repeat(theta_w, N_PHI) * dphi * np.abs(sin_t)

    R_RING = 5
    cx, cy = L // 2, L // 2
    iy_disk, ix_disk = precompute_disk_bonds(L, R_RING)
    x_b = (ix_disk - cx).astype(float)
    y_b = (iy_disk - cy).astype(float)
    positions = np.column_stack([x_b, y_b])
    N_bonds = len(positions)
    print(f"  R={R_RING}: {N_bonds} bonds")

    I_tr = np.zeros(len(k_vals))
    for j, k0 in enumerate(k_vals):
        # q = k_out - k_in. k_in = k*x_hat, k_out = k*(sinθ cosφ, sinθ sinφ, cosθ)
        # Ring bonds at z=cz: q·r = qx*x_b + qy*y_b (qz irrelevant, z_b=0 relative)
        qx = k0 * (sin_t * cos_p - 1)
        qy = k0 * sin_t * sin_p
        Q = np.column_stack([qx, qy])
        phase = Q @ positions.T
        F = np.sum(np.exp(1j * phase), axis=1)
        F2 = np.abs(F)**2
        # Z-structure factor for z-bond: Z = |1+e^{-ik_z}|² = 4cos²(k cosθ_z/2)
        # θ is polar angle from z-axis (= bond axis), so cos_t = cosθ_z. Correct.
        Z = 4 * np.cos(k0 * cos_t / 2)**2
        I_tr[j] = np.sum(F2 * Z * w_tr * w_quad)

    print(f"\n  {'k':>5s}  {'I_tr':>12s}")
    for j in range(len(k_vals)):
        print(f"  {k_vals[j]:5.2f}  {I_tr[j]:12.2f}")
    itr_ratio = I_tr[0] / I_tr[-1]
    print(f"  I_tr ratio k=0.3/k=1.5 = {itr_ratio:.1f}x")
    print(f"  Sanity vs file 43: expected ~58x")
    assert 40 < itr_ratio < 80, f"I_tr ratio {itr_ratio:.1f} outside [40,80]"

    Z_avg = 8 * np.pi * (1 + np.sin(k_vals) / k_vals)
    print(f"\n  Z_avg(k): CV = {cv(Z_avg):.1f}%")

    # =================================================================
    # C1: sigma_per_bond * I_tr vs Z_avg (I11)
    # =================================================================
    print(f"\n{'='*65}")
    print(f"Test C1 (I11): sigma_bond(k) * I_tr(k) vs Z_avg(k)")
    print(f"{'='*65}")

    product = sigma_bond * I_tr
    product_n = product / product[0]
    Z_avg_n = Z_avg / Z_avg[0]

    print(f"\n  {'k':>5s}  {'sigma_b':>10s}  {'I_tr':>10s}  {'product':>12s}"
          f"  {'prod_n':>8s}  {'Z_n':>8s}  {'p/Z':>8s}")
    for j in range(len(k_vals)):
        pz = product_n[j] / Z_avg_n[j]
        print(f"  {k_vals[j]:5.2f}  {sigma_bond[j]:10.6f}  {I_tr[j]:10.1f}"
              f"  {product[j]:12.3f}  {product_n[j]:8.4f}"
              f"  {Z_avg_n[j]:8.4f}  {pz:8.4f}")

    cv_pz = cv(product_n / Z_avg_n)
    print(f"\n  CV(product / Z_avg) = {cv_pz:.1f}%")
    print(f"  product drops {product[0]/product[-1]:.1f}x,"
          f" Z_avg varies {Z_avg[0]/Z_avg[-1]:.2f}x")

    if cv_pz < 10:
        print(f"  ==> IDENTITY HOLDS (CV={cv_pz:.1f}%)")
    else:
        print(f"  ==> IDENTITY FAILS (CV={cv_pz:.1f}%)")
        print(f"      Compensation is NOT algebraic sigma_bond * I_tr = Z_avg")

    # =================================================================
    # C2: N_eff = sigma_ring / sigma_per_bond
    # =================================================================
    print(f"\n{'='*65}")
    print(f"Test C2: N_eff(k) = sigma_ring / sigma_bond")
    print(f"  const => incoherent. varies => interference k-dependent.")
    print(f"{'='*65}")

    N_eff = sigma_ring / sigma_bond
    N_eff_n = N_eff / N_eff[0]

    print(f"\n  {'k':>5s}  {'sigma_ring':>10s}  {'sigma_bond':>10s}"
          f"  {'N_eff':>8s}  {'N_eff_n':>8s}")
    for j in range(len(k_vals)):
        print(f"  {k_vals[j]:5.2f}  {sigma_ring[j]:10.4f}"
              f"  {sigma_bond[j]:10.6f}  {N_eff[j]:8.1f}"
              f"  {N_eff_n[j]:8.4f}")

    print(f"\n  N_eff range: {N_eff.min():.1f} to {N_eff.max():.1f}")
    print(f"  N_bonds in ring = {N_bonds}")
    print(f"  CV(N_eff) = {cv(N_eff):.1f}%")
    print(f"  N_eff drops {N_eff[0]/N_eff[-1]:.1f}x from k=0.3 to k=1.5")

    # Compare N_eff shape to I_tr/Z_avg (coherent Born prediction)
    I_over_Z = I_tr / Z_avg
    I_over_Z_n = I_over_Z / I_over_Z[0]
    cv_neff_vs_ioz = cv(N_eff_n / I_over_Z_n)
    print(f"\n  N_eff vs I_tr/Z_avg (coherent Born form factor):")
    print(f"  {'k':>5s}  {'N_eff_n':>8s}  {'(I/Z)_n':>8s}  {'ratio':>8s}")
    for j in range(len(k_vals)):
        print(f"  {k_vals[j]:5.2f}  {N_eff_n[j]:8.4f}"
              f"  {I_over_Z_n[j]:8.4f}"
              f"  {N_eff_n[j]/I_over_Z_n[j]:8.4f}")
    print(f"  CV(N_eff / (I_tr/Z_avg)) = {cv_neff_vs_ioz:.1f}%")
    if cv_neff_vs_ioz < 10:
        print(f"  ==> N_eff follows coherent Born form factor shape")
    else:
        print(f"  ==> N_eff does NOT follow coherent Born form factor")

    # C1 and C2 are complementary views of the same identity:
    #   C1: sigma_bond * I_tr / Z_avg = const  ↔  C2: N_eff / (I_tr/Z_avg) = const
    # Both must pass for the algebraic identity to hold.
    print(f"\n  C1+C2 summary:")
    print(f"    C1: sigma_bond * I_tr / Z_avg = const? CV = {cv_pz:.1f}%")
    print(f"    C2: N_eff / (I_tr/Z_avg) = const?     CV = {cv_neff_vs_ioz:.1f}%")
    print(f"    Both must be <10% for algebraic identity.")

    # =================================================================
    # C3: Functional form of sigma_per_bond (I12)
    # =================================================================
    print(f"\n{'='*65}")
    print(f"Test C3 (I12): Functional form of sigma_bond(k)")
    print(f"{'='*65}")

    s_n = sigma_bond / sigma_bond[0]  # normalized to k=0.3

    candidates = {
        'const': np.ones(len(k_vals)),
        'sin2(k)': np.sin(k_vals)**2 / np.sin(k_vals[0])**2,
        'k^2': k_vals**2 / k_vals[0]**2,
        '1-cos(k)': (1-np.cos(k_vals)) / (1-np.cos(k_vals[0])),
        '1/sin2(k)': (1/np.sin(k_vals)**2) / (1/np.sin(k_vals[0])**2),
        'Z_avg/sin2': (Z_avg/np.sin(k_vals)**2) / (Z_avg[0]/np.sin(k_vals[0])**2),
        'Z_avg/I_tr': (Z_avg/I_tr) / (Z_avg[0]/I_tr[0]),  # C1 hypothesis candidate
    }

    print(f"\n  sigma_bond normalized (k=0.3 = 1):")
    print(f"  {'k':>5s}  {'sigma_n':>8s}", end="")
    for name in candidates:
        w = max(len(name), 8)
        print(f"  {name:>{w}s}", end="")
    print()
    for j in range(len(k_vals)):
        print(f"  {k_vals[j]:5.2f}  {s_n[j]:8.4f}", end="")
        for name, shape in candidates.items():
            w = max(len(name), 8)
            print(f"  {shape[j]:{w}.4f}", end="")
        print()

    print(f"\n  CV(sigma_n / candidate) — lower = better fit:")
    results = []
    for name, shape in candidates.items():
        cv_fit = cv(s_n / shape)
        results.append((cv_fit, name))
        print(f"    {name:>15s}: CV = {cv_fit:.1f}%")
    results.sort()
    print(f"\n  Best fit: {results[0][1]} (CV={results[0][0]:.1f}%)")

    # Power law fit (unweighted log-log — rough guide only)
    log_k = np.log(k_vals)
    log_s = np.log(s_n)
    p_fit, _ = np.polyfit(log_k, log_s, 1)
    print(f"  Power law (unweighted): sigma_bond ~ k^{p_fit:.2f}")

    # =================================================================
    # C4: Non-Born enhancement — per-bond vs ring
    # =================================================================
    print(f"\n{'='*65}")
    print(f"Test C4: Non-Born R(k) per-bond vs ring")
    print(f"{'='*65}")

    # Per-bond Born shape: Z_avg/sin2(k)
    born_shape = Z_avg / np.sin(k_vals)**2
    born_n = born_shape / born_shape[0]

    # R_bond = sigma_bond / sigma_Born_bond (normalized)
    R_bond = s_n / born_n
    R_bond_n = R_bond / R_bond[0]

    # R_ring = sigma_ring / (I_tr/sin2) (normalized to coherent Born)
    born_ring = I_tr / np.sin(k_vals)**2
    born_ring_n = born_ring / born_ring[0]
    sigma_ring_n = sigma_ring / sigma_ring[0]
    R_ring = sigma_ring_n / born_ring_n
    R_ring_n = R_ring / R_ring[0]

    print(f"\n  {'k':>5s}  {'R_bond_n':>10s}  {'R_ring_n':>10s}"
          f"  {'Rb/Rr':>8s}")
    for j in range(len(k_vals)):
        print(f"  {k_vals[j]:5.2f}  {R_bond_n[j]:10.4f}"
              f"  {R_ring_n[j]:10.4f}  {R_bond_n[j]/R_ring_n[j]:8.4f}")

    cv_RbRr = cv(R_bond_n / R_ring_n)
    print(f"\n  R_bond grows {R_bond_n[-1]:.1f}x, R_ring grows {R_ring_n[-1]:.1f}x")
    print(f"  CV(R_bond/R_ring) = {cv_RbRr:.1f}%")
    if cv_RbRr < 10:
        print(f"  ==> Non-Born enhancement SAME shape per-bond and ring")
    elif cv_RbRr < 25:
        print(f"  ==> Non-Born enhancement approximately same shape")
    else:
        print(f"  ==> Non-Born enhancement DIFFERENT per-bond vs ring")
        print(f"      Ring interference modifies the non-Born spectrum")

    t_total = time.time() - t0
    print(f"\n{'='*65}")
    print(f"Total: {t_total:.0f}s ({t_total/60:.1f} min)")
