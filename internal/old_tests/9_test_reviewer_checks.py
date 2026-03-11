"""
Reviewer-requested verification tests for Phase 3-4.

Results (L=60, α=0.3, R=5):
  T1 uz_zero:           σ_tr=0.00e+00 exact                    [PASS]
  T2 y_incidence:       σ(+x)=28.16, σ(+y)=28.20, ratio=1.001 [PASS]
  T3 sphere_conv:       13×24 σ_tr=14.23, 25×48 σ_tr=14.42, diff=1.3%  [PASS]
  T4 r_m_independence:  r_m=10→13.57, 12→13.24, 15→14.23, spread=7.3%  [PASS <15%]
  T5 kR_universality:   σ_tr/R² at kR=1.5: (k=0.5,R=3)=0.75 vs (k=0.3,R=5)=1.82
                         ratio=2.43 — FAIL (near-field at k=0.3, r_m/λ=0.72)

Notes:
  - T2: compares σ (total power), not σ_tr — because integrate_sigma_3d hardcodes
    transport kernel cos_θs = sinθ cosφ for +x incidence. A proper test would use
    an adapted kernel (cos_θs = sinθ sinφ for +y). Current test verifies that
    scattered power is isotropic, not that σ_tr is direction-independent.
  - T4: near-field systematic ~7% at k=0.5 (r_m/λ = 0.8-1.2). All r_m values
    are below far-field threshold (r_m/λ >> 1). At r_m/λ > 3 spread should
    drop below ~1%. Threshold 15% is practical, not derived from theory.
  - T5: kR universality failure has a key implication — if σ_tr/R² does not
    collapse vs kR, then κ depends on R_loop and is NOT an intrinsic property
    of the vortex. The integral κ = ∫ ... σ_tr dk is only well-defined if
    σ_tr(k,R) = R² F(kR) holds. Current failure is attributed to near-field
    (k=0.3, r_m/λ=0.72). Needs L≥160, r_m≥40 to test in far-field regime.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/9_test_reviewer_checks.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from gauge_3d import make_vortex_force
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           compute_sphere_f2, integrate_sigma_3d,
                           estimate_n_steps_3d)

K1, K2 = 1.0, 0.5


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


def test_kR_universality():
    """σ_tr/R² should collapse vs kR for different R values.
    Test at matched kR values from different (k, R) pairs.
    Uses existing diagnostic data format: fixed k, varying R.
    """
    L, DW, DS, DT, sx = 60, 12, 1.5, 0.25, 8.0
    x_start = DW + 5
    r_m = 15
    ALPHA = 0.3
    thetas = np.linspace(0, np.pi, 13)
    phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
    gamma = make_damping_3d(L, DW, DS)

    # Measure σ_tr at (k, R) pairs that share kR values
    # kR=1.5: (k=0.5, R=3) and (k=0.3, R=5)
    # kR=2.5: (k=0.5, R=5) and (k=5/7, R=7)... not clean
    # Simpler: fixed k=0.5, R=3,5,7 → kR=1.5, 2.5, 3.5
    # Then fixed k=0.3, R=5,7 → kR=1.5, 2.1
    # Compare σ_tr/R² at same kR

    pairs = [(0.5, 3), (0.3, 5), (0.5, 5), (0.5, 7)]
    results = []
    for k0, R in pairs:
        ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
        ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma, DT, ns,
                          rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f_def = make_vortex_force(ALPHA, R, L, K1, K2)
        d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma, DT, ns,
                        rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        sigma, sigma_tr = integrate_sigma_3d(f2, thetas, phis)
        kR = k0 * R
        scaled = sigma_tr / R**2
        results.append((k0, R, kR, sigma_tr, scaled))
        print(f"  k={k0}, R={R}: kR={kR:.1f}, σ_tr={sigma_tr:.2f}, "
              f"σ_tr/R²={scaled:.4f}")

    # Compare at kR=1.5: (k=0.5,R=3) vs (k=0.3,R=5)
    s1 = [r for r in results if abs(r[2] - 1.5) < 0.01]
    if len(s1) == 2:
        ratio = s1[1][4] / s1[0][4]
        print(f"  kR=1.5 universality: ratio={ratio:.3f}", end="")
        if abs(ratio - 1) < 0.3:
            print(" [PASS — <30%]")
        else:
            print(f" [FAIL — {abs(ratio-1)*100:.0f}% deviation]")


def test_y_incidence():
    """+y incidence should give same σ_tr as +x (ring axisymmetric on z).
    Wave propagating in +y with uy-polarization (longitudinal).
    By z-axis symmetry of ring, σ_tr(+y) = σ_tr(+x).
    """
    L, DW, DS, DT, sx = 60, 12, 1.5, 0.25, 8.0
    x_start = DW + 5
    r_m = 15
    k0 = 0.5
    R_LOOP = 5
    ALPHA = 0.3
    c = np.sqrt(K1 + 4 * K2)
    thetas = np.linspace(0, np.pi, 13)
    phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
    gamma = make_damping_3d(L, DW, DS)
    ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)

    # +x incidence (standard)
    ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
    ref_x = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma, DT, ns,
                        rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    f_def = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)
    def_x = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma, DT, ns,
                        rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    f2_x = compute_sphere_f2(def_x['ux'], def_x['uy'], def_x['uz'],
                              ref_x['ux'], ref_x['uy'], ref_x['uz'], r_m)
    _, str_x = integrate_sigma_3d(f2_x, thetas, phis)

    # +y incidence: uy wave packet propagating in +y
    omega_k = 2 * np.sin(k0 / 2) * c
    iy_arr = np.arange(L, dtype=float)
    env = np.exp(-((iy_arr - x_start) ** 2) / (2 * sx ** 2))
    uy_1d = env * np.cos(k0 * iy_arr)
    vy_1d = omega_k * env * np.sin(k0 * iy_arr)
    uy0 = np.broadcast_to(uy_1d[np.newaxis, :, np.newaxis], (L, L, L)).copy()
    vy0 = np.broadcast_to(vy_1d[np.newaxis, :, np.newaxis], (L, L, L)).copy()
    zeros = np.zeros((L, L, L))

    ref_y = run_fdtd_3d(force_plain, zeros.copy(), zeros.copy(), gamma, DT, ns,
                        rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns,
                        uy0=uy0.copy(), vy0=vy0.copy())
    def_y = run_fdtd_3d(f_def, zeros.copy(), zeros.copy(), gamma, DT, ns,
                        rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns,
                        uy0=uy0.copy(), vy0=vy0.copy())
    f2_y = compute_sphere_f2(def_y['ux'], def_y['uy'], def_y['uz'],
                              ref_y['ux'], ref_y['uy'], ref_y['uz'], r_m)
    _, str_y = integrate_sigma_3d(f2_y, thetas, phis)

    # Compare σ (total), not σ_tr — integrate_sigma_3d hardcodes
    # transport kernel cos_θs = sinθ cosφ for +x incidence.
    # σ = ∫ f² dΩ has no direction dependence.
    sig_x, _ = integrate_sigma_3d(f2_x, thetas, phis)
    sig_y, _ = integrate_sigma_3d(f2_y, thetas, phis)
    ratio = sig_y / sig_x if sig_x > 0 else float('inf')
    print(f"  y_incidence: σ(+x)={sig_x:.2f}, σ(+y)={sig_y:.2f}, "
          f"ratio={ratio:.4f}", end="")
    assert abs(ratio - 1) < 0.02, f"σ ratio {ratio:.4f} deviates >2% from 1"
    print(" [PASS]")


def test_uz_zero():
    """uz-polarized wave should not scatter (Option B).
    Option B: gauge R(2πα) acts on (ux, uy) only → uz decoupled.
    """
    L, DW, DS, DT, sx = 60, 12, 1.5, 0.25, 8.0
    x_start = DW + 5
    r_m = 15
    k0 = 0.5
    thetas = np.linspace(0, np.pi, 13)
    phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
    gamma = make_damping_3d(L, DW, DS)

    # uz-polarized wave packet
    ux_env, vx_env = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
    zeros = np.zeros_like(ux_env)
    ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)

    ref = run_fdtd_3d(force_plain, zeros.copy(), zeros.copy(), gamma, DT, ns,
                      rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns,
                      uz0=ux_env.copy(), vz0=vx_env.copy())
    f_def = make_vortex_force(0.3, 5, L, K1, K2)
    d = run_fdtd_3d(f_def, zeros.copy(), zeros.copy(), gamma, DT, ns,
                    rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns,
                    uz0=ux_env.copy(), vz0=vx_env.copy())

    f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                           ref['ux'], ref['uy'], ref['uz'], r_m)
    sigma, sigma_tr = integrate_sigma_3d(f2, thetas, phis)

    print(f"  uz_zero: σ_tr={sigma_tr:.2e}, σ={sigma:.2e}", end="")
    assert sigma_tr < 1e-10, f"uz scattering nonzero: σ_tr={sigma_tr}"
    print(" [PASS]")


def test_sphere_convergence():
    """σ_tr at 13×24 vs 25×48 angular grid — check aliasing.
    Same FDTD run, different sphere grids → different f² integration.
    Needs single FDTD pair, then re-sample at two resolutions.
    """
    L, DW, DS, DT, sx = 60, 12, 1.5, 0.25, 8.0
    x_start = DW + 5
    r_m = 15
    k0 = 0.5
    R_LOOP = 5
    ALPHA = 0.3
    gamma = make_damping_3d(L, DW, DS)

    results = {}
    for label, n_th, n_ph in [('coarse', 13, 24), ('fine', 25, 48)]:
        thetas = np.linspace(0, np.pi, n_th)
        phis = np.linspace(0, 2 * np.pi, n_ph, endpoint=False)
        iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
        ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)

        ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma, DT, ns,
                          rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f_def = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)
        d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma, DT, ns,
                        rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)

        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        sigma, sigma_tr = integrate_sigma_3d(f2, thetas, phis)
        results[label] = (sigma, sigma_tr, n_th, n_ph)

    sc, stc, _, _ = results['coarse']
    sf, stf, _, _ = results['fine']
    rel = abs(stf - stc) / stf if stf > 0 else 0
    print(f"  sphere_conv: 13×24 σ_tr={stc:.2f}, 25×48 σ_tr={stf:.2f}, "
          f"diff={rel*100:.1f}%", end="")
    assert rel < 0.10, f"sphere aliasing {rel*100:.1f}% > 10%"
    print(" [PASS]")


def test_r_m_independence():
    """σ_tr should be constant vs measurement radius r_m.
    At k=0.5, λ=12.6 — far-field needs r_m >> λ.
    L=60 limits r_m to ~18 (clear of PML at DW=12).
    This test documents the near-field systematic, not a pass/fail.
    """
    L, DW, DS, DT, sx = 60, 12, 1.5, 0.25, 8.0
    x_start = DW + 5
    k0 = 0.5
    R_LOOP = 5
    ALPHA = 0.3
    thetas = np.linspace(0, np.pi, 13)
    phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    gamma = make_damping_3d(L, DW, DS)

    results = []
    for r_m in [10, 12, 15]:
        iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
        ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)

        ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma, DT, ns,
                          rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f_def = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)
        d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma, DT, ns,
                        rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)

        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        sigma, sigma_tr = integrate_sigma_3d(f2, thetas, phis)
        results.append((r_m, sigma_tr))
        print(f"  r_m={r_m}: σ_tr={sigma_tr:.2f} (r_m/λ={r_m*k0/(2*np.pi):.2f})")

    # Report variation as diagnostic (not strict pass/fail)
    vals = [v for _, v in results]
    spread = (max(vals) - min(vals)) / np.mean(vals)
    print(f"  spread: {spread*100:.1f}%", end="")
    if spread < 0.15:
        print(" [PASS — <15%]")
    else:
        print(f" [WARN — near-field systematic]")


if __name__ == '__main__':
    test_uz_zero()
    test_y_incidence()
    test_r_m_independence()
    test_sphere_convergence()
    test_kR_universality()
