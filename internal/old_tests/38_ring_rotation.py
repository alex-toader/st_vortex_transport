"""
Route 38 (F13): Ring rotation — orientation vs flat integrand.

Three ring orientations at α=0.30, R=5, L=80:

Results (310s):

  orientation                 bonds  CV%     σ(0.3)   σ(1.5)  verdict
  xy plane (axis=z, standard)   81   7.5%    40.74     2.81   FLAT
  xz plane (axis=y, perp)       81   8.4%    40.79     2.73   FLAT
  yz plane (axis=x, along)      81  42.5%    85.30    46.23   NOT FLAT

  xz vs standard: max rel_diff = 2.7%, mean ratio = 0.990.
  yz vs standard: σ_rot/σ_std = 2.1 (k=0.3) to 16.5 (k=1.5).

  Strain/displacement ratio on x-bonds (yz plane):
    k=0.3: 0.30 (strain << displ)
    k=0.7: 0.69 (comparable)
    k=1.1: 1.05 (strain > displ)
    k=1.5: 1.36 (strain dominates)

  CONCLUSIONS:

  1. Axis perpendicular to propagation (xz, xy): σ_tr identical, CV~7-8%.
     Gauged bonds (y-bonds or z-bonds) are perp to wave → Δu=0 on bond →
     strain coupling null → pure displacement coupling → flat integrand.

  2. Axis along propagation (yz): CV=42.5% NOT FLAT.
     x-bonds have Δu ≠ 0 → strain coupling non-null.
     σ_tr nearly constant in k (85→46, only 1.8×).
     Strain contribution grows with k: 0.30 at k=0.3 to 1.36 at k=1.5.
     At high k, strain dominates displacement → σ_tr(k) flattens →
     integrand sin²(k)·σ_tr grows with k → NOT FLAT.

  3. Definitively confirms strain null mechanism (F12): flat integrand
     requires gauged bonds perpendicular to propagation.
     This explains NN vs NNN: NN z-bonds are always perp to +x,
     while NNN diagonal bonds have a parallel component → strain non-null.

  For paper §4.3: "The flat integrand requires that gauged bonds be
  perpendicular to the wave propagation direction, ensuring a geometric
  strain null (Δu = 0 on each bond). Rotating the ring axis to align
  with propagation activates strain coupling and destroys the flatness
  (CV = 42.5% vs 7.5%)."

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/38_ring_rotation.py
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


def precompute_disk_bonds_yz(L, R_loop):
    """Dirac disk in yz plane at x=cx.

    Ring in yz plane (axis along x). Disk bounded by ring.
    x-bonds from (iz, iy, cx-1) to (iz, iy, cx) are gauged
    where (iy-cy)^2 + (iz-cz)^2 <= R_loop^2.

    Returns (iz_disk, iy_disk): 1D integer arrays.
    """
    cy = cz = L // 2
    iz_all, iy_all = np.mgrid[0:L, 0:L]
    mask = (iy_all - cy)**2 + (iz_all - cz)**2 <= R_loop**2
    iz_disk, iy_disk = np.where(mask)
    return iz_disk, iy_disk


def precompute_disk_bonds_xz(L, R_loop):
    """Dirac disk in xz plane at y=cy.

    Ring in xz plane (axis along y). Disk bounded by ring.
    y-bonds from (iz, cy-1, ix) to (iz, cy, ix) are gauged
    where (ix-cx)^2 + (iz-cz)^2 <= R_loop^2.

    Returns (iz_disk, ix_disk): 1D integer arrays.
    """
    cx = cz = L // 2
    iz_all, ix_all = np.mgrid[0:L, 0:L]
    mask = (ix_all - cx)**2 + (iz_all - cz)**2 <= R_loop**2
    iz_disk, ix_disk = np.where(mask)
    return iz_disk, ix_disk


def make_vortex_yz(alpha, R_loop, L, K1=1.0, K2=0.5):
    """Vortex ring in yz plane (axis along x).

    Dirac disk crossed by x-bonds: (iz, iy, cx-1) <-> (iz, iy, cx).
    Wave in +x: u varies along x → strain coupling non-null.
    Peierls R(2πα) acts on (ux, uy) as in standard ring.
    """
    cx = L // 2
    iz_disk, iy_disk = precompute_disk_bonds_yz(L, R_loop)

    phi = 2.0 * np.pi * (float(alpha) % 1.0)
    cm1 = np.cos(phi) - 1.0
    s_phi = np.sin(phi)
    if abs(s_phi) < 1e-12:
        s_phi = 0.0
    if abs(cm1) < 1e-12:
        cm1 = 0.0

    ix_lo = cx - 1
    ix_hi = cx
    n_bonds = len(iz_disk)

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if cm1 == 0.0 and s_phi == 0.0:
            return fx, fy, fz

        ux_hi = ux[iz_disk, iy_disk, ix_hi]
        uy_hi = uy[iz_disk, iy_disk, ix_hi]
        ux_lo = ux[iz_disk, iy_disk, ix_lo]
        uy_lo = uy[iz_disk, iy_disk, ix_lo]

        # Lower (x-1) site gets K1*(R - I)*u_upper
        fx[iz_disk, iy_disk, ix_lo] += K1 * (cm1 * ux_hi - s_phi * uy_hi)
        fy[iz_disk, iy_disk, ix_lo] += K1 * (s_phi * ux_hi + cm1 * uy_hi)

        # Upper (x) site gets K1*(R^T - I)*u_lower
        fx[iz_disk, iy_disk, ix_hi] += K1 * (cm1 * ux_lo + s_phi * uy_lo)
        fy[iz_disk, iy_disk, ix_hi] += K1 * (-s_phi * ux_lo + cm1 * uy_lo)

        return fx, fy, fz

    force_fn.n_bonds = n_bonds
    force_fn.orientation = "yz_plane (axis=x)"
    return force_fn


def make_vortex_xz(alpha, R_loop, L, K1=1.0, K2=0.5):
    """Vortex ring in xz plane (axis along y).

    Dirac disk crossed by y-bonds: (iz, cy-1, ix) <-> (iz, cy, ix).
    Wave in +x: u constant in y → strain coupling null (same as standard).
    Peierls R(2πα) acts on (ux, uy) as in standard ring.
    """
    cy = L // 2
    iz_disk, ix_disk = precompute_disk_bonds_xz(L, R_loop)

    phi = 2.0 * np.pi * (float(alpha) % 1.0)
    cm1 = np.cos(phi) - 1.0
    s_phi = np.sin(phi)
    if abs(s_phi) < 1e-12:
        s_phi = 0.0
    if abs(cm1) < 1e-12:
        cm1 = 0.0

    iy_lo = cy - 1
    iy_hi = cy
    n_bonds = len(iz_disk)

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if cm1 == 0.0 and s_phi == 0.0:
            return fx, fy, fz

        ux_hi = ux[iz_disk, iy_hi, ix_disk]
        uy_hi = uy[iz_disk, iy_hi, ix_disk]
        ux_lo = ux[iz_disk, iy_lo, ix_disk]
        uy_lo = uy[iz_disk, iy_lo, ix_disk]

        # Lower (y-1) site gets K1*(R - I)*u_upper
        fx[iz_disk, iy_lo, ix_disk] += K1 * (cm1 * ux_hi - s_phi * uy_hi)
        fy[iz_disk, iy_lo, ix_disk] += K1 * (s_phi * ux_hi + cm1 * uy_hi)

        # Upper (y) site gets K1*(R^T - I)*u_lower
        fx[iz_disk, iy_hi, ix_disk] += K1 * (cm1 * ux_lo + s_phi * uy_lo)
        fy[iz_disk, iy_hi, ix_disk] += K1 * (-s_phi * ux_lo + cm1 * uy_lo)

        return fx, fy, fz

    force_fn.n_bonds = n_bonds
    force_fn.orientation = "xz_plane (axis=y)"
    return force_fn


if __name__ == '__main__':
    t0 = time.time()

    from gauge_3d import make_vortex_force, precompute_disk_bonds

    print("Route 38 (F13): Ring rotation — strain null mechanism test")
    print(f"  α={ALPHA}, R={R_LOOP}, L={L}")
    print(f"  k = {list(k_vals)}")
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

    # Three orientations
    # Standard vortex from gauge_3d doesn't have n_bonds attribute — wrap it
    def make_standard():
        f = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)
        _, ix_d = precompute_disk_bonds(L, R_LOOP)
        f.n_bonds = len(ix_d)
        f.orientation = "xy_plane (axis=z)"
        return f

    cases = [
        ("xy plane (axis=z, standard)", make_standard),
        ("xz plane (axis=y, perp to prop)",
         lambda: make_vortex_xz(ALPHA, R_LOOP, L, K1, K2)),
        ("yz plane (axis=x, along prop)",
         lambda: make_vortex_yz(ALPHA, R_LOOP, L, K1, K2)),
    ]

    all_results = []

    for label, factory in cases:
        f_test = factory()
        print(f"\n{label}  ({f_test.n_bonds} bonds)")

        t1 = time.time()
        sigma_tr = np.zeros(len(k_vals))
        for j, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_test, ux0.copy(), vx0.copy(), gamma, DT, ns,
                            rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
            f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                   ref['ux'], ref['uy'], ref['uz'], r_m)
            _, st = integrate_sigma_3d(f2, thetas, phis)
            assert st > -1e-6, f"σ_tr={st} at k={k0}, {label}"
            sigma_tr[j] = max(st, 0.0)
        dt = time.time() - t1

        integrand = np.sin(k_vals)**2 * sigma_tr
        if np.mean(integrand) > 1e-10:
            cv = np.std(integrand) / np.mean(integrand) * 100
        else:
            cv = 0.0

        all_results.append({
            'label': label, 'sigma_tr': sigma_tr.copy(),
            'integrand': integrand.copy(), 'cv': cv, 'dt': dt,
            'n_bonds': f_test.n_bonds,
        })

        print(f"  σ_tr: " + "  ".join(f"{s:.4f}" for s in sigma_tr))
        print(f"  sin²(k)·σ_tr CV = {cv:.1f}%  ({dt:.0f}s)")

    # Summary
    print()
    print("=" * 70)
    print("Summary: ring orientation and flat integrand")
    print("=" * 70)
    print(f"  {'orientation':>35s}  {'bonds':>5s}  {'CV%':>6s}  {'verdict':>10s}  "
          f"{'σ(0.3)':>8s}  {'σ(1.5)':>8s}")
    print(f"  {'-'*80}")
    for r in all_results:
        cv = r['cv']
        verdict = "FLAT" if cv < 10 else ("marginal" if cv < 20 else "NOT FLAT")
        print(f"  {r['label']:>35s}  {r['n_bonds']:5d}  {cv:5.1f}%  {verdict:>10s}  "
              f"{r['sigma_tr'][0]:8.4f}  {r['sigma_tr'][-1]:8.4f}")

    # xz vs standard consistency check
    print()
    s_std = all_results[0]['sigma_tr']
    s_xz = all_results[1]['sigma_tr']
    rel_diff_xz = np.abs(s_xz - s_std) / s_std
    print(f"  xz vs standard: max rel_diff = {rel_diff_xz.max():.4f} "
          f"(at k={k_vals[np.argmax(rel_diff_xz)]:.1f})")
    if rel_diff_xz.max() > 0.05:
        print("  WARNING: xz and standard differ by >5% — check implementation")
    else:
        print("  OK: same mechanism (strain null on perp bonds)")

    # Ratio analysis
    print()
    for i in [1, 2]:
        s_rot = all_results[i]['sigma_tr']
        ratios = s_rot / s_std
        print(f"  {all_results[i]['label']}:")
        print(f"    σ_rot/σ_std: " + "  ".join(f"{r:.3f}" for r in ratios))
        print(f"    mean ratio = {np.mean(ratios):.3f}")

    # Strain/displacement ratio estimate for axis=x
    print()
    print("  Strain/displacement ratio estimate (axis=x, yz plane):")
    print("    displacement ~ K1*cm1*ux_hi")
    print("    strain ~ K1*cm1*(ux_hi - ux_lo) ~ K1*cm1*(e^{ik}-1)*ux")
    print("    ratio = |e^{ik}-1| = 2*sin(k/2)")
    for k0 in k_vals:
        r = 2 * np.sin(k0 / 2)
        print(f"    k={k0:.1f}: strain/displ = {r:.3f}  "
              f"({'strain << displ' if r < 0.5 else 'comparable' if r < 1.0 else 'strain > displ'})")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
