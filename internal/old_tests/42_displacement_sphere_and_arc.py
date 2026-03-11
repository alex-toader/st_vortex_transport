"""
Route 42 (F14 + F22): Displacement coupling on sphere + half-ring arc.

Part A — F14: Displacement coupling on sphere (1302 bonds).
Mass sphere (file 34) uses strain coupling dK*(u_nbr - u_site): CV=89-152%.
This test uses displacement coupling K_eff*u_nbr on the SAME bonds.
If CV < 10% → displacement coupling universally flat (any geometry).
If CV > 10% → geometry matters, ring is special.

Part B — F22: Half-ring (semicircular arc).
Gauge only bonds in upper semicircle of Dirac disk (iy >= cy).
Tests coherent vs incoherent scattering:
  σ(half) ≈ σ(full)/2 → incoherent (each bond independent)
  σ(half) ≈ σ(full)/4 → coherent (amplitude halved → σ quartered)
  σ(half) ≈ σ(full)   → closure matters (topological)

Results:

Part A — Displacement sphere: NOT FLAT at any K_eff.
  K=-0.3: CV=33.9%,  K=-0.5: CV=34.8%,  K=-1.0: CV=45.2%,  K=-1.3: CV=46.4%
  Displacement coupling is NOT sufficient for flat integrand — ring geometry
  is essential. Same bond type (displacement), same coupling sign, but sphere
  gives CV≈35-46% while ring gives CV≈7.5-9%.
  Completes the coupling×geometry table:
    displacement + ring   → FLAT (7.5%)    [file 37]
    displacement + sphere → NOT FLAT (35%) [this file]
    strain + ring         → ZERO (null)    [file 37]
    strain + sphere       → NOT FLAT (112%) [file 34]

Part B — Half-ring vs full ring: INCOHERENT (σ ∝ N_bonds).
  Full ring: 81 bonds, CV=7.5%.  Half ring: 46 bonds, CV=3.6%.
  Mean σ_half/σ_full = 0.50, closest to incoherent prediction (0.57).
  Stationary phase (N^{3/2}): 0.43.  Coherent (N²): 0.32.
  Ratio drifts from 0.45 (low k) to 0.54 (high k) — mild k-dependence.
  Flatness robust to arc reduction (half-ring CV even better than full).
  11 bonds on diameter (iy=cy).

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/42_displacement_sphere_and_arc.py
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
R_LOOP = 5
ALPHA = 0.30

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])


# ── Part A: Displacement coupling on sphere ──

def make_displacement_sphere(K_eff, R_defect, L, K1=1.0, K2=0.5):
    """Displacement coupling on NN bonds of a sphere.

    On each NN bond with both endpoints inside sphere:
      site_lo gets: K_eff * u_hi
      site_hi gets: K_eff * u_lo

    This is displacement coupling (force ∝ u_neighbor), NOT strain coupling.
    Same bond set as make_mass_sphere_force in file 34.
    Acts on (ux, uy, uz) — all 3 components.
    """
    cx = cy = cz = L // 2

    iz, iy, ix = np.mgrid[0:L, 0:L, 0:L]
    r2 = (ix - cx)**2 + (iy - cy)**2 + (iz - cz)**2
    inside = r2 <= R_defect**2

    # Bond masks: both endpoints inside
    mask_px = inside[:, :, :-1] & inside[:, :, 1:]
    mask_py = inside[:, :-1, :] & inside[:, 1:, :]
    mask_pz = inside[:-1, :, :] & inside[1:, :, :]

    n_bonds = int(mask_px.sum() + mask_py.sum() + mask_pz.sum())

    # Precompute bond indices for efficiency
    pz_lo, py_px, px_px = np.where(mask_px)
    pz_py, py_lo, px_py = np.where(mask_py)
    pz_lo_z, py_pz, px_pz = np.where(mask_pz)

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if K_eff == 0.0:
            return fx, fy, fz

        for u, f in [(ux, fx), (uy, fy), (uz, fz)]:
            # +x bonds: lo=(iz,iy,ix), hi=(iz,iy,ix+1)
            u_hi = u[pz_lo, py_px, px_px + 1]
            u_lo = u[pz_lo, py_px, px_px]
            f[pz_lo, py_px, px_px] += K_eff * u_hi
            f[pz_lo, py_px, px_px + 1] += K_eff * u_lo

            # +y bonds: lo=(iz,iy,ix), hi=(iz,iy+1,ix)
            u_hi = u[pz_py, py_lo + 1, px_py]
            u_lo = u[pz_py, py_lo, px_py]
            f[pz_py, py_lo, px_py] += K_eff * u_hi
            f[pz_py, py_lo + 1, px_py] += K_eff * u_lo

            # +z bonds: lo=(iz,iy,ix), hi=(iz+1,iy,ix)
            u_hi = u[pz_lo_z + 1, py_pz, px_pz]
            u_lo = u[pz_lo_z, py_pz, px_pz]
            f[pz_lo_z, py_pz, px_pz] += K_eff * u_hi
            f[pz_lo_z + 1, py_pz, px_pz] += K_eff * u_lo

        return fx, fy, fz

    force_fn.n_bonds = n_bonds
    force_fn.coupling = "displacement"
    return force_fn


# ── Part B: Half-ring (semicircular arc) ──

def make_half_ring_force(alpha, R_loop, L, K1=1.0, K2=0.5):
    """Peierls vortex on upper semicircle only (iy >= cy).

    Same as make_vortex_force but only gauges bonds where iy >= cy.
    Number of bonds ≈ half of full ring.
    """
    cz = L // 2
    cy = L // 2
    iy_disk, ix_disk = precompute_disk_bonds(L, R_loop)

    # Filter: keep only bonds with iy >= cy (upper semicircle)
    mask_upper = iy_disk >= cy
    iy_half = iy_disk[mask_upper]
    ix_half = ix_disk[mask_upper]

    n_bonds_full = len(iy_disk)
    n_bonds_half = len(iy_half)

    phi = 2.0 * np.pi * (float(alpha) % 1.0)
    cm1 = np.cos(phi) - 1.0
    s_phi = np.sin(phi)
    if abs(s_phi) < 1e-12:
        s_phi = 0.0
    if abs(cm1) < 1e-12:
        cm1 = 0.0

    iz_lo = cz - 1
    iz_hi = cz

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if cm1 == 0.0 and s_phi == 0.0:
            return fx, fy, fz

        ux_hi = ux[iz_hi, iy_half, ix_half]
        uy_hi = uy[iz_hi, iy_half, ix_half]
        ux_lo = ux[iz_lo, iy_half, ix_half]
        uy_lo = uy[iz_lo, iy_half, ix_half]

        fx[iz_lo, iy_half, ix_half] += K1 * (cm1 * ux_hi - s_phi * uy_hi)
        fy[iz_lo, iy_half, ix_half] += K1 * (s_phi * ux_hi + cm1 * uy_hi)

        fx[iz_hi, iy_half, ix_half] += K1 * (cm1 * ux_lo + s_phi * uy_lo)
        fy[iz_hi, iy_half, ix_half] += K1 * (-s_phi * ux_lo + cm1 * uy_lo)

        return fx, fy, fz

    n_on_diameter = int(np.sum(iy_disk == cy))

    force_fn.n_bonds_full = n_bonds_full
    force_fn.n_bonds = n_bonds_half
    force_fn.n_on_diameter = n_on_diameter
    force_fn.bond_fraction = n_bonds_half / n_bonds_full
    force_fn.coupling = "Peierls half-ring"
    return force_fn


if __name__ == '__main__':
    t0 = time.time()

    print("Route 42: Displacement sphere (F14) + Half-ring arc (F22)")
    print(f"  R={R_LOOP}, α={ALPHA}, L={L}")
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

    # ── Build all test cases ──
    # (label, force_fn, category)
    cases = []

    # Part A: displacement sphere
    for K_eff in [-0.3, -0.5, -1.0, -1.3]:
        f = make_displacement_sphere(K_eff, R_LOOP, L, K1, K2)
        cases.append((f"displ sphere K={K_eff}", f, "A"))

    # Part B: half-ring + full ring reference
    f_full = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)
    iy_d, ix_d = precompute_disk_bonds(L, R_LOOP)
    f_full.n_bonds = len(iy_d)
    f_full.coupling = "Peierls full ring"
    cases.append(("Peierls full ring", f_full, "B"))

    f_half = make_half_ring_force(ALPHA, R_LOOP, L, K1, K2)
    cases.append(("Peierls half ring", f_half, "B"))

    # ── Run all cases ──
    all_results = []

    for label, f_test, cat in cases:
        n = f_test.n_bonds
        print(f"\n{label}  ({n} bonds, {f_test.coupling})")

        t1 = time.time()
        sigma_tr = np.zeros(len(k_vals))
        unstable = False
        for j, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_test, ux0.copy(), vx0.copy(), gamma, DT, ns,
                            rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
            f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                   ref['ux'], ref['uy'], ref['uz'], r_m)
            _, st = integrate_sigma_3d(f2, thetas, phis)
            if st > 1e6:
                print(f"  WARNING: σ_tr={st:.2e} at k={k0} — UNSTABLE")
                unstable = True
                sigma_tr[j] = st
                break
            sigma_tr[j] = max(st, 0.0)
        dt = time.time() - t1

        integrand = np.sin(k_vals)**2 * sigma_tr
        if np.mean(integrand) > 1e-10 and not unstable:
            cv = np.std(integrand) / np.mean(integrand) * 100
        else:
            cv = -1.0  # unstable marker

        all_results.append({
            'label': label, 'cat': cat, 'sigma_tr': sigma_tr.copy(),
            'integrand': integrand.copy(), 'cv': cv, 'dt': dt,
            'n_bonds': n, 'unstable': unstable,
        })

        if not unstable:
            print(f"  σ_tr: " + "  ".join(f"{s:.4f}" for s in sigma_tr))
            print(f"  sin²(k)·σ_tr CV = {cv:.1f}%  ({dt:.0f}s)")
        else:
            print(f"  UNSTABLE — skipping CV analysis  ({dt:.0f}s)")

    # ── Part A Summary ──
    print()
    print("=" * 75)
    print("Part A: Displacement coupling on sphere vs ring")
    print("=" * 75)
    print(f"  {'label':>25s}  {'bonds':>6s}  {'CV%':>7s}  {'verdict':>10s}"
          f"  {'σ(0.3)':>8s}  {'σ(1.5)':>8s}")
    print(f"  {'-'*70}")
    for r in all_results:
        if r['cat'] != 'A':
            continue
        cv = r['cv']
        if r['unstable']:
            verdict = "UNSTABLE"
        elif cv < 10:
            verdict = "FLAT"
        elif cv < 20:
            verdict = "marginal"
        else:
            verdict = "NOT FLAT"
        print(f"  {r['label']:>25s}  {r['n_bonds']:6d}  {cv:6.1f}%  {verdict:>10s}"
              f"  {r['sigma_tr'][0]:8.4f}  {r['sigma_tr'][-1]:8.4f}")

    # Reference from file 34 (strain coupling)
    print(f"  {'— strain sphere (file 34, same geometry) —':>55s}")
    print(f"  {'strain sphere void':>25s}  {'1302':>6s}  {'112%':>7s}  {'NOT FLAT':>10s}"
          f"  {'1.92':>8s}  {'51.12':>8s}")
    print(f"  {'— vortex ring reference —':>45s}")
    print(f"  {'Peierls ring NN α=0.30':>25s}  {'81':>6s}  {'7.5%':>7s}  {'FLAT':>10s}"
          f"  {'40.74':>8s}  {'2.81':>8s}")
    print(f"  {'displ ring K=-1.3 (f37)':>25s}  {'81':>6s}  {'8.9%':>7s}  {'FLAT':>10s}"
          f"  {'44.98':>8s}  {'2.97':>8s}")

    # ── Part B Summary ──
    print()
    print("=" * 75)
    print("Part B: Half-ring vs full ring — coherent or incoherent?")
    print("=" * 75)

    r_full = [r for r in all_results if r['label'] == "Peierls full ring"][0]
    r_half = [r for r in all_results if r['label'] == "Peierls half ring"][0]

    frac = f_half.bond_fraction
    print(f"  Full ring:  {r_full['n_bonds']} bonds, CV = {r_full['cv']:.1f}%")
    print(f"  Half ring:  {r_half['n_bonds']} bonds, CV = {r_half['cv']:.1f}%")
    print(f"  Bonds on diameter (iy=cy): {f_half.n_on_diameter}")
    print(f"  Bond fraction: {r_half['n_bonds']}/{r_full['n_bonds']}"
          f" = {frac:.4f}")
    print()

    # Flatness comparison
    print(f"  Flatness: full CV={r_full['cv']:.1f}%,"
          f" half CV={r_half['cv']:.1f}%")
    flat_robust = abs(r_half['cv'] - r_full['cv']) < 5
    print(f"  Flatness robust to arc reduction: "
          f"{'YES' if flat_robust else 'NO'}")
    print()

    s_full = r_full['sigma_tr']
    s_half = r_half['sigma_tr']
    ratio = s_half / s_full

    # Three predictions
    r_incoh = frac           # σ ∝ N
    r_stat = frac**1.5       # σ ∝ N^{3/2} (stationary phase)
    r_coh = frac**2          # σ ∝ N²

    print(f"  {'k':>5s}  {'σ_full':>8s}  {'σ_half':>8s}  {'ratio':>7s}"
          f"  {'incoh':>7s}  {'stat':>7s}  {'coh':>7s}")
    for j, k0 in enumerate(k_vals):
        print(f"  {k0:5.1f}  {s_full[j]:8.4f}  {s_half[j]:8.4f}  {ratio[j]:7.4f}"
              f"  {r_incoh:7.4f}  {r_stat:7.4f}  {r_coh:7.4f}")

    mean_ratio = np.mean(ratio)
    print(f"\n  Mean ratio = {mean_ratio:.4f}")
    print(f"  Incoherent (σ∝N):         {r_incoh:.4f}")
    print(f"  Stationary phase (σ∝N^{{3/2}}): {r_stat:.4f}")
    print(f"  Coherent (σ∝N²):          {r_coh:.4f}")

    dists = {
        'INCOHERENT': abs(mean_ratio - r_incoh),
        'STATIONARY PHASE': abs(mean_ratio - r_stat),
        'COHERENT': abs(mean_ratio - r_coh),
    }
    best = min(dists, key=dists.get)
    print(f"  → Closest to {best}")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
