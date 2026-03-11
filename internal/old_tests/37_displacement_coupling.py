"""
Route 37 (F5+F7): Displacement vs strain coupling on ring.

Constructs scatterers on the same ring geometry (R=5, Dirac disk) with:
  - Pure displacement coupling: force += K_eff * u_neighbor (no strain)
  - Pure strain coupling: force += dK * (u_neighbor - u_site)
Both act on (ux, uy) only — apples-to-apples comparison.

Tests K_eff = 0 (sanity), -0.5, -1.0, -1.3 (matching cm1 at α≈0.3),
+0.5, +1.3 (positive). Strain: dK = -1.3, +1.0.
NN only, R=5, k=0.3-1.5 (7 pts), standard grid.

Results (L=80, r_m=20, 679s):

  Sanity (K=0): σ_tr = 0.00 exact. PASS.

  Displacement coupling:
    K_eff    CV%      verdict    σ(0.3)   σ(1.5)   note
    -0.5    33.1%    NOT FLAT    16.37    0.57
    -1.0    15.6%    marginal    35.74    1.94
    -1.3     8.9%    FLAT        44.98    2.97     matches Peierls diag α=0.3
    +0.5   135.4%    NOT FLAT   133.28    0.78     amplifies low-k
    +1.3   102.4%    UNSTABLE    ~10⁹     ~10⁸     system blows up

  Strain coupling on ring:
    dK      CV%      verdict    σ(0.3)   σ(1.5)
    -1.3     0.0%    zero       0.0000   0.0000
    +1.0     0.0%    zero       0.0000   0.0000

  CONCLUSIONS:

  1. Strain coupling on z-bonds of Dirac disk gives ZERO scattering.
     Reason: packet propagates in +x, z-bonds have u_hi = u_lo
     (wave constant in z) → Δu = 0. Strain coupling sees no perturbation.
     This is a GEOMETRIC NULL — independent of dK magnitude.

  2. Displacement coupling alone CAN produce flat integrand.
     K=-1.3 gives CV=8.9%, matching Peierls diagonal (8.8%) exactly.
     Confirms: displacement coupling is sufficient for flatness.

  3. Flatness requires NEGATIVE K_eff AND strong magnitude:
     - K=-0.5: CV=33% (too weak)
     - K=-1.0: CV=16% (marginal)
     - K=-1.3: CV=9% (flat)
     - K=+0.5: CV=135% (wrong sign, amplifies)
     - K=+1.3: UNSTABLE (σ~10⁹)

  4. Positive K_eff is always bad. Physical reason:
     K_eff > 0 strengthens effective coupling to K1+K_eff but self-energy
     stays at K1 → force imbalance → amplification at low k → instability
     at large K_eff. Negative K_eff weakens coupling → stable scatterer.

  5. The CV(|K_eff|) trend for negative K:
     CV = 33% (0.5), 16% (1.0), 9% (1.3) — gradual decrease.
     No sharp threshold; flatness improves continuously with magnitude.
     Matches file 36: diagonal CV = 62% (cm1=0.05), 51% (0.19),
     25% (0.69), 9% (1.31).

  For paper §4.3: displacement coupling (force ∝ u_neighbor, not ∝ Δu) on
  ring geometry is the sufficient condition for flat integrand. Strain coupling
  on the same bonds produces exactly zero scattering (geometric null).
  This definitively identifies the coupling mechanism.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/37_displacement_coupling.py
"""

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/37_displacement_coupling.py
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


def make_displacement_force(K_eff, R_loop, L, K1=1.0, K2=0.5):
    """Force function with pure displacement coupling on Dirac disk bonds.

    On each gauged z-bond (iz_lo, iy, ix) <-> (iz_hi, iy, ix):
      site lo gets: K_eff * ux_hi  (and same for uy)
      site hi gets: K_eff * ux_lo  (symmetric)

    This is displacement coupling: force ∝ u_neighbor.
    No self-energy modification, no strain, no rotation.
    Acts on (ux, uy) identically — scalar-like but on displacement.
    """
    cz = L // 2
    iy_disk, ix_disk = precompute_disk_bonds(L, R_loop)

    iz_lo = cz - 1
    iz_hi = cz
    n_bonds = len(iy_disk)

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if K_eff == 0.0:
            return fx, fy, fz

        ux_hi = ux[iz_hi, iy_disk, ix_disk]
        uy_hi = uy[iz_hi, iy_disk, ix_disk]
        ux_lo = ux[iz_lo, iy_disk, ix_disk]
        uy_lo = uy[iz_lo, iy_disk, ix_disk]

        # Lower site: K_eff * u_upper
        fx[iz_lo, iy_disk, ix_disk] += K_eff * ux_hi
        fy[iz_lo, iy_disk, ix_disk] += K_eff * uy_hi

        # Upper site: K_eff * u_lower (symmetric)
        fx[iz_hi, iy_disk, ix_disk] += K_eff * ux_lo
        fy[iz_hi, iy_disk, ix_disk] += K_eff * uy_lo

        return fx, fy, fz

    force_fn.n_bonds = n_bonds
    force_fn.coupling_type = "displacement"
    return force_fn


def make_strain_ring_force(dK, R_loop, L, K1=1.0, K2=0.5):
    """Force function with strain coupling on Dirac disk bonds.

    On each gauged z-bond: changes K1 → K1 + dK for that bond.
    Force = dK * (u_neighbor - u_site) = strain coupling.

    Acts on (ux, uy) only — same components as displacement coupling
    for apples-to-apples comparison. uz unmodified in both cases.

    This is the ring-geometry version of the mass sphere test (file 34).
    """
    cz = L // 2
    iy_disk, ix_disk = precompute_disk_bonds(L, R_loop)

    iz_lo = cz - 1
    iz_hi = cz
    n_bonds = len(iy_disk)

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if dK == 0.0:
            return fx, fy, fz

        # Only (ux, uy) — same as displacement coupling
        for u, f in [(ux, fx), (uy, fy)]:
            u_hi = u[iz_hi, iy_disk, ix_disk]
            u_lo = u[iz_lo, iy_disk, ix_disk]
            du = u_hi - u_lo

            # Lower site: dK * (u_hi - u_lo) = dK * du
            f[iz_lo, iy_disk, ix_disk] += dK * du
            # Upper site: dK * (u_lo - u_hi) = -dK * du
            f[iz_hi, iy_disk, ix_disk] -= dK * du

        return fx, fy, fz

    force_fn.n_bonds = n_bonds
    force_fn.coupling_type = "strain"
    return force_fn


if __name__ == '__main__':
    t0 = time.time()

    print("Route 37 (F5+F7): Displacement vs strain coupling on ring")
    print(f"  R={R_LOOP}, L={L}")
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

    # Test cases: (label, force_fn_factory)
    cases = [
        ("displ K=0 (sanity)", lambda: make_displacement_force(0.0, R_LOOP, L)),
        ("displ K=-0.5", lambda: make_displacement_force(-0.5, R_LOOP, L)),
        ("displ K=-1.0", lambda: make_displacement_force(-1.0, R_LOOP, L)),
        ("displ K=-1.3", lambda: make_displacement_force(-1.3, R_LOOP, L)),
        ("displ K=+0.5", lambda: make_displacement_force(+0.5, R_LOOP, L)),
        ("displ K=+1.3", lambda: make_displacement_force(+1.3, R_LOOP, L)),
        ("strain dK=-1.3", lambda: make_strain_ring_force(-1.3, R_LOOP, L)),
        ("strain dK=+1.0", lambda: make_strain_ring_force(+1.0, R_LOOP, L)),
    ]

    all_results = []

    for label, factory in cases:
        f_test = factory()
        print(f"\n{label}  ({f_test.n_bonds} bonds, {f_test.coupling_type})")

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
            if st < 0:
                print(f"  WARNING: σ_tr={st:.2e} at k={k0}, clipped to 0")
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
            'type': f_test.coupling_type,
        })

        print(f"  σ_tr: " + "  ".join(f"{s:.4f}" for s in sigma_tr))
        print(f"  sin²(k)·σ_tr CV = {cv:.1f}%  ({dt:.0f}s)")

    # Summary
    print()
    print("=" * 70)
    print("Summary: displacement vs strain coupling on ring")
    print("=" * 70)
    print(f"  {'label':>18s}  {'type':>10s}  {'CV%':>6s}  {'verdict':>10s}  "
          f"{'σ(0.3)':>8s}  {'σ(1.5)':>8s}")
    print(f"  {'-'*65}")
    for r in all_results:
        cv = r['cv']
        verdict = "FLAT" if cv < 10 else ("marginal" if cv < 20 else "NOT FLAT")
        print(f"  {r['label']:>18s}  {r['type']:>10s}  {cv:5.1f}%  {verdict:>10s}  "
              f"{r['sigma_tr'][0]:8.4f}  {r['sigma_tr'][-1]:8.4f}")

    # Reference
    print()
    print("  Reference:")
    print("    Peierls full α=0.30 (file 28):       CV = 7.5%   FLAT")
    print("    Peierls diagonal α=0.30 (file 35):   CV = 8.8%   FLAT")
    print("    Mass sphere K1=0 R=5 (file 34):      CV = 112%   NOT FLAT")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
