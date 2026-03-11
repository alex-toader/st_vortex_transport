"""
Route 34 (F2): Mass sphere — non-gauge defect control test.

Is flat integrand gauge-specific or geometric?
Sphere of altered K1 at center. NN bonds with BOTH endpoints inside sphere
get K1_inside instead of K1. NNN (K2) unchanged. No gauge, no rotation.

Tests K1_inside = 1.0 (sanity), 2.0 (stiff), 0.5 (soft), 0.0 (void)
at R_defect=5, plus K1_inside=0.0 at R_defect=2.

Results (L=80, r_m=20, 518s):

  Sanity (K1_inside=1.0): σ_tr = 0.00 exact at all k. PASS.

  sin²(k)·σ_tr CV:
    defect           K1_in  R  bonds    CV%    verdict
    stiff              2.0  5   1302   89.4%   NOT FLAT
    soft               0.5  5   1302   92.1%   NOT FLAT
    void               0.0  5   1302  111.9%   NOT FLAT
    void small         0.0  2     60  151.3%   NOT FLAT
    vortex NN α=0.30    —   5     81    7.5%   FLAT

  σ_tr(k) spectra:
    k     stiff   soft    void    void_sm   vortex_NN(α=0.3)
   0.3     1.18   0.41    1.92    0.020       40.74
   0.5     1.80   0.85    5.27    0.107       14.16
   0.7     2.74   1.58    8.64    0.282        7.69
   0.9     4.04   1.98   12.04    0.476        5.05
   1.1     4.88   2.76   19.78    1.187        3.78
   1.3     6.78   3.62   30.98    1.457        3.14
   1.5     8.37   4.77   51.12    5.123        2.81

  CONCLUSION:

    Flat integrand is GAUGE-SPECIFIC, not geometric.

    Mass sphere (non-gauge defect) gives CV = 89-152% at ALL coupling
    strengths and sizes. Vortex ring with Peierls gauge gives CV = 7.5%.

    σ_tr spectrum is qualitatively OPPOSITE:
    - Mass sphere: σ_tr INCREASES with k (short λ resolves sphere better)
    - Vortex ring: σ_tr DECREASES with k (then rises at k>1.7)

    This rules out geometric explanations for the flat integrand.
    The 1/sin²(k) spectral shape requires Peierls gauge coupling.
    1302 modified bonds (mass sphere) vs 81 gauged bonds (ring) —
    flatness is NOT about defect size or number of perturbed bonds.

    For paper §4.3: "The flat integrand is specific to gauge coupling.
    A non-gauge defect (mass sphere, 1302 bonds) produces CV > 89%,
    demonstrating that the spectral flatness requires the Peierls
    rotation structure, not merely a compact perturbation."

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/34_mass_sphere.py
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
from parallel_fdtd import compute_references

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
N_WORKERS = 4

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

# NOTE on sin²(k) convention:
# The drag integrand uses sin²(k), NOT sin²(k/2).
# From κ formula (file 7): γ₁ = (c/(4π²)) ∫ sin²(k) σ_tr(k) dk
# where sin²(k) = ω²·v_g²/c⁴ = [2c sin(k/2)]²·[c cos(k/2)]²/c⁴
#                = 4 sin²(k/2)cos²(k/2) = sin²(k).
# This convention is consistent across all project files (7-34).

k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])


def make_mass_sphere_force(K1_inside, R_defect, L, K1=1.0, K2=0.5):
    """Force function: NN bonds with both endpoints inside sphere get K1_inside.

    Computes full scalar Laplacian, then applies correction dK1 on
    interior NN bonds. NNN (K2) unchanged.
    """
    cx = cy = cz = L // 2

    iz, iy, ix = np.mgrid[0:L, 0:L, 0:L]
    r2 = (ix - cx)**2 + (iy - cy)**2 + (iz - cz)**2
    inside = r2 <= R_defect**2

    dK1 = float(K1_inside) - K1

    # Bond masks: both endpoints inside (float for multiplication)
    mask_px = (inside[:, :, :-1] & inside[:, :, 1:]).astype(float)
    mask_py = (inside[:, :-1, :] & inside[:, 1:, :]).astype(float)
    mask_pz = (inside[:-1, :, :] & inside[1:, :, :]).astype(float)

    n_sites = int(inside.sum())
    n_bonds = int(mask_px.sum() + mask_py.sum() + mask_pz.sum())

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if dK1 == 0.0:
            return fx, fy, fz

        for u, f in [(ux, fx), (uy, fy), (uz, fz)]:
            # +x bonds
            du = u[:, :, 1:] - u[:, :, :-1]
            corr = dK1 * du * mask_px
            f[:, :, :-1] += corr
            f[:, :, 1:] -= corr

            # +y bonds
            du = u[:, 1:, :] - u[:, :-1, :]
            corr = dK1 * du * mask_py
            f[:, :-1, :] += corr
            f[:, 1:, :] -= corr

            # +z bonds
            du = u[1:, :, :] - u[:-1, :, :]
            corr = dK1 * du * mask_pz
            f[:-1, :, :] += corr
            f[1:, :, :] -= corr

        return fx, fy, fz

    force_fn.n_sites = n_sites
    force_fn.n_bonds = n_bonds
    return force_fn


if __name__ == '__main__':
    t0 = time.time()

    print("Route 34 (F2): Mass sphere — non-gauge defect")
    print(f"  L={L}, r_m={r_m}")
    print(f"  k = {list(k_vals)}")
    print()

    gamma = make_damping_3d(L, DW, DS)
    iz_s, iy_s, ix_s = make_sphere_points(L, r_m, thetas, phis)

    # References (parallel, plain lattice)
    print("Computing references...")
    t1 = time.time()
    refs = compute_references(k_vals, L, DW, DS, DT, sx, r_m,
                              thetas, phis, K1, K2, n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")

    # Test cases: (K1_inside, R_defect, label)
    cases = [
        (1.0, 5, "sanity (no defect)"),
        (2.0, 5, "stiff"),
        (0.5, 5, "soft"),
        (0.0, 5, "void"),
        (0.0, 2, "void small"),
    ]

    all_results = []
    for K1_in, R_def, label in cases:
        f_ms = make_mass_sphere_force(K1_in, R_def, L, K1, K2)
        assert f_ms.n_bonds > 0, f"No bonds modified! Check R_defect={R_def}"
        print(f"\nK1_inside={K1_in}, R_defect={R_def} ({label})")
        print(f"  {f_ms.n_sites} sites, {f_ms.n_bonds} NN bonds modified")

        t1 = time.time()
        sigma_tr = np.zeros(len(k_vals))
        for j, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_ms, ux0.copy(), vx0.copy(), gamma, DT, ns,
                            rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
            f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                   ref['ux'], ref['uy'], ref['uz'], r_m)
            _, st = integrate_sigma_3d(f2, thetas, phis)
            sigma_tr[j] = st
        dt = time.time() - t1

        integrand = np.sin(k_vals)**2 * sigma_tr
        if np.mean(integrand) > 1e-10:
            cv = np.std(integrand) / np.mean(integrand) * 100
        else:
            cv = 0.0

        all_results.append({
            'K1_in': K1_in, 'R_def': R_def, 'label': label,
            'sigma_tr': sigma_tr.copy(), 'cv': cv,
            'n_sites': f_ms.n_sites, 'n_bonds': f_ms.n_bonds,
            'dt': dt,
        })

        print(f"  σ_tr: " + "  ".join(f"{s:.4f}" for s in sigma_tr))
        print(f"  min σ_tr = {sigma_tr.min():.4f}, "
              f"max σ_tr = {sigma_tr.max():.4f}")
        print(f"  sin²(k)·σ_tr CV = {cv:.1f}%  ({dt:.0f}s)")

    # Noise floor from sanity run
    noise_floor = all_results[0]['sigma_tr'].max()
    print()
    print(f"Noise floor (sanity run): {noise_floor:.2e}")
    print("SNR check (min σ_tr / noise_floor):")
    for r in all_results[1:]:
        snr = r['sigma_tr'].min() / max(noise_floor, 1e-20)
        reliable = "OK" if snr > 10 else "LOW SNR — CV unreliable"
        print(f"  {r['label']:>15s}: SNR = {snr:.1f}  ({reliable})")

    # Summary table
    print()
    print("=" * 70)
    print("Summary: flat integrand test (sin²(k)·σ_tr)")
    print("=" * 70)
    print(f"  {'defect':>20s}  {'K1_in':>6s}  {'R':>3s}  {'bonds':>6s}  "
          f"{'CV%':>6s}  {'verdict':>10s}")
    print(f"  {'-'*60}")
    for r in all_results:
        if r['label'].startswith("sanity"):
            max_s = r['sigma_tr'].max()
            verdict = f"σ={max_s:.1e} {'PASS' if max_s < 1e-6 else 'FAIL'}"
        elif r['cv'] < 10:
            verdict = "FLAT"
        elif r['cv'] < 20:
            verdict = "marginal"
        else:
            verdict = "NOT FLAT"
        print(f"  {r['label']:>20s}  {r['K1_in']:6.1f}  {r['R_def']:3d}  "
              f"{r['n_bonds']:6d}  {r['cv']:5.1f}%  {verdict:>10s}")
    print(f"  {'— vortex ring ref (file 28, same params) —':>50s}")
    print(f"  {'vortex NN α=0.30':>20s}  {'—':>6s}  {'5':>3s}  {'81':>6s}  "
          f"{'7.5':>5s}%  {'FLAT':>10s}")
    print(f"  {'vortex NN α=0.05':>20s}  {'—':>6s}  {'5':>3s}  {'81':>6s}  "
          f"{'34.2':>5s}%  {'NOT FLAT':>10s}")
    print(f"  {'vortex NNN α=0.30':>20s}  {'—':>6s}  {'5':>3s}  {'312':>6s}  "
          f"{'24.2':>5s}%  {'NOT FLAT':>10s}")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
