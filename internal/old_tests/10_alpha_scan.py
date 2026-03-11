"""
α scan: κ(α) from α=0.1 to α=0.5.

Goal: find which α gives κ=1 (if any).
Reuses reference (plain Laplacian) across α values — 2× faster than naive.

Results (L=80, r_m=20, R_loop=5, N_pol=2):
    α   κ_data   κ_tail  κ_total       β  tail%
  0.10    0.033    0.039    0.072   -1.55   54.3%
  0.15    0.111    0.100    0.211   -1.84   47.5%
  0.20    0.246    0.217    0.464   -1.91   46.9%
  0.25    0.424    0.408    0.832   -1.86   49.0%
  0.30    0.617    0.672    1.290   -1.76   52.1%
  0.40    0.940    1.270    2.210   -1.56   57.5%
  0.50    1.062    1.556    2.618   -1.48   59.4%

  κ=1.0 between α=0.25 (0.83) and α=0.30 (1.29). Interpolation: α ≈ 0.27.
  κ(α) monotonically increasing, smooth.
  AB predicts κ ∝ sin²(πα) — measured ratio κ(0.5)/κ(0.25) = 3.15 vs AB 2.0.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/10_alpha_scan.py
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
from importlib import import_module
kappa_mod = import_module('7_kappa_extraction')

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
k_vals = np.array([0.15, 0.2, 0.3, 0.5, 0.7, 0.9])
alpha_vals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

x_start = DW + 5
iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
gamma_pml = make_damping_3d(L, DW, DS)


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


# ── Step 1: reference runs (one per k, reused across α) ─────

print("Computing references...")
refs = {}
for k0 in k_vals:
    ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
    ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
    ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                      DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    refs[k0] = (ref, ux0, vx0, ns)
print(f"  {len(k_vals)} references done.\n")


# ── Step 2: scan α values ───────────────────────────────────

print(f"{'α':>5s}  {'κ_data':>7s}  {'κ_tail':>7s}  {'κ_total':>7s}  {'β':>6s}  {'tail%':>5s}")
print("-" * 50)

results = []
for alpha in alpha_vals:
    sigma_tr_arr = np.zeros(len(k_vals))

    for i, k0 in enumerate(k_vals):
        ref, ux0, vx0, ns = refs[k0]  # ref is read-only (compute_sphere_f2 doesn't modify)
        f_def = make_vortex_force(alpha, R_LOOP, L, K1, K2)
        d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                        DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        _, st = integrate_sigma_3d(f2, thetas, phis)
        sigma_tr_arr[i] = st

    kd, kt, ktot, beta, terr = kappa_mod.compute_kappa(k_vals, sigma_tr_arr, R_LOOP)
    tail_pct = kt / ktot * 100 if ktot > 0 else 0
    results.append((alpha, kd, kt, ktot, beta, tail_pct))
    print(f"{alpha:5.2f}  {kd:7.3f}  {kt:7.3f}  {ktot:7.3f}  {beta:6.2f}  {tail_pct:5.1f}%")

print()
print("=" * 50)
print(f"R_loop={R_LOOP}, L={L}, r_m={r_m}, N_pol=2")

# Interpolate α where κ = 1
alphas = [r[0] for r in results]
kappas = [r[3] for r in results]
for i in range(len(kappas) - 1):
    if kappas[i] <= 1.0 <= kappas[i + 1]:
        frac = (1.0 - kappas[i]) / (kappas[i + 1] - kappas[i])
        alpha_interp = alphas[i] + frac * (alphas[i + 1] - alphas[i])
        print(f"κ = 1.0 at α = {alpha_interp:.3f} (interpolated between "
              f"α={alphas[i]} → κ={kappas[i]:.3f} and "
              f"α={alphas[i+1]} → κ={kappas[i+1]:.3f})")

# Implication for P10
print()
print("Implication: κ=1 requires α≈0.27, not α=1/2 (topological prediction).")
print("Either: (a) α=1/2 is wrong (P2b gap), or")
print("        (b) lattice model (NNN ungauged, Option B) has ~2.6x systematic.")
