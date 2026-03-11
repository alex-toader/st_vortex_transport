"""
Tail reduction: Gauss-Legendre quadrature eliminates tail extrapolation.

Problem: original grid k = [0.15, 0.2, 0.3, 0.5, 0.7, 0.9] gives
κ(α=0.3) = 1.29 with tail = 52%. Half the result is extrapolation.

Solution: 6-point Gauss-Legendre on [0, π]. Nodes placed where sin²(k)
has most weight. No tail — integral is direct.

Results:
  GL6 nodes: k = [0.107, 0.532, 1.196, 1.946, 2.609, 3.036]
  σ_tr:          [82.0,  13.3,   3.9,   3.4,   7.1,  32.0]

  κ_GL6 = 1.828 vs κ_orig = 1.290 → +42%

  CRITICAL: σ_tr INCREASES at k > 2 (BZ boundary artifact).
  At k=3.036: λ ≈ 2.07 sites — sub-lattice, physics invalid.
  GL6 overestimates because it samples lattice artifacts.
  Approach ABANDONED in favor of extended grid (file 15).

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/14_tail_reduction.py
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
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 5
ALPHA = 0.3
N_POL = 2

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
x_start = DW + 5
iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
gamma_pml = make_damping_3d(L, DW, DS)


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


# ── GL6 nodes on [0, π] ──────────────────────────────────

x_gl, w_gl = np.polynomial.legendre.leggauss(6)
k_gl = 0.5 * np.pi * (x_gl + 1)

print(f"GL6 nodes: k = [{', '.join(f'{v:.3f}' for v in k_gl)}]")
print(f"sin²(k):       [{', '.join(f'{np.sin(v)**2:.3f}' for v in k_gl)}]")
print()


# ── Measure σ_tr at GL6 nodes ────────────────────────────

f_def = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)
sigma_tr = np.zeros(len(k_gl))

for i, k0 in enumerate(k_gl):
    ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
    ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)

    ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                      DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                    DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                           ref['ux'], ref['uy'], ref['uz'], r_m)
    _, st = integrate_sigma_3d(f2, thetas, phis)
    sigma_tr[i] = st
    print(f"  k={k0:.3f}: σ_tr={st:.2f}, sin²(k)={np.sin(k0)**2:.3f}, "
          f"integrand={st * np.sin(k0)**2:.3f}")


# ── Compute κ ─────────────────────────────────────────────

prefactor = N_POL * R_LOOP / (4 * np.pi**2)
integrand = np.sin(k_gl)**2 * sigma_tr
kappa_gl6 = prefactor * (np.pi / 2) * np.sum(w_gl * integrand)

print()
print("=" * 50)
print(f"κ_GL6  = {kappa_gl6:.3f}  (6-point Gauss-Legendre, no tail)")
print(f"κ_orig = 1.290  (6-point uniform + tail 52%)")
print(f"diff   = {(kappa_gl6 - 1.290) / 1.290 * 100:+.1f}%")
