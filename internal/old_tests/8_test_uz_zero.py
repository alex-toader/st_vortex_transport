"""
Test: uz decoupling under Option B.

Option B: gauge R(2πα) acts on (ux, uy) only → uz decoupled.

Test 1: pure uz wave → σ_tr = 0 (regression test for component independence).
Test 2: mixed ux+uz wave → uz_sc = 0 AND ux_sc identical to ux-only case.
  Test 2 is the physics test: uz presence does not perturb ux scattering.

Results (L=80, α=0.3, R=5):
  Test 1 (uz-only):
    k=0.3: σ_tr=0.00e+00, max|sc_ux|=0.00e+00, max|sc_uy|=0.00e+00, max|sc_uz|=0.00e+00 [PASS]
    k=0.5: σ_tr=0.00e+00, max|sc_ux|=0.00e+00, max|sc_uy|=0.00e+00, max|sc_uz|=0.00e+00 [PASS]
    k=0.7: σ_tr=0.00e+00, max|sc_ux|=0.00e+00, max|sc_uy|=0.00e+00, max|sc_uz|=0.00e+00 [PASS]
  Test 2 (mixed ux+uz, k=0.5):
    ux scatter diff (ux-only vs ux+uz): 0.00e+00 [PASS]
    uz scatter in mixed:                0.00e+00 [PASS]

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/8_test_uz_zero.py
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
x_start = DW + 5
R_LOOP = 5
ALPHA = 0.3


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
gamma = make_damping_3d(L, DW, DS)

for k0 in [0.3, 0.5, 0.7]:
    # uz-polarized wave packet (reuse make_wave_packet_3d for envelope, assign to uz)
    ux_env, vx_env = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
    zeros = np.zeros_like(ux_env)
    ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)

    ref = run_fdtd_3d(force_plain, zeros.copy(), zeros.copy(), gamma, DT, ns,
                      rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns,
                      uz0=ux_env.copy(), vz0=vx_env.copy())

    f_def = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)
    d = run_fdtd_3d(f_def, zeros.copy(), zeros.copy(), gamma, DT, ns,
                    rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns,
                    uz0=ux_env.copy(), vz0=vx_env.copy())

    f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                           ref['ux'], ref['uy'], ref['uz'], r_m)
    sigma, sigma_tr = integrate_sigma_3d(f2, thetas, phis)

    # Also check scattered field directly
    sc_ux = np.max(np.abs(d['ux'] - ref['ux']))
    sc_uy = np.max(np.abs(d['uy'] - ref['uy']))
    sc_uz = np.max(np.abs(d['uz'] - ref['uz']))

    status = "PASS" if sigma_tr < 1e-10 else "FAIL"
    print(f"  uz-only k={k0}: σ_tr={sigma_tr:.2e}, "
          f"max|sc_ux|={sc_ux:.2e}, max|sc_uy|={sc_uy:.2e}, max|sc_uz|={sc_uz:.2e} "
          f"[{status}]")

# ── Test 2: mixed ux+uz wave — uz should not affect ux scattering ──

print("\nTest 2: mixed ux+uz — decoupling check")

k0 = 0.5
ux_env, vx_env = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
zeros = np.zeros_like(ux_env)
ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
f_def = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)

# Run A: ux-only (reference for comparison)
ref_a = run_fdtd_3d(force_plain, ux_env.copy(), vx_env.copy(), gamma, DT, ns,
                    rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
def_a = run_fdtd_3d(f_def, ux_env.copy(), vx_env.copy(), gamma, DT, ns,
                    rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)

# Run B: ux+uz mixed (same ux, add uz component)
ref_b = run_fdtd_3d(force_plain, ux_env.copy(), vx_env.copy(), gamma, DT, ns,
                    rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns,
                    uz0=ux_env.copy(), vz0=vx_env.copy())
def_b = run_fdtd_3d(f_def, ux_env.copy(), vx_env.copy(), gamma, DT, ns,
                    rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns,
                    uz0=ux_env.copy(), vz0=vx_env.copy())

# Check 1: ux scattered field identical in A and B
sc_ux_a = def_a['ux'] - ref_a['ux']
sc_ux_b = def_b['ux'] - ref_b['ux']
diff_ux = np.max(np.abs(sc_ux_a - sc_ux_b))

# Check 2: uz scattered field zero in B
sc_uz_b = def_b['uz'] - ref_b['uz']
max_uz = np.max(np.abs(sc_uz_b))

status1 = "PASS" if diff_ux < 1e-10 else "FAIL"
status2 = "PASS" if max_uz < 1e-10 else "FAIL"
print(f"  ux scatter diff (A vs B): {diff_ux:.2e} [{status1}]")
print(f"  uz scatter in mixed:      {max_uz:.2e} [{status2}]")
