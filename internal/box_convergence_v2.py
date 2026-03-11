"""Box size convergence v2: fixed source-ring distance.

Source always at L//2 - 20 (distance 20 from ring at center).
r_m = 20 for all runs.

Run: cd src && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../internal/box_convergence_v2.py
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from gauge_3d import make_vortex_force
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           compute_sphere_f2, integrate_sigma_3d,
                           estimate_n_steps_3d)

K1, K2 = 1.0, 0.5
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
ALPHA = 0.30
R = 5

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

k_test = np.array([0.3, 0.9, 1.5])

print("=" * 65)
print("Box Convergence v2: fixed source-ring distance = 20")
print(f"R={R}, α={ALPHA}, DW={DW}, r_m={r_m}")
print("=" * 65)

def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)

results = {}
for L in [80, 100, 120]:
    center = L // 2
    x_start = center - 20  # fixed distance 20 from ring

    # Check geometry is safe
    pml_inner = DW
    pml_outer = L - DW
    sphere_min_x = center - r_m
    sphere_max_x = center + r_m
    print(f"\nL={L}: center={center}, source at x={x_start}")
    print(f"  PML zone: [0,{pml_inner}] and [{pml_outer},{L}]")
    print(f"  Sphere x: [{sphere_min_x}, {sphere_max_x}]")
    print(f"  Source in PML-free? {pml_inner < x_start < pml_outer}")
    print(f"  Sphere in PML-free? {pml_inner < sphere_min_x} and {sphere_max_x < pml_outer}")

    if x_start <= pml_inner:
        print(f"  SKIP: source inside PML!")
        continue

    gamma_pml = make_damping_3d(L, DW, DS)
    iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)

    t0 = time.time()
    sigma_tr = np.zeros(len(k_test))
    f_def = make_vortex_force(ALPHA, R, L, K1, K2)

    for i, k0 in enumerate(k_test):
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
        print(f"  k={k0}: σ_tr = {st:.3f}")

    results[L] = sigma_tr
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.0f}s")

print("\n" + "=" * 65)
print("Convergence (fixed geometry, varying box)")
print("=" * 65)

L_list = sorted(results.keys())
header = f"{'k':>4s}" + "".join(f"  {'L='+str(L):>10s}" for L in L_list)
print(f"\n{header}")
for i, kv in enumerate(k_test):
    vals = [results[L][i] for L in L_list]
    line = f"{kv:4.1f}" + "".join(f"  {v:10.3f}" for v in vals)
    print(line)

if len(L_list) >= 2:
    print(f"\n{'k':>4s}  {'Δ(first-last)':>14s}")
    for i, kv in enumerate(k_test):
        v_first = results[L_list[0]][i]
        v_last = results[L_list[-1]][i]
        d = (v_first - v_last) / v_last * 100
        print(f"{kv:4.1f}  {d:+13.1f}%")
