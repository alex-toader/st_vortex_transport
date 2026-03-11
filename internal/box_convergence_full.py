"""Full 7-point k scan at L=100 (converged box size).

Generates converged σ_tr data to replace L=80 hardcoded values.
Source at x = L//2 - 20 (fixed distance 20 from ring).

Run: cd src && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../internal/box_convergence_full.py
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
L = 100
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
ALPHA = 0.30
R = 5

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# Hardcoded L=80 data for comparison
sigma_L80 = np.array([40.740, 14.157, 7.072, 4.852, 3.493, 2.581, 1.754])

center = L // 2
x_start = center - 20

print("=" * 65)
print(f"Full k-scan at L={L} (converged box)")
print(f"R={R}, α={ALPHA}, DW={DW}, r_m={r_m}")
print(f"center={center}, source at x={x_start}")
print("=" * 65)

gamma_pml = make_damping_3d(L, DW, DS)
iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)

def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)

f_def = make_vortex_force(ALPHA, R, L, K1, K2)

t0 = time.time()
sigma_tr = np.zeros(len(k_vals))
for i, k0 in enumerate(k_vals):
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
    print(f"  k={k0:.1f}: σ_tr = {st:.3f} (L=80: {sigma_L80[i]:.3f}, Δ={(st-sigma_L80[i])/sigma_L80[i]*100:+.1f}%)")

elapsed = time.time() - t0
print(f"\nTotal time: {elapsed:.0f}s")

# Integrand flatness
integrand_new = np.sin(k_vals)**2 * sigma_tr
integrand_old = np.sin(k_vals)**2 * sigma_L80

cv_new = np.std(integrand_new) / np.abs(np.mean(integrand_new)) * 100
cv_old = np.std(integrand_old) / np.abs(np.mean(integrand_old)) * 100

print(f"\nIntegrand sin²(k)·σ_tr:")
print(f"  L={L}: CV = {cv_new:.1f}%")
print(f"  L=80:  CV = {cv_old:.1f}%")

print(f"\n{'k':>4s}  {'σ_tr(L=100)':>12s}  {'σ_tr(L=80)':>11s}  {'Δ':>7s}  {'integ(100)':>11s}  {'integ(80)':>10s}")
for i, kv in enumerate(k_vals):
    d = (sigma_tr[i] - sigma_L80[i]) / sigma_L80[i] * 100
    print(f"{kv:4.1f}  {sigma_tr[i]:12.3f}  {sigma_L80[i]:11.3f}  {d:+6.1f}%  {integrand_new[i]:11.3f}  {integrand_old[i]:10.3f}")

# Power law exponent
from numpy.polynomial.polynomial import polyfit
lx = np.log(k_vals)
ly_new = np.log(sigma_tr)
ly_old = np.log(sigma_L80)
p_new = np.polyfit(lx, ly_new, 1)[0]
p_old = np.polyfit(lx, ly_old, 1)[0]
print(f"\nσ_tr exponent: L={L}: {p_new:.3f}, L=80: {p_old:.3f}")

# Output for hardcoding
print(f"\n# For sigma_ring.py (L={L}, R={R}, α={ALPHA}):")
print(f"sigma_ring_L100 = np.array([{', '.join(f'{s:.3f}' for s in sigma_tr)}])")
