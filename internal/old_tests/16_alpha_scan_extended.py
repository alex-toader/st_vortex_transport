"""
Extended α scan: κ(α) with 9-point k-grid (k=0.15 to 1.5).

Same as 10_alpha_scan.py but with 3 extra k values (1.1, 1.3, 1.5).
References shared across α. Tail from k=1.5 to π (power law fit on k>0.8).

Results (L=80, r_m=20, R_loop=5, N_pol=2, 9 k-pts):

  Full grid (k=0.15 to 1.5):
    α   κ_data   κ_tail  κ_total       β  tail%
  0.10    0.063    0.044    0.107   -0.32  41.0%  *unreliable
  0.15    0.191    0.099    0.290   -0.66  34.0%  *unreliable
  0.20    0.412    0.180    0.592   -0.92  30.4%  *unreliable
  0.25    0.715    0.293    1.008   -1.08  29.0%
  0.30    1.066    0.433    1.499   -1.16  28.9%
  0.40    1.710    0.734    2.444   -1.18  30.0%
  0.50    1.977    0.877    2.855   -1.17  30.7%

  Clean grid (k≥0.3, drop near-field artifact):
    α   κ_data   κ_tail  κ_total  tail%
  0.25    0.625    0.293    0.918  31.9%
  0.30    0.944    0.433    1.377  31.5%
  0.40    1.545    0.734    2.279  32.2%
  0.50    1.798    0.877    2.675  32.8%

  κ=1.0 at α ≈ 0.259 (clean grid).
  α<0.2: β>-1 → tail not convergent, κ unreliable.
  α=0.20: β=-0.92, borderline (just above -1.0 threshold).

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/16_alpha_scan_extended.py
"""

import sys
import os
import time
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
N_POL = 2

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
x_start = DW + 5
iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
gamma_pml = make_damping_3d(L, DW, DS)

k_vals = np.array([0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
alpha_vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


# ── References (shared across α) ───────────────────────

t0 = time.time()
print(f"Computing {len(k_vals)} references (L={L})...")
refs = {}
for k0 in k_vals:
    ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
    ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
    ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                      DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    refs[k0] = (ref, ux0, vx0, ns)
t_ref = time.time() - t0
print(f"  Done ({t_ref:.0f}s)\n")


# ── α scan ──────────────────────────────────────────────

prefactor = N_POL * R_LOOP / (4 * np.pi**2)

print(f"{'α':>5s}  {'κ_data':>7s}  {'κ_tail':>7s}  {'κ_total':>7s}"
      f"  {'β':>6s}  {'tail%':>5s}  {'time':>5s}")
print("-" * 55)

results = []
for alpha in alpha_vals:
    t1 = time.time()
    sigma_tr = np.zeros(len(k_vals))
    f_def = make_vortex_force(alpha, R_LOOP, L, K1, K2)

    for i, k0 in enumerate(k_vals):
        ref, ux0, vx0, ns = refs[k0]
        d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                        DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        _, st = integrate_sigma_3d(f2, thetas, phis)
        sigma_tr[i] = st

    # κ: trapezoidal on data + power law tail from k=1.5 to π
    integrand_data = np.sin(k_vals)**2 * sigma_tr
    kd = prefactor * np.trapz(integrand_data, k_vals)

    # Tail fit on k > 0.8 (4 points: 0.9, 1.1, 1.3, 1.5)
    mask = k_vals > 0.8
    beta, logA = np.polyfit(np.log(k_vals[mask]), np.log(sigma_tr[mask]), 1)
    A = np.exp(logA)
    k_tail = np.linspace(1.5, np.pi, 200)[1:]
    integrand_tail = np.sin(k_tail)**2 * A * k_tail**beta
    kt = prefactor * np.trapz(integrand_tail, k_tail)

    ktot = kd + kt
    tail_pct = kt / ktot * 100 if ktot > 0 else 0

    # Clean grid: k >= 0.3 only (drop near-field artifact at k < 0.3)
    mask_clean = k_vals >= 0.3
    integrand_clean = np.sin(k_vals[mask_clean])**2 * sigma_tr[mask_clean]
    kd_clean = prefactor * np.trapz(integrand_clean, k_vals[mask_clean])
    ktot_clean = kd_clean + kt
    tail_pct_clean = kt / ktot_clean * 100 if ktot_clean > 0 else 0

    reliable = beta < -1.0
    dt_alpha = time.time() - t1

    results.append((alpha, kd, kt, ktot, beta, tail_pct, sigma_tr.copy(),
                     kd_clean, ktot_clean, tail_pct_clean, reliable))
    flag = " " if reliable else "*"
    print(f"{alpha:5.2f}  {kd:7.3f}  {kt:7.3f}  {ktot:7.3f}"
          f"  {beta:6.2f}  {tail_pct:4.1f}%  {dt_alpha:5.1f}s{flag}")

print("  (* = β > -1, tail not convergent, κ unreliable)")

t_total = time.time() - t0
print()
print("=" * 55)
print(f"Total time: {t_total:.0f}s ({t_total/60:.1f} min)")
print(f"R_loop={R_LOOP}, L={L}, r_m={r_m}, N_pol={N_POL}")
print(f"k grid: {len(k_vals)} pts from {k_vals[0]} to {k_vals[-1]}")

# Clean grid results (k >= 0.3, drop near-field)
print()
print("Clean grid (k >= 0.3, drop near-field artifact):")
print(f"{'α':>5s}  {'κ_data':>7s}  {'κ_tail':>7s}  {'κ_total':>7s}  {'tail%':>5s}")
print("-" * 40)
for r in results:
    alpha, _, _, _, beta, _, _, kd_c, ktot_c, tp_c, reliable = r
    flag = " " if reliable else "*"
    print(f"{alpha:5.2f}  {kd_c:7.3f}  {r[2]:7.3f}  {ktot_c:7.3f}  {tp_c:4.1f}%{flag}")

# Interpolate α where κ = 1 (clean grid)
print()
alphas = [r[0] for r in results]
kappas_clean = [r[8] for r in results]  # ktot_clean
for i in range(len(kappas_clean) - 1):
    if kappas_clean[i] <= 1.0 <= kappas_clean[i + 1]:
        frac = (1.0 - kappas_clean[i]) / (kappas_clean[i + 1] - kappas_clean[i])
        alpha_interp = alphas[i] + frac * (alphas[i + 1] - alphas[i])
        print(f"κ = 1.0 at α ≈ {alpha_interp:.3f} (clean grid, between "
              f"α={alphas[i]}→{kappas_clean[i]:.3f} and "
              f"α={alphas[i+1]}→{kappas_clean[i+1]:.3f})")

# Reliable results only (β < -1)
print()
print("Reliable results (β < -1, α ≥ 0.2):")
for r in results:
    alpha, kd, kt, ktot, beta, tail_pct, _, kd_c, ktot_c, tp_c, reliable = r
    if reliable:
        print(f"  α={alpha:.2f}: κ={ktot_c:.3f} (clean), β={beta:.2f}, tail={tp_c:.0f}%")
