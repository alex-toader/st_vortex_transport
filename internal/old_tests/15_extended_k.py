"""
Extended k-grid: measure σ_tr at k = 1.1, 1.3, 1.5 for α = 0.3.

Combine with existing data [0.15, 0.2, 0.3, 0.5, 0.7, 0.9] to get
8-point grid. Recompute κ with reduced tail (k=1.5 to π instead of
k=0.9 to π).

Existing κ_orig = 1.29 (tail 52%). Expected: tail drops to ~15-20%.

Also check: does σ_tr continue power law at k > 0.9, or does BZ
artifact appear? (From GL6: turnover at k ≈ 2, artifact at k > 2.5.)

Results:
  New measurements: k=1.1: σ_tr=3.78, k=1.3: σ_tr=3.14, k=1.5: σ_tr=2.81
  Power law continues smoothly (β=-1.14, all 4 pts within 3% of fit).
  No BZ artifact at k ≤ 1.5.

  Extended grid (9 pts): κ = 1.50 (tail 29%)
  Original grid (6 pts): κ = 1.30 (tail 53%)
  → κ increases +16% with extended grid, tail drops from 53% to 29%.

  Integrand sin²(k)·σ_tr(k) slowly decreasing: 3.55 (k=0.3) → 2.80 (k=1.5),
  ~20% variation over the range. Approximate scaling σ_tr ~ 1/sin²(k) but not exact.

  Non-monotonicity at low k: σ_tr(0.15) = 69.3, σ_tr(0.20) = 89.4 (peak), σ_tr(0.30) = 40.7.
  Near-field artifact (r_m/λ < 1 at k < 0.3). Method 2 ("clean") excludes k < 0.3 for this reason.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/15_extended_k.py
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


# ── Measure only NEW k values ────────────────────────────

k_new = np.array([1.1, 1.3, 1.5])
f_def = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)

print(f"α={ALPHA}, R={R_LOOP}, L={L}")
print(f"New k values: {k_new}")
print()

sigma_tr_new = np.zeros(len(k_new))
for i, k0 in enumerate(k_new):
    ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
    ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)

    ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                      DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                    DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                           ref['ux'], ref['uy'], ref['uz'], r_m)
    _, st = integrate_sigma_3d(f2, thetas, phis)
    sigma_tr_new[i] = st
    print(f"  k={k0:.1f}: σ_tr={st:.2f}, sin²(k)={np.sin(k0)**2:.3f}, "
          f"integrand={st * np.sin(k0)**2:.3f}")


# ── Combine with existing data ───────────────────────────

# Existing measurements from 7_kappa_extraction.py (α=0.3, R=5, L=80)
# Verified by fresh re-run of 7_kappa_extraction.py
k_old = np.array([0.15, 0.2, 0.3, 0.5, 0.7, 0.9])
str_old = np.array([69.3, 89.4, 40.7, 14.2, 7.7, 5.0])

k_all = np.concatenate([k_old, k_new])
str_all = np.concatenate([str_old, sigma_tr_new])
idx = np.argsort(k_all)
k_all = k_all[idx]
str_all = str_all[idx]

print()
print("Combined σ_tr(k) spectrum:")
print(f"{'k':>5s}  {'σ_tr':>7s}  {'sin²(k)':>7s}  {'integrand':>9s}")
print("-" * 35)
for k0, st in zip(k_all, str_all):
    s2 = np.sin(k0)**2
    print(f"{k0:5.2f}  {st:7.2f}  {s2:7.3f}  {s2*st:9.3f}")


# ── κ with extended grid ─────────────────────────────────

prefactor = N_POL * R_LOOP / (4 * np.pi**2)

# Method 1: trapezoidal on extended grid, tail from k=1.5 to π
integrand_data = np.sin(k_all)**2 * str_all
kappa_data = prefactor * np.trapz(integrand_data, k_all)

# Tail: power law fit on k > 0.8 (now 4 points: 0.9, 1.1, 1.3, 1.5)
mask = k_all > 0.8
beta, logA = np.polyfit(np.log(k_all[mask]), np.log(str_all[mask]), 1)
assert beta < -1.0, f"tail diverges: beta={beta:.2f}"
A = np.exp(logA)
k_tail = np.linspace(k_all[-1], np.pi, 200)[1:]
integrand_tail = np.sin(k_tail)**2 * A * k_tail**beta
kappa_tail = prefactor * np.trapz(integrand_tail, k_tail)
kappa_total = kappa_data + kappa_tail
tail_pct = kappa_tail / kappa_total * 100

print()
print("=" * 50)
print(f"Extended grid (9 pts, k=0.15 to 1.5):")
print(f"  κ_data = {kappa_data:.3f}")
print(f"  κ_tail = {kappa_tail:.3f} (k=1.5→π, ~k^{beta:.2f})")
print(f"  κ_total = {kappa_total:.3f}  (tail {tail_pct:.0f}%)")
print()
print(f"  vs κ_orig = 1.290  (tail 52%)")
print(f"  diff = {(kappa_total - 1.290) / 1.290 * 100:+.1f}%")
print()

# Method 2: drop k < 0.3 (near-field unreliable), use k = 0.3 to 1.5
mask2 = k_all >= 0.3
k_clean = k_all[mask2]
str_clean = str_all[mask2]
integrand_clean = np.sin(k_clean)**2 * str_clean
kappa_clean = prefactor * np.trapz(integrand_clean, k_clean)

# Tail from 1.5 to π (same fit)
kappa_clean_total = kappa_clean + kappa_tail
tail_pct2 = kappa_tail / kappa_clean_total * 100

print(f"Clean grid (7 pts, k=0.3 to 1.5, drop near-field):")
print(f"  κ_data = {kappa_clean:.3f}")
print(f"  κ_tail = {kappa_tail:.3f}")
print(f"  κ_total = {kappa_clean_total:.3f}  (tail {tail_pct2:.0f}%)")
