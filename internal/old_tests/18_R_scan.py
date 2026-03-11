"""
Route 17: R-scan — κ(R) at fixed α=0.3.

Test γ ∝ c/R universality: if κ = N_pol·R/(4π²) ∫ sin²(k)·σ_tr(k,R) dk
is R-independent, then γ = κ·c/R scales as 1/R.

R = 3, 5, 7, 9 at α=0.3. L=80, r_m=20.
R=10 excluded (needs L≥120 per reviewer).

Results (clean grid k=0.3-1.5, tail from power law fit):

    R  κ_total     β  tail%  γ=κc/R
    3    0.379  -1.03   34%  0.2189
    5    1.377  -1.16   31%  0.4771
    7    3.207  -1.09   32%  0.7934   (r_m/R=2.9, near-field warning)
    9    6.696  -1.08   33%  1.2887   (r_m/R=2.2, near-field warning)

  κ is NOT R-independent. γ ∝ R^1.59 (not R^-1.0).
  κ ∝ R^2.6. Even far-field only (R=3,5): κ ∝ R^2.5.

  Raw σ_tr(k, R) table:

    k       R=3      R=5      R=7      R=9
   0.30    13.16    40.74    72.70   117.84
   0.50     6.43    14.16    22.23    34.54
   0.70     3.54     7.69    12.22    19.43
   0.90     2.28     5.05     8.16    13.34
   1.10     1.72     3.78     6.26    10.29
   1.30     1.43     3.14     5.25     8.56
   1.50     1.36     2.81     4.71     7.74

  sin²(k)·σ_tr (integrand):

    k       R=3      R=5      R=7      R=9
   0.30     1.15     3.56     6.35    10.29
   0.50     1.48     3.25     5.11     7.94
   0.70     1.47     3.19     5.07     8.06
   0.90     1.40     3.10     5.01     8.19
   1.10     1.36     3.00     4.97     8.17
   1.30     1.33     2.91     4.87     7.95
   1.50     1.35     2.79     4.68     7.71

  σ_tr scaling with R (R^p fit, all 4 R values):

    k     R^p
   0.30  1.99   (geometric, kR=0.9-2.7)
   0.50  1.51
   0.70  1.53
   0.90  1.58
   1.10  1.61
   1.30  1.61
   1.50  1.56

  σ_tr scaling: k=0.3 → R^1.99 (geometric), k=0.5-1.5 → R^1.51-1.61 (sub-geometric).
  The scaling argument (σ_tr~R², k~1/R → κ~O(1)) fails because:
    1. σ_tr ~ R^1.6 on average, not R²
    2. Flat integrand means all k contribute, not just k~1/R

  κ=1 at α≈0.26 is specific to R=5, not universal.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/18_R_scan.py
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
ALPHA = 0.3
N_POL = 2

c = np.sqrt(K1 + 4 * K2)

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
x_start = DW + 5
iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
gamma_pml = make_damping_3d(L, DW, DS)

# Clean grid only (k ≥ 0.3)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
R_vals = [3, 5, 7, 9]


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


# ── References (shared across R) ─────────────────────
t0 = time.time()
print(f"Computing {len(k_vals)} references (L={L}, clean grid k>=0.3)...")
refs = {}
for k0 in k_vals:
    ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
    ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
    ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                      DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    refs[k0] = (ref, ux0, vx0, ns)
t_ref = time.time() - t0
print(f"  Done ({t_ref:.0f}s)\n")


# ── R scan ────────────────────────────────────────────

print("=" * 65)
print(f"R-scan: alpha={ALPHA}, L={L}, r_m={r_m}, N_pol={N_POL}")
print("=" * 65)

summary = []
for R in R_vals:
    if r_m < 3 * R:
        print(f"\n  WARNING: r_m/R = {r_m/R:.1f} < 3 — near-field at ring edge")
    t1 = time.time()
    sigma_tr = np.zeros(len(k_vals))
    f_def = make_vortex_force(ALPHA, R, L, K1, K2)

    for i, k0 in enumerate(k_vals):
        ref, ux0, vx0, ns = refs[k0]
        d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                        DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        _, st = integrate_sigma_3d(f2, thetas, phis)
        sigma_tr[i] = st

    # κ from clean grid (trapezoidal)
    prefactor = N_POL * R / (4 * np.pi**2)
    integrand = np.sin(k_vals)**2 * sigma_tr
    kd = prefactor * np.trapz(integrand, k_vals)

    # Tail: power law fit on k > 0.8
    mask = k_vals > 0.8
    beta, logA = np.polyfit(np.log(k_vals[mask]), np.log(sigma_tr[mask]), 1)
    A = np.exp(logA)
    k_tail = np.linspace(1.5, np.pi, 200)[1:]
    integrand_tail = np.sin(k_tail)**2 * A * k_tail**beta
    kt = prefactor * np.trapz(integrand_tail, k_tail)

    ktot = kd + kt
    tail_pct = kt / ktot * 100 if ktot > 0 else 0
    reliable = beta < -1.0

    dt = time.time() - t1

    print(f"\nR = {R}  ({dt:.0f}s)")
    print(f"  {'k':>5s}  {'σ_tr':>8s}  {'sin²·σ':>7s}  {'kR':>5s}")
    print(f"  {'-'*30}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {sigma_tr[i]:8.2f}  "
              f"{integrand[i]:7.3f}  {k_vals[i]*R:5.2f}")

    flag = " " if reliable else "*"
    print(f"  κ_data={kd:.3f}, κ_tail={kt:.3f}, κ_total={ktot:.3f} "
          f"(tail {tail_pct:.0f}%, β={beta:.2f}){flag}")

    # γ = κ·c/R
    gamma = ktot * c / R
    summary.append((R, kd, kt, ktot, beta, tail_pct, reliable, gamma,
                     sigma_tr.copy()))


# ── Summary ───────────────────────────────────────────
print()
print("=" * 65)
print("Summary: κ(R) and γ(R)")
print("=" * 65)
print(f"{'R':>3s}  {'κ_data':>7s}  {'κ_total':>7s}  {'β':>6s}  "
      f"{'tail%':>5s}  {'γ=κc/R':>7s}")
print("-" * 45)
for R, kd, kt, ktot, beta, tp, rel, gamma, _ in summary:
    flag = " " if rel else "*"
    print(f"{R:3d}  {kd:7.3f}  {ktot:7.3f}  {beta:6.2f}  "
          f"{tp:4.1f}%  {gamma:7.4f}{flag}")

# Separate reliable from unreliable
R_arr = np.array([s[0] for s in summary])
kappas = np.array([s[3] for s in summary])
gammas = np.array([s[7] for s in summary])
rel_mask = np.array([s[6] for s in summary])

# Statistics on reliable only
if np.any(rel_mask):
    k_rel = kappas[rel_mask]
    g_rel = gammas[rel_mask]
    R_rel = R_arr[rel_mask]
    print()
    print(f"Reliable only (β < -1): R = {R_rel.tolist()}")
    print(f"  κ: mean={np.mean(k_rel):.3f}, "
          f"CV={np.std(k_rel)/np.mean(k_rel)*100:.1f}%")
    print(f"  γ: mean={np.mean(g_rel):.4f}, "
          f"CV={np.std(g_rel)/np.mean(g_rel)*100:.1f}%")

    if len(R_rel) >= 2:
        log_fit = np.polyfit(np.log(R_rel), np.log(g_rel), 1)
        print(f"  γ ∝ R^{log_fit[0]:.2f} (expect -1.0 for γ∝c/R)")

# Also fit without largest R (possible near-field contamination)
nf_mask = rel_mask & (R_arr * 3 <= r_m)  # r_m/R >= 3
if np.any(nf_mask) and np.sum(nf_mask) < np.sum(rel_mask):
    R_nf = R_arr[nf_mask]
    g_nf = gammas[nf_mask]
    k_nf = kappas[nf_mask]
    print()
    print(f"Reliable + far-field (r_m/R >= 3): R = {R_nf.tolist()}")
    print(f"  κ: mean={np.mean(k_nf):.3f}, "
          f"CV={np.std(k_nf)/np.mean(k_nf)*100:.1f}%")
    if len(R_nf) >= 2:
        log_fit2 = np.polyfit(np.log(R_nf), np.log(g_nf), 1)
        print(f"  γ ∝ R^{log_fit2[0]:.2f}")

# All points (for reference)
print()
print(f"All R: κ mean={np.mean(kappas):.3f}, "
      f"CV={np.std(kappas)/np.mean(kappas)*100:.1f}%")

# σ_tr scaling with R at fixed k
print()
print("σ_tr(R) at each k:")
print(f"{'k':>5s}", end="")
for R, *_ in summary:
    print(f"  {'R='+str(R):>8s}", end="")
print("   R^p")
for i, k0 in enumerate(k_vals):
    print(f"{k0:5.2f}", end="")
    str_at_k = np.array([s[8][i] for s in summary])
    for st in str_at_k:
        print(f"  {st:8.2f}", end="")
    # Fit σ_tr ∝ R^p at this k (reliable only)
    if np.sum(rel_mask) >= 2:
        p, _ = np.polyfit(np.log(R_arr[rel_mask]),
                          np.log(str_at_k[rel_mask]), 1)
        print(f"   {p:.2f}", end="")
    print()

# σ_tr/R² scaling (geometric cross-section)
print()
print("σ_tr/R² at each k:")
print(f"{'k':>5s}", end="")
for R, *_ in summary:
    print(f"  {'R='+str(R):>8s}", end="")
print()
for i, k0 in enumerate(k_vals):
    print(f"{k0:5.2f}", end="")
    for s in summary:
        print(f"  {s[8][i]/s[0]**2:8.3f}", end="")
    print()

t_total = time.time() - t0
print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
