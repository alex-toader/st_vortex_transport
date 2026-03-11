"""
Route 18: Flat integrand investigation — sin²(k)·σ_tr ≈ const?

Decomposition: sin²(k) = (ω·vg/c²)², so sin²(k)·σ_tr = Q·vg/c⁴
where Q = σ_tr·vg·ω². Note Q·vg = c⁴ × integrand (tautology).

Results (L=80, R=5, clean grid k=0.3-1.5, 7 pts):

    α    CV(Q)  CV(integrand)  κ_data  κ_analytic  diff
  0.25    6.9%     11.9%       0.625     0.636     +1.8%
  0.30    4.2%      7.5%       0.944     0.954     +1.1%
  0.40    9.3%      3.9%       1.545     1.553     +0.5%
  0.50   12.1%      5.6%       1.798     1.806     +0.4%

Key findings:
  1. Integrand sin²(k)·σ_tr is flat to 4-12% CV for all reliable α.
     Flattest at α=0.40 (3.9% CV).
  2. Q = σ_tr·vg·ω² is NOT universally the flattest quantity.
     Q flatter than integrand at low α (0.25, 0.30).
     Integrand flatter than Q at high α (0.40, 0.50).
  3. κ_analytic = prefactor × Q_mean/c⁴ × ∫vg dk works to <2% for all α.
     (Assumes Q decorrelated from vg; approximation error ~ CV(Q).)
  4. Q/α² ≈ 200 at small α, drops to 139 at α=0.50 (saturation).
     Q/sin²(πα) monotonically increases → NOT AB scaling.

Clean k-grid only (k ≥ 0.3, no near-field artifact).
Reliable α only (β < -1 from file 16).

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/17_flat_integrand.py
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

# Lattice dispersion
c = np.sqrt(K1 + 4 * K2)  # = sqrt(3) ≈ 1.732

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
x_start = DW + 5
iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
gamma_pml = make_damping_3d(L, DW, DS)

# Clean grid only (k ≥ 0.3)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
alpha_vals = [0.25, 0.30, 0.40, 0.50]

# Lattice quantities
omega = 2 * c * np.sin(k_vals / 2)
vg = c * np.cos(k_vals / 2)


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


# ── References (shared across α) ───────────────────────
t0 = time.time()
print(f"Computing {len(k_vals)} references (L={L}, clean grid k≥0.3)...")
refs = {}
for k0 in k_vals:
    ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
    ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
    ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                      DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    refs[k0] = (ref, ux0, vx0, ns)
t_ref = time.time() - t0
print(f"  Done ({t_ref:.0f}s)\n")


# ── Scattering + invariant analysis ────────────────────

prefactor = N_POL * R_LOOP / (4 * np.pi**2)

print("=" * 70)
print("Flat integrand test: Q(k,α) = σ_tr · vg · ω²")
print("=" * 70)

summary = []
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

    # Compute invariants
    Q = sigma_tr * vg * omega**2
    Qvg = Q * vg  # = c⁴ × integrand (tautological check: CV(Qvg) ≡ CV(I))
    integrand = np.sin(k_vals)**2 * sigma_tr

    Q_mean = np.mean(Q)
    Q_cv = np.std(Q) / Q_mean * 100
    Qvg_mean = np.mean(Qvg)
    Qvg_cv = np.std(Qvg) / Qvg_mean * 100
    I_mean = np.mean(integrand)
    I_cv = np.std(integrand) / I_mean * 100

    # κ from data (trapezoidal on clean grid)
    kd = prefactor * np.trapz(integrand, k_vals)

    # κ analytic estimate: assumes Q ≈ const (decorrelated from vg),
    # then ∫ Q·vg dk ≈ Q_mean · ∫vg dk. Approximation error ~ CV(Q).
    int_vg = np.trapz(vg, k_vals)
    kd_analytic = prefactor * Q_mean / c**4 * int_vg

    dt = time.time() - t1

    print(f"\nα = {alpha:.2f}  ({dt:.0f}s)")
    print(f"  {'k':>5s}  {'σ_tr':>7s}  {'sin²·σ':>7s}  "
          f"{'Q=σvgω²':>8s}  {'Q·vg':>7s}")
    print(f"  {'-'*45}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {sigma_tr[i]:7.2f}  "
              f"{integrand[i]:7.3f}  {Q[i]:8.2f}  {Qvg[i]:7.2f}")

    print(f"  Q·vg: mean={Qvg_mean:.2f}, CV={Qvg_cv:.1f}%  <- true invariant")
    print(f"  Q:    mean={Q_mean:.2f}, CV={Q_cv:.1f}%")
    print(f"  I:    mean={I_mean:.3f}, CV={I_cv:.1f}%")
    print(f"  κ_data = {kd:.3f} (trapz), κ_analytic = {kd_analytic:.3f} "
          f"(diff {(kd_analytic - kd) / kd * 100:+.1f}%)")
    summary.append((alpha, Q_mean, Q_cv, Qvg_mean, Qvg_cv, I_cv,
                     kd, kd_analytic))


# ── Summary table ──────────────────────────────────────
print()
print("=" * 70)
print("Summary")
print("=" * 70)
print(f"{'α':>5s}  {'Q(α)':>7s}  {'CV(Q)':>6s}  {'CV(Qvg)':>7s}  {'CV(I)':>6s}  "
      f"{'κ_data':>7s}  {'κ_anal':>7s}  {'diff':>6s}")
print("-" * 65)
for alpha, Qm, Qcv, Qvgm, Qvgcv, Icv, kd, ka in summary:
    diff = (ka - kd) / kd * 100
    print(f"{alpha:5.2f}  {Qm:7.2f}  {Qcv:5.1f}%  {Qvgcv:6.1f}%  {Icv:5.1f}%  "
          f"{kd:7.3f}  {ka:7.3f}  {diff:+5.1f}%")

# Q(α) scaling: test Q ∝ α² (AB-like)
alphas_arr = np.array([s[0] for s in summary])
Q_arr = np.array([s[1] for s in summary])
q_ratio = Q_arr / alphas_arr**2
cv_ratio = np.std(q_ratio) / np.mean(q_ratio) * 100
print()
print("Q(α) scaling:")
for i in range(len(summary)):
    print(f"  α={alphas_arr[i]:.2f}: Q={Q_arr[i]:.2f}, "
          f"Q/α² = {Q_arr[i]/alphas_arr[i]**2:.1f}")
print(f"  Q/α² CV = {cv_ratio:.1f}% — "
      f"{'consistent with AB scaling' if cv_ratio < 20 else 'deviates from AB'}")

# Also test Q ∝ sin²(πα) (exact AB)
q_ratio_ab = Q_arr / np.sin(np.pi * alphas_arr)**2
cv_ab = np.std(q_ratio_ab) / np.mean(q_ratio_ab) * 100
print()
print("Q(α) vs sin²(πα) (exact AB):")
for i in range(len(summary)):
    print(f"  α={alphas_arr[i]:.2f}: Q/sin²(πα) = "
          f"{q_ratio_ab[i]:.1f}")
print(f"  Q/sin²(πα) CV = {cv_ab:.1f}%")

t_total = time.time() - t0
print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
