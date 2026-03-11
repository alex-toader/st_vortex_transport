"""
Phase 4: κ extraction from 3D FDTD scattering.

Measures σ_tr(k) at multiple k values for given α and R_loop,
then computes κ = (N_pol × R/(4π²)) ∫₀^π sin²(k) σ_tr(k) dk.

Formula derivation:
  γ = Σ_pol ∫ n₁(ω) σ_tr_pol(ω) vg(ω) dω    [bombardment rate]
  n₁(ω) = ω²/(4π²c³)               [zero-point Debye, per branch: g₁/(2π²c³) × ½]
  Change variable ω → k:  dω = vg dk, ω = 2c sin(k/2)
  Per branch: γ₁ = (c/(4π²)) ∫₀^π sin²(k) σ_tr(k) dk

  Option B: gauge acts on (ux, uy) only → uz does not scatter (uz_sc = 0 exact).
  By xy-symmetry of the ring: σ_tr(ux-pol) = σ_tr(uy-pol) = σ_tr (measured).
  N_pol = 2 effective polarizations.

  γ = N_pol × (c/(4π²)) ∫₀^π sin²(k) σ_tr(k) dk
  κ = γR/c = (N_pol × R/(4π²)) ∫₀^π sin²(k) σ_tr(k) dk

Results (L=80, r_m=20, N_pol=2, full recording):
  α=0.3: κ = 1.30  (σ_tr ~ k^-1.76, tail 52%)
  α=0.5: κ = 2.64  (σ_tr ~ k^-1.48, tail 60%)
  Tail sensitivity (3-pt vs 2-pt fit): ±0.03 — NOT a statistical error,
  measures sensitivity to including one more data point in power law fit.
  Real tail uncertainty is larger (power law form itself is assumed).

  N_pol=2 verified: σ_tr(uy-pol)/σ_tr(ux-pol) = 1.0000 at k=0.5.
  α-ratio: κ(0.5)/κ(0.3) = 2.03 vs AB sin²(πα) ratio 1.53 — 32% discrepancy.

  WARNING: κ(α=0.5) = 2.64 is a serious discrepancy with target κ=1.
  If α=1/2 (topological prediction), this is NOT just "systematic uncertainty"
  — it's a 2.6× discrepancy. NNN ungauged would make κ LARGER (more scattering
  if NNN were gauged), so NNN does not explain the excess.
  Possible explanations: (1) α ≠ 1/2, (2) large lattice artifacts at α=0.5,
  (3) genuine model problem at strong scattering.
  At α=0.3: κ=1.30 is within ~30% of target — much more consistent.

Systematic uncertainties:
  - Tail contributes ~50-60%: power law extrapolation from 3-4 data points
  - Near-field at low k (λ > r_m): k < 0.3 data unreliable at r_m=20
  - NNN bonds ungauged: ~50% of spring constant bypasses Dirac surface
  - r_m dependence: ~7% spread at k=0.5 (near-field systematic)
  - kR universality fails at k=0.3 (r_m/λ = 0.72, not far-field)
  - Direction averaging: only x-incidence measured, ring is axisymmetric

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/7_kappa_extraction.py
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


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


def measure_sigma_tr(k_vals, alpha, R_loop, L, DW, DS, DT, sx, r_m,
                     thetas, phis, n_buf=50):
    """Measure σ_tr at multiple k values. Returns arrays (sigma, sigma_tr)."""
    x_start = DW + 5
    iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
    gamma_pml = make_damping_3d(L, DW, DS)

    sigma_arr = np.zeros(len(k_vals))
    sigma_tr_arr = np.zeros(len(k_vals))

    for i, k0 in enumerate(k_vals):
        ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
        ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, n_buf, K1, K2)

        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                          DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f_def = make_vortex_force(alpha, R_loop, L, K1, K2)
        d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                        DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)

        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        s, st = integrate_sigma_3d(f2, thetas, phis)
        sigma_arr[i] = s
        sigma_tr_arr[i] = st

    return sigma_arr, sigma_tr_arr


def compute_kappa(k_data, str_data, R_loop, n_pol=2):
    """Compute κ from measured σ_tr data + power law tail extrapolation.

    n_pol: number of polarizations that scatter (2 for Option B: ux, uy only).

    Returns (kappa_data, kappa_tail, kappa_total, beta).
    """
    prefactor = n_pol * R_loop / (4 * np.pi**2)

    # Trapezoidal on measured data
    integrand_data = np.sin(k_data)**2 * str_data
    kappa_data = prefactor * np.trapz(integrand_data, k_data)

    # Power law tail: fit on points with k > 0.4 (avoid near-field low-k)
    mask = k_data > 0.4
    beta, logA = np.polyfit(np.log(k_data[mask]), np.log(str_data[mask]), 1)
    assert beta < -1.0, f"Power law slope {beta:.2f} > -1.0: unphysical (σ_tr should decrease)"
    A = np.exp(logA)
    k_tail = np.linspace(k_data[-1], np.pi, 200)[1:]  # [1:] avoids double-counting
    integrand_tail = np.sin(k_tail)**2 * A * k_tail**beta
    kappa_tail = prefactor * np.trapz(integrand_tail, k_tail)

    # Tail error estimate: fit on last 2 vs all k>0.4
    beta2, logA2 = np.polyfit(np.log(k_data[-2:]), np.log(str_data[-2:]), 1)
    A2 = np.exp(logA2)
    integrand_tail2 = np.sin(k_tail)**2 * A2 * k_tail**beta2
    kappa_tail2 = prefactor * np.trapz(integrand_tail2, k_tail)
    tail_err = abs(kappa_tail2 - kappa_tail)

    return kappa_data, kappa_tail, kappa_data + kappa_tail, beta, tail_err


# ── Configuration ────────────────────────────────────────────

L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

k_vals = np.array([0.15, 0.2, 0.3, 0.5, 0.7, 0.9])
R_LOOP = 5


# ── Measure and compute ─────────────────────────────────────

for ALPHA in [0.3, 0.5]:
    print(f"α = {ALPHA}, R_loop = {R_LOOP}")

    sigma, sigma_tr = measure_sigma_tr(
        k_vals, ALPHA, R_LOOP, L, DW, DS, DT, sx, r_m, thetas, phis)

    for i, k0 in enumerate(k_vals):
        print(f"  k={k0:.2f} (kR={k0*R_LOOP:.1f}): "
              f"σ={sigma[i]:.1f}, σ_tr={sigma_tr[i]:.1f}")

    kd, kt, ktot, beta, terr = compute_kappa(k_vals, sigma_tr, R_LOOP)
    print(f"  κ_data (k={k_vals[0]:.2f}-{k_vals[-1]:.2f}) = {kd:.3f}")
    print(f"  κ_tail (k={k_vals[-1]:.2f}-π, ~k^{beta:.2f}) = {kt:.3f} ± {terr:.3f}")
    print(f"  κ_total = {ktot:.3f} ± {terr:.3f}")
    print()


# ── Summary ──────────────────────────────────────────────────

print("=" * 50)
print(f"Domain: L={L}, r_m={r_m}, R_loop={R_LOOP}")
print(f"Angular grid: {len(thetas)}×{len(phis)} = {len(thetas)*len(phis)} pts")
print(f"k range: {k_vals[0]:.2f} to {k_vals[-1]:.2f} ({len(k_vals)} points)")
print(f"Full recording (rec_n = n_steps)")


# ── N_pol=2 verification: uy-polarized wave ──────────────────

print()
print("=" * 50)
print("N_pol check: σ_tr(uy-pol) vs σ_tr(ux-pol) at k=0.5, α=0.3")

k_check = 0.5
x_start = DW + 5
iz_c, iy_c, ix_c = make_sphere_points(L, r_m, thetas, phis)
gamma_pml = make_damping_3d(L, DW, DS)
ns_c = estimate_n_steps_3d(k_check, L, x_start, sx, r_m, DT, 50, K1, K2)

# ux-polarized (standard)
ux0, vx0 = make_wave_packet_3d(L, k_check, x_start, sx, K1, K2)
ref_x = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                    DT, ns_c, rec_iz=iz_c, rec_iy=iy_c, rec_ix=ix_c, rec_n=ns_c)
f_def = make_vortex_force(0.3, R_LOOP, L, K1, K2)
def_x = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                    DT, ns_c, rec_iz=iz_c, rec_iy=iy_c, rec_ix=ix_c, rec_n=ns_c)
f2_x = compute_sphere_f2(def_x['ux'], def_x['uy'], def_x['uz'],
                          ref_x['ux'], ref_x['uy'], ref_x['uz'], r_m)
_, str_x = integrate_sigma_3d(f2_x, thetas, phis)

# uy-polarized: swap ux0 → uy0 (wave packet in uy, traveling in +x)
uy0 = ux0.copy()
vy0 = vx0.copy()
zeros = np.zeros_like(ux0)
ref_y = run_fdtd_3d(force_plain, zeros.copy(), zeros.copy(), gamma_pml,
                    DT, ns_c, rec_iz=iz_c, rec_iy=iy_c, rec_ix=ix_c, rec_n=ns_c,
                    uy0=uy0.copy(), vy0=vy0.copy())
def_y = run_fdtd_3d(f_def, zeros.copy(), zeros.copy(), gamma_pml,
                    DT, ns_c, rec_iz=iz_c, rec_iy=iy_c, rec_ix=ix_c, rec_n=ns_c,
                    uy0=uy0.copy(), vy0=vy0.copy())
f2_y = compute_sphere_f2(def_y['ux'], def_y['uy'], def_y['uz'],
                          ref_y['ux'], ref_y['uy'], ref_y['uz'], r_m)
_, str_y = integrate_sigma_3d(f2_y, thetas, phis)

print(f"  σ_tr(ux-pol) = {str_x:.2f}")
print(f"  σ_tr(uy-pol) = {str_y:.2f}")
print(f"  ratio = {str_y/str_x:.4f}")
print(f"  N_pol=2 {'CONFIRMED' if abs(str_y/str_x - 1) < 0.05 else 'FAILED'}"
      f" (tolerance 5%)")
