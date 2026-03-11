"""
Test: +z incidence (along ring axis) vs +x incidence (perpendicular to ring axis).

The ring is in the xy-plane with axis along z. For a full direction average:
  <σ_tr> = (1/2) ∫_0^π σ_tr(θ_inc) sin(θ_inc) dθ_inc

Currently only θ_inc = π/2 (+x, +y) is measured. This test adds θ_inc = 0 (+z).

Test: ux-polarized wave packet propagating in +z.
  - σ(+z) vs σ(+x): quantifies directional dependence
  - σ_tr(+z) with transport kernel (1 - cosθ) where θ = angle from +z

Results (L=80, α=0.3, R=5, k=0.5):
  σ(+x)    = 28.99,  σ_tr(+x) = 14.16  (kernel: 1 - sinθ cosφ)
  σ(+z)    = 68.74,  σ_tr(+z) = 80.05  (kernel: 1 - cosθ)
  ratio σ(+z)/σ(+x) = 2.37
  ratio σ_tr(+z)/σ_tr(+x) = 5.65

  +z scatters much more: wave passes THROUGH Dirac disk (entire surface),
  while +x wave only grazes the ring edge.

  Direction average ⟨σ_tr⟩ = ½∫ σ_tr(θ_inc) sin(θ_inc) dθ_inc:
    sin(0) = 0 suppresses polar contribution, but σ_tr(+z) is 5.6× larger.
    Crude 2-point interpolation: ⟨σ_tr⟩/σ_tr(+x) ≈ 2.0–2.5.
    This would increase κ by ~100–150%.

  CONCERN: gauge invariance. A perfectly gauge-invariant construction should give
  σ_tr independent of whether the wave crosses the Dirac surface. The large +z/+x
  ratio may be a gauge artifact from NNN bonds being ungauged — the Dirac surface
  is not fully transparent.

  RESOLVED by file 13 (direction scan): intermediate angles + R-dependence confirm
  gauge artifact (σ(+z) ~ R^2.7 = disk area). Direction average ⟨σ_tr⟩ = 0.97×σ_tr(+x)
  — the +x measurement IS the direction average to 3%.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/12_test_z_incidence.py
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

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
gamma = make_damping_3d(L, DW, DS)


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


def integrate_sigma_z_incidence(f2, thetas, phis):
    """σ and σ_tr for +z incidence direction.

    Transport kernel: cos_θs = cos(θ) (angle from +z).
    σ_tr = ∫ (1 - cos θ) f² dΩ.
    """
    N_th = len(thetas)
    N_ph = len(phis)
    f2_2d = f2.reshape(N_th, N_ph)

    d_th = thetas[1] - thetas[0] if N_th > 1 else np.pi
    d_ph = phis[1] - phis[0] if N_ph > 1 else 2 * np.pi

    w_th = np.ones(N_th) * d_th
    w_th[0] *= 0.5
    w_th[-1] *= 0.5

    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    sin_th = np.sin(TH)
    cos_th = np.cos(TH)   # transport kernel for +z incidence

    dOmega = sin_th * w_th[:, np.newaxis] * d_ph

    sigma = np.sum(f2_2d * dOmega)
    sigma_tr = np.sum((1 - cos_th) * f2_2d * dOmega)
    return sigma, sigma_tr


def make_wave_packet_z(L, k0, z_start, sz, K1=1.0, K2=0.5):
    """Gaussian wave packet propagating in +z, ux-polarized (transverse).

    ux(z) = envelope(z) * cos(k0 * z), uniform in x and y.
    """
    c = np.sqrt(K1 + 4 * K2)
    omega_k = 2 * np.sin(k0 / 2) * c
    iz_arr = np.arange(L, dtype=float)
    env = np.exp(-((iz_arr - z_start) ** 2) / (2 * sz ** 2))
    ux_1d = env * np.cos(k0 * iz_arr)
    vx_1d = omega_k * env * np.sin(k0 * iz_arr)

    # ux varies along z (axis 0), uniform in y (axis 1) and x (axis 2)
    ux0 = np.broadcast_to(ux_1d[:, np.newaxis, np.newaxis], (L, L, L)).copy()
    vx0 = np.broadcast_to(vx_1d[:, np.newaxis, np.newaxis], (L, L, L)).copy()
    return ux0, vx0


# ── +x incidence (standard) ───────────────────────────────

k0 = 0.5
x_start = DW + 5
f_def = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)

print(f"L={L}, α={ALPHA}, R={R_LOOP}, k={k0}, r_m={r_m}")
print()

# +x reference and defect
ux0_x, vx0_x = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)

print(f"+x incidence ({ns} steps)...")
ref_x = run_fdtd_3d(force_plain, ux0_x.copy(), vx0_x.copy(), gamma,
                    DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
def_x = run_fdtd_3d(f_def, ux0_x.copy(), vx0_x.copy(), gamma,
                    DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
f2_x = compute_sphere_f2(def_x['ux'], def_x['uy'], def_x['uz'],
                          ref_x['ux'], ref_x['uy'], ref_x['uz'], r_m)
sigma_x, sigma_tr_x = integrate_sigma_3d(f2_x, thetas, phis)
print(f"  σ(+x) = {sigma_x:.2f},  σ_tr(+x) = {sigma_tr_x:.2f}")


# ── +z incidence ──────────────────────────────────────────

z_start = DW + 5
ux0_z, vx0_z = make_wave_packet_z(L, k0, z_start, sx, K1, K2)

# Travel time for +z: same logic as estimate_n_steps_3d but along z-axis
ns_z = estimate_n_steps_3d(k0, L, z_start, sx, r_m, DT, 50, K1, K2)

print(f"\n+z incidence ({ns_z} steps)...")
ref_z = run_fdtd_3d(force_plain, ux0_z.copy(), vx0_z.copy(), gamma,
                    DT, ns_z, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns_z)
def_z = run_fdtd_3d(f_def, ux0_z.copy(), vx0_z.copy(), gamma,
                    DT, ns_z, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns_z)
f2_z = compute_sphere_f2(def_z['ux'], def_z['uy'], def_z['uz'],
                          ref_z['ux'], ref_z['uy'], ref_z['uz'], r_m)

# σ and σ_tr with +z transport kernel
sigma_z, sigma_tr_z = integrate_sigma_z_incidence(f2_z, thetas, phis)

print(f"  σ(+z) = {sigma_z:.2f},  σ_tr(+z) = {sigma_tr_z:.2f}")


# ── Comparison ────────────────────────────────────────────

print()
print("=" * 50)
print(f"  σ(+x)    = {sigma_x:.2f}")
print(f"  σ(+z)    = {sigma_z:.2f}")
print(f"  ratio σ(+z)/σ(+x) = {sigma_z/sigma_x:.3f}")
print()
print(f"  σ(+x)    = {sigma_x:.2f},   σ(+z)    = {sigma_z:.2f},   ratio = {sigma_z/sigma_x:.3f}")
print(f"  σ_tr(+x) = {sigma_tr_x:.2f}  (kernel: 1 - sinθ cosφ)")
print(f"  σ_tr(+z) = {sigma_tr_z:.2f}  (kernel: 1 - cosθ)")
print(f"  ratio σ_tr(+z)/σ_tr(+x) = {sigma_tr_z/sigma_tr_x:.3f}")
print()

# Direction-averaged σ_tr (crude: 2 points only)
# <σ_tr> ≈ weight_perp × σ_tr(+x) + weight_para × σ_tr(+z)
# For uniform sphere: weight of θ_inc ∈ [π/4, 3π/4] (perpendicular-ish) vs rest
# Simplest: <σ_tr> = (2 × σ_tr(+x) + σ_tr(+z)) / 3  (solid angle weighted)
# More precise: σ_tr(θ_inc=π/2) weighted by sin(π/2), σ_tr(θ_inc=0) weighted by sin(0)→0
# So +z contributes negligibly to direction average!
# But let's be honest and just report both numbers.

print("Direction average note:")
print("  In ⟨σ_tr⟩ = ½ ∫ σ_tr(θ_inc) sin(θ_inc) dθ_inc,")
print("  θ_inc = 0 (+z) has sin(0) = 0 weight → negligible contribution.")
print("  θ_inc = π/2 (+x, +y) has sin(π/2) = 1 → dominant.")
print("  The in-plane measurement is the relevant one for the average.")
