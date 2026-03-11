"""
Phase 3 tests for scattering_3d.py — wave packet, sphere, sigma_tr pipeline.

4 tests:
  T1  Wave packet propagation: peak arrives at correct time
  T2  Sphere sampling: unique pixels, points within domain
  T3  Integration sanity: known f^2 = const gives sigma = 4*pi*r_m^2 * const
  T4  No-scattering baseline: alpha=0 gives sigma_tr ~ 0

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/5_test_scattering_3d.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from gauge_3d import make_vortex_force
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           compute_sphere_f2, integrate_sigma_3d,
                           group_velocity_3d, estimate_n_steps_3d)

K1, K2 = 1.0, 0.5
C_3D = np.sqrt(K1 + 4 * K2)  # sqrt(3)

passed = 0
failed = 0


def report(name, ok, detail=""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {tag}: {name}" + (f"  ({detail})" if detail else ""))


# ── T1: Wave packet propagation speed ────────────────────────

print("T1: Wave packet propagation speed")
L1 = 60
DW1 = 12
DS1 = 1.5
DT1 = 0.2
k0_test = 0.5
x_start1 = DW1 + 5
sx1 = 8.0

ux0, vx0 = make_wave_packet_3d(L1, k0_test, x_start1, sx1, K1, K2)

# Verify shape
report("shape", ux0.shape == (L1, L1, L1), f"got {ux0.shape}")

# Verify uniformity in y, z
report("uniform in y", np.max(np.abs(ux0[15, :, :] - ux0[15, 0:1, :])) == 0.0)
report("uniform in z", np.max(np.abs(ux0[:, 15, :] - ux0[0:1, 15, :])) == 0.0)

# Run FDTD without defect, measure peak position
gamma1 = make_damping_3d(L1, DW1, DS1)

def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)

n_steps1 = 40  # keep wave in interior (away from PML)
ux_final, _, _ = run_fdtd_3d(force_plain, ux0, vx0, gamma1, DT1, n_steps1)

# Peak position along center line (iy=cy, iz=cz) via sliding-window envelope
cy1 = L1 // 2
profile_init = ux0[cy1, cy1, :]
profile_final = ux_final[cy1, cy1, :]

def envelope_peak(profile, hw=4):
    """Peak of sliding-window RMS envelope, sub-pixel via parabolic fit."""
    x_arr = np.arange(len(profile), dtype=float)
    env = np.array([np.sqrt(np.mean(profile[max(0, i-hw):i+hw+1]**2))
                    for i in range(len(profile))])
    ip = np.argmax(env)
    if 1 <= ip <= len(env) - 2:
        a, b, c = env[ip-1], env[ip], env[ip+1]
        dp = 0.5 * (a - c) / (a - 2*b + c) if (a - 2*b + c) != 0 else 0
        return ip + dp
    return float(ip)

x_peak_init = envelope_peak(profile_init)
x_peak_final = envelope_peak(profile_final)
dx_measured = x_peak_final - x_peak_init
vg = group_velocity_3d(k0_test, K1, K2)
dx_expected = vg * n_steps1 * DT1
err = abs(dx_measured - dx_expected) / dx_expected
report("propagation speed", err < 0.05, f"err = {err:.3f}")


# ── T2: Sphere sampling ──────────────────────────────────────

print("T2: Sphere sampling")
L2 = 60
r_m2 = 15
thetas2 = np.linspace(0, np.pi, 13)       # 15 deg step
phis2 = np.linspace(0, 2 * np.pi, 24, endpoint=False)  # 15 deg step

iz2, iy2, ix2 = make_sphere_points(L2, r_m2, thetas2, phis2)
N_pts2 = len(thetas2) * len(phis2)

report("point count", len(iz2) == N_pts2, f"got {len(iz2)}, expected {N_pts2}")

# All points within domain
report("within domain",
       np.all(iz2 >= 0) and np.all(iz2 < L2) and
       np.all(iy2 >= 0) and np.all(iy2 < L2) and
       np.all(ix2 >= 0) and np.all(ix2 < L2))

# Check unique pixels (excluding poles)
interior_mask = (thetas2 > 0.01) & (thetas2 < np.pi - 0.01)
n_interior = np.sum(interior_mask) * len(phis2)
# Count unique (iz, iy, ix) triples for interior angles
interior_indices = []
for j, th in enumerate(thetas2):
    if th < 0.01 or th > np.pi - 0.01:
        continue
    for k in range(len(phis2)):
        idx = j * len(phis2) + k
        interior_indices.append((iz2[idx], iy2[idx], ix2[idx]))
n_unique = len(set(interior_indices))
report("unique interior pixels", n_unique > 0.9 * n_interior,
       f"{n_unique}/{n_interior} unique")


# ── T3: Integration sanity ───────────────────────────────────

print("T3: Integration sanity")
# If f^2 = 1 everywhere on sphere, sigma should be 4*pi (full solid angle)
thetas3 = np.linspace(0, np.pi, 37)       # 5 deg step
phis3 = np.linspace(0, 2 * np.pi, 72, endpoint=False)  # 5 deg step
f2_const = np.ones(len(thetas3) * len(phis3))

sigma3, sigma_tr3 = integrate_sigma_3d(f2_const, thetas3, phis3)
report("sigma(f2=1) = 4*pi", abs(sigma3 - 4 * np.pi) / (4 * np.pi) < 0.01,
       f"got {sigma3:.4f}, expected {4*np.pi:.4f}")

# sigma_tr with f2=1: integral (1-cos_theta_s) dOmega
# cos_theta_s = sin(theta)*cos(phi). integral over phi of cos_theta_s = 0.
# So sigma_tr(f2=1) = integral dOmega = 4*pi.
report("sigma_tr(f2=1) = 4*pi", abs(sigma_tr3 - 4 * np.pi) / (4 * np.pi) < 0.01,
       f"got {sigma_tr3:.4f}, expected {4*np.pi:.4f}")

# f^2 = cos^2(theta_s): integral cos^2(theta_s) dOmega = 4*pi/3
TH3, PH3 = np.meshgrid(thetas3, phis3, indexing='ij')
f2_cos2 = (np.sin(TH3) * np.cos(PH3)).ravel()**2
sigma_cos2, _ = integrate_sigma_3d(f2_cos2, thetas3, phis3)
report("sigma(cos^2 theta_s) = 4pi/3",
       abs(sigma_cos2 - 4 * np.pi / 3) / (4 * np.pi / 3) < 0.02,
       f"got {sigma_cos2:.4f}, expected {4*np.pi/3:.4f}")


# ── T4: No-scattering baseline ───────────────────────────────
# Reference: plain Laplacian (no gauge code). Defect: make_vortex_force(alpha=0).
# Tests that gauge_3d at alpha=0 reduces to Laplacian in full FDTD context.

print("T4: No-scattering baseline (alpha=0)")
L4 = 60
DW4 = 12
DS4 = 1.5
DT4 = 0.25
k0_4 = 0.5
x_start4 = DW4 + 5
sx4 = 8.0
R_LOOP4 = 5.0
r_m4 = 15
N_BUF4 = 50  # extra steps after wave reaches sphere (buffer for scattered pulse)

thetas4 = np.linspace(0, np.pi, 7)       # 30 deg step
phis4 = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # 30 deg step

iz4, iy4, ix4 = make_sphere_points(L4, r_m4, thetas4, phis4)
gamma4 = make_damping_3d(L4, DW4, DS4)
ux04, vx04 = make_wave_packet_3d(L4, k0_4, x_start4, sx4, K1, K2)
ns4 = estimate_n_steps_3d(k0_4, L4, x_start4, sx4, r_m4, DT4, N_BUF4, K1, K2)

# Reference: plain Laplacian (no gauge code at all)
def force_plain4(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)

# Full recording (rec_n=ns4) — required for correct per-point normalization
ref4 = run_fdtd_3d(force_plain4, ux04.copy(), vx04.copy(), gamma4, DT4, ns4,
                   rec_iz=iz4, rec_iy=iy4, rec_ix=ix4, rec_n=ns4)

# Defect: gauge at alpha=0 — different code path, should give same result
f_def4 = make_vortex_force(0.0, R_LOOP4, L4, K1, K2)
def4 = run_fdtd_3d(f_def4, ux04.copy(), vx04.copy(), gamma4, DT4, ns4,
                   rec_iz=iz4, rec_iy=iy4, rec_ix=ix4, rec_n=ns4)

f2_4 = compute_sphere_f2(def4['ux'], def4['uy'], def4['uz'],
                         ref4['ux'], ref4['uy'], ref4['uz'], r_m4)
sigma4, sigma_tr4 = integrate_sigma_3d(f2_4, thetas4, phis4)

report("sigma_tr(alpha=0) ~ 0", sigma_tr4 < 1e-20,
       f"sigma_tr = {sigma_tr4:.2e}")

# Verify incident field is nonzero (wave reached sphere)
inc2_max = np.max(np.mean(ref4['ux']**2 + ref4['uy']**2 + ref4['uz']**2, axis=0))
report("incident field nonzero at sphere", inc2_max > 1e-6,
       f"max <|u_inc|^2> = {inc2_max:.4e}")


# ── T5: Scattering at alpha=0.3 ────────────────────────────
# Sanity: alpha > 0 vortex produces measurable scattering.

print("T5: Scattering at alpha=0.3")
ALPHA5 = 0.3

# Reuse T4 geometry (same L, PML, sphere, wave packet)
f_def5 = make_vortex_force(ALPHA5, R_LOOP4, L4, K1, K2)
def5 = run_fdtd_3d(f_def5, ux04.copy(), vx04.copy(), gamma4, DT4, ns4,
                   rec_iz=iz4, rec_iy=iy4, rec_ix=ix4, rec_n=ns4)

f2_5 = compute_sphere_f2(def5['ux'], def5['uy'], def5['uz'],
                         ref4['ux'], ref4['uy'], ref4['uz'], r_m4)
sigma5, sigma_tr5 = integrate_sigma_3d(f2_5, thetas4, phis4)

report("sigma_tr(alpha=0.3) > 0", sigma_tr5 > 1e-3,
       f"sigma_tr = {sigma_tr5:.4f}")
# Physical bound: (1 - cos theta_s) in [0, 2] => sigma_tr <= 2*sigma.
# sigma_tr > sigma is normal (backscattering-dominated).
report("sigma_tr <= 2*sigma (physical bound)", sigma_tr5 <= 2 * sigma5,
       f"sigma = {sigma5:.4f}, sigma_tr = {sigma_tr5:.4f}")


# ── Summary ──────────────────────────────────────────────────

print()
print(f"Phase 3 scattering_3d: {passed}/{passed+failed} PASS")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
