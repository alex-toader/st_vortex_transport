"""
Phase 3 diagnostics: σ_tr scaling with R_loop and k.

Three investigations:
  D1  σ_tr vs R_loop at fixed k=0.5, α=0.3 — geometric scaling
  D2  uz fraction in scattered field — Option B check
  D3  σ_tr vs k at fixed R_loop=5, α=0.3 — frequency dependence

Physics expectations:
  - D1: ring (not disk) → may see σ_tr ~ R^1 at kR >> 1, not R^2
  - D2: Option B rotates (ux,uy) only → uz_sc should be negligible
  - D3: 3D vortex ring ≠ 2D AB → σ_tr ~ 1/k NOT expected generically

Results (L=80, r_m=20, α=0.3, full recording):
  D1: σ_tr ~ R^1.51 (intermediate between ring R^1 and disk R^2, kR = 1.5–4.5)
      R=3: σ_tr=6.4, R=5: σ_tr=14.2, R=7: σ_tr=22.2, R=9: σ_tr=34.5
  D2: uz fraction = 0.00% exactly (uz decoupled: gauge acts on ux,uy only,
      scalar Laplacian treats components independently → uz=0 is exact invariant)
  D3: σ_tr ~ k^-1.9 (steeper than 2D AB k^-1, consistent with 3D finite ring)
      k=0.3: σ_tr=40.7, k=0.5: σ_tr=14.2, k=0.7: σ_tr=7.7, k=0.9: σ_tr=5.1

NOTE: Earlier results with rec_n=50 (partial recording) gave σ_tr ~ k^+0.6.
That was an artifact of the recording window missing the incident pulse at
different sphere points. Full recording (rec_n = n_steps) is required.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/6_scattering_diagnostics.py
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
ALPHA = 0.3

# Common domain
L = 80
DW = 15
DS = 1.5
DT = 0.25
r_m = 20
N_BUF = 50  # extra steps after wave reaches sphere
sx = 8.0
x_start = DW + 5

thetas = np.linspace(0, np.pi, 13)       # 15 deg step
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
iz_s, iy_s, ix_s = make_sphere_points(L, r_m, thetas, phis)
gamma = make_damping_3d(L, DW, DS)


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


# ── D1: σ_tr vs R_loop ──────────────────────────────────────

print("D1: σ_tr vs R_loop (k=0.5, α=0.3)")
k0_d1 = 0.5

ux0_d1, vx0_d1 = make_wave_packet_3d(L, k0_d1, x_start, sx, K1, K2)
ns_d1 = estimate_n_steps_3d(k0_d1, L, x_start, sx, r_m, DT, N_BUF, K1, K2)

# Single reference (plain Laplacian)
ref_d1 = run_fdtd_3d(force_plain, ux0_d1.copy(), vx0_d1.copy(), gamma, DT, ns_d1,
                     rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns_d1)

R_loops = [3, 5, 7, 9]
sigma_tr_R = []
sigma_R = []
uz_fracs = []

for R_loop in R_loops:
    f_def = make_vortex_force(ALPHA, R_loop, L, K1, K2)
    d = run_fdtd_3d(f_def, ux0_d1.copy(), vx0_d1.copy(), gamma, DT, ns_d1,
                    rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns_d1)

    f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                           ref_d1['ux'], ref_d1['uy'], ref_d1['uz'], r_m)
    sigma, sigma_tr = integrate_sigma_3d(f2, thetas, phis)
    sigma_tr_R.append(sigma_tr)
    sigma_R.append(sigma)

    # uz fraction (D2 data)
    sc_ux = d['ux'] - ref_d1['ux']
    sc_uy = d['uy'] - ref_d1['uy']
    sc_uz = d['uz'] - ref_d1['uz']
    uxy2 = np.mean(sc_ux**2 + sc_uy**2)
    uz2 = np.mean(sc_uz**2)
    frac = uz2 / (uxy2 + uz2) if (uxy2 + uz2) > 0 else 0.0
    uz_fracs.append(frac)

    print(f"  R={R_loop}: σ={sigma:.2f}, σ_tr={sigma_tr:.2f}, "
          f"kR={k0_d1*R_loop:.1f}, uz_frac={frac:.4f}")

log_R = np.log(np.array(R_loops, dtype=float))
log_sR = np.log(np.array(sigma_tr_R))
slope_R = np.polyfit(log_R, log_sR, 1)[0]
print(f"  slope σ_tr ~ R^{slope_R:.2f}")
print()


# ── D2: uz decomposition summary ────────────────────────────

print("D2: uz fraction in scattered field (Option B)")
for i, R_loop in enumerate(R_loops):
    print(f"  R={R_loop}: {uz_fracs[i]*100:.2f}%")
print()


# ── D3: σ_tr vs k ──────────────────────────────────────────

print("D3: σ_tr vs k (R_loop=5, α=0.3)")
R_loop_d3 = 5
k_vals = [0.3, 0.5, 0.7, 0.9]
sigma_tr_k = []
sigma_k = []

for k0 in k_vals:
    ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
    ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, N_BUF, K1, K2)

    ref_k = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma, DT, ns,
                        rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
    f_def = make_vortex_force(ALPHA, R_loop_d3, L, K1, K2)
    def_k = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma, DT, ns,
                        rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)

    f2 = compute_sphere_f2(def_k['ux'], def_k['uy'], def_k['uz'],
                           ref_k['ux'], ref_k['uy'], ref_k['uz'], r_m)
    sigma, sigma_tr = integrate_sigma_3d(f2, thetas, phis)
    sigma_tr_k.append(sigma_tr)
    sigma_k.append(sigma)

    print(f"  k={k0:.1f}: σ={sigma:.2f}, σ_tr={sigma_tr:.2f}, kR={k0*R_loop_d3:.1f}")

log_k = np.log(np.array(k_vals))
log_sk = np.log(np.array(sigma_tr_k))
slope_k = np.polyfit(log_k, log_sk, 1)[0]
print(f"  slope σ_tr ~ k^{slope_k:.2f}")
print()

print("=" * 50)
print("Summary:")
print(f"  σ_tr ~ R^{slope_R:.2f}  (ring: R^1, disk: R^2)")
print(f"  σ_tr ~ k^{slope_k:.2f}  (2D AB: k^-1)")
print(f"  uz fraction: {min(uz_fracs)*100:.2f}% - {max(uz_fracs)*100:.2f}%")
