"""
Direction scan: σ_tr(θ_inc) for wave incident at angle θ_inc from +z axis.

Tests:
  A) Angular scan: θ_inc = 0, π/6, π/4, π/3, π/2
     Maps how σ_tr transitions from +z (along ring axis) to +x (perpendicular).
     Determines the direction-averaged σ_tr.

  B) R-dependence at +z: R_loop = 3, 5, 7
     If σ(+z) ∝ R² → Dirac disk area dominates (gauge artifact).
     If σ(+z) ∝ R → ring perimeter dominates (physical).
     Compare with +x where σ ~ R^1.5 (known).

Results (L=80, α=0.3, R=5, k=0.5):

  Part A — Angular scan:
    θ_inc      σ     σ_tr   σ_tr/σ_tr(+x)
    0 (+z)   68.7    79.5     5.52
    π/6      54.9    51.1     3.55
    π/4      44.9    34.2     2.37
    π/3      36.6    22.6     1.57
    π/2 (+x) 29.4    14.4     1.00

    Direction average: ⟨σ_tr⟩/σ_tr(+x) = 0.967
    → +x measurement IS the direction average to 3%.
    → Direction averaging is NOT a significant systematic.

  Part B — R-dependence at +z:
    R=3: σ=15.7, σ_tr=17.8
    R=5: σ=68.7, σ_tr=79.5
    R=7: σ=149.5, σ_tr=176.4
    σ(+z) ~ R^2.68, σ_tr(+z) ~ R^2.72
    → Scales with DISK AREA (R²), not ring perimeter (R).
    → σ/πR² → 0.97 at R=7: approaching geometric disk cross-section.
    → This is a GAUGE ARTIFACT: Dirac surface partially opaque due to NNN ungauged.

  Conclusion:
    The large σ_tr(+z) is not physical — it's the Dirac disk acting as a scatterer
    because NNN bonds are ungauged (surface not fully transparent). The +x measurement
    avoids this artifact (wave doesn't cross the disk). Direction average ≈ σ_tr(+x)
    because sin(θ_inc) suppression kills the polar artifact. The ±3% correction is
    within other systematics.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/13_test_direction_scan.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from gauge_3d import make_vortex_force
from scattering_3d import make_sphere_points, compute_sphere_f2

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
k0 = 0.5
ALPHA = 0.3
R_LOOP = 5

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
gamma = make_damping_3d(L, DW, DS)


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


def make_wave_packet_angle(L, k0, theta_inc, sx, K1=1.0, K2=0.5):
    """Gaussian wave packet propagating at angle theta_inc from +z, in xz-plane.

    Direction: n_hat = (sin(theta_inc), 0, cos(theta_inc)).
    Displacement: ux-polarized (transverse for theta != 0, pi).
    Plane wave front perpendicular to n_hat.

    Returns (ux0, vx0) arrays of shape (L, L, L).
    """
    c = np.sqrt(K1 + 4 * K2)
    omega_k = 2 * np.sin(k0 / 2) * c
    cx = cz = (L - 1) / 2.0
    sin_t = np.sin(theta_inc)
    cos_t = np.cos(theta_inc)

    # s = coordinate along propagation direction
    # s(ix, iz) = (ix - cx) * sin_t + (iz - cz) * cos_t
    ix_arr = np.arange(L, dtype=float)
    iz_arr = np.arange(L, dtype=float)
    IZ, IY, IX = np.meshgrid(iz_arr, ix_arr, ix_arr, indexing='ij')

    s = (IX - cx) * sin_t + (IZ - cz) * cos_t

    # Start packet at distance d from center, on the -n_hat side
    d = L / 2.0 - DW - 5  # same distance as standard DW+5 from edge
    s_start = -d

    env = np.exp(-((s - s_start) ** 2) / (2 * sx ** 2))
    # Phase: k0 * (s - s_start) so that phase=0 at packet center
    phase = k0 * (s - s_start)
    ux0 = env * np.cos(phase)
    vx0 = omega_k * env * np.sin(phase)
    return ux0, vx0


def integrate_sigma_direction(f2, thetas, phis, theta_inc):
    """σ and σ_tr for incidence at angle theta_inc from +z (phi_inc = 0).

    Transport kernel: cos_θs = sin(θ_inc) sin(θ) cos(φ) + cos(θ_inc) cos(θ).
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
    cos_th = np.cos(TH)

    cos_ts = (np.sin(theta_inc) * sin_th * np.cos(PH)
              + np.cos(theta_inc) * cos_th)

    dOmega = sin_th * w_th[:, np.newaxis] * d_ph

    sigma = np.sum(f2_2d * dOmega)
    sigma_tr = np.sum((1 - cos_ts) * f2_2d * dOmega)
    return sigma, sigma_tr


def estimate_nsteps(k0, L, DW, sx, r_m, dt, K1, K2, n_buf=50):
    """Steps for wave to reach center and scatter to r_m."""
    c = np.sqrt(K1 + 4 * K2)
    vg = c * np.cos(k0 / 2)
    d = L / 2.0 - DW - 5
    t_travel = (d + 2 * sx) / vg + r_m / c
    return int(t_travel / dt) + n_buf


# ── Part A: Angular scan ──────────────────────────────────

angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
angle_names = ['0 (+z)', 'π/6', 'π/4', 'π/3', 'π/2 (+x)']

ns = estimate_nsteps(k0, L, DW, sx, r_m, DT, K1, K2)
f_def = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)

print(f"Part A: Angular scan (L={L}, α={ALPHA}, R={R_LOOP}, k={k0}, {ns} steps)")
print(f"{'θ_inc':>8s}  {'σ':>7s}  {'σ_tr':>7s}  {'σ/σ(+x)':>8s}  {'σ_tr/σ_tr(+x)':>14s}")
print("-" * 55)

results_a = []
for theta_inc, name in zip(angles, angle_names):
    ux0, vx0 = make_wave_packet_angle(L, k0, theta_inc, sx, K1, K2)

    ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma,
                      DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma,
                    DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                           ref['ux'], ref['uy'], ref['uz'], r_m)
    sigma, sigma_tr = integrate_sigma_direction(f2, thetas, phis, theta_inc)
    results_a.append((theta_inc, name, sigma, sigma_tr))
    print(f"{name:>8s}  {sigma:7.1f}  {sigma_tr:7.1f}", end="")
    if len(results_a) > 1:
        s0, st0 = results_a[-1][2], results_a[-1][3]
        sx_val, stx = results_a[0][2], results_a[0][3]  # will normalize to +x at end
    print()

# Normalize to +x
sigma_x = results_a[-1][2]
str_x = results_a[-1][3]
print()
for theta_inc, name, sigma, sigma_tr in results_a:
    print(f"{name:>8s}  σ/σ(+x) = {sigma/sigma_x:.3f}  "
          f"σ_tr/σ_tr(+x) = {sigma_tr/str_x:.3f}")

# Direction average
print()
print("Direction average:")
th_arr = np.array([r[0] for r in results_a])
str_arr = np.array([r[3] for r in results_a])
# Trapezoidal: <σ_tr> = (1/2) ∫ σ_tr(θ) sin(θ) dθ
integrand = str_arr * np.sin(th_arr)
avg_str = 0.5 * np.trapz(integrand, th_arr)
print(f"  ⟨σ_tr⟩ = {avg_str:.1f}  (5-point trapezoidal)")
print(f"  ⟨σ_tr⟩ / σ_tr(+x) = {avg_str / str_x:.3f}")
print(f"  Effect on κ: multiply by {avg_str / str_x:.2f}")


# ── Part B: R-dependence at +z ────────────────────────────

print()
print(f"Part B: R-dependence at +z (L={L}, α={ALPHA}, k={k0})")
print(f"{'R':>3s}  {'σ(+z)':>7s}  {'σ_tr(+z)':>8s}  {'πR²':>5s}  {'σ/πR²':>6s}")
print("-" * 40)

R_vals = [3, 5, 7]
results_b = []
for R in R_vals:
    f_def_R = make_vortex_force(ALPHA, R, L, K1, K2)
    # +z incidence
    ux0, vx0 = make_wave_packet_angle(L, k0, 0, sx, K1, K2)
    ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma,
                      DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    d = run_fdtd_3d(f_def_R, ux0.copy(), vx0.copy(), gamma,
                    DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                           ref['ux'], ref['uy'], ref['uz'], r_m)
    sigma, sigma_tr = integrate_sigma_direction(f2, thetas, phis, 0)
    results_b.append((R, sigma, sigma_tr))
    geo = np.pi * R**2
    print(f"{R:3d}  {sigma:7.1f}  {sigma_tr:8.1f}  {geo:5.1f}  {sigma/geo:6.3f}")

# Fit power law σ(+z) ~ R^p
Rs = np.array([r[0] for r in results_b], dtype=float)
sigmas_z = np.array([r[1] for r in results_b])
strs_z = np.array([r[2] for r in results_b])

p_sigma, _ = np.polyfit(np.log(Rs), np.log(sigmas_z), 1)
p_str, _ = np.polyfit(np.log(Rs), np.log(strs_z), 1)

print()
print(f"  σ(+z) ~ R^{p_sigma:.2f}  (disk area scaling: R^2.0)")
print(f"  σ_tr(+z) ~ R^{p_str:.2f}  (disk area scaling: R^2.0)")
print()
if p_sigma > 1.7:
    print("  σ(+z) scales closer to disk area → likely Dirac surface artifact")
elif p_sigma < 1.3:
    print("  σ(+z) scales closer to ring perimeter → likely physical")
else:
    print("  σ(+z) intermediate scaling → mixed contribution")
