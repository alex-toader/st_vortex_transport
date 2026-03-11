"""
scattering_3d.py — 3D scattering measurement: wave packet, sphere, sigma_tr.

Plane wave in +x -> vortex ring defect -> measure transport cross-section.

3D analog of measurement_ring.py + make_wave_packet from 2D infrastructure.

Wave packet: Gaussian envelope * cos(k0*x), propagating in +x.
Dispersion: omega(k) = 2 sin(k/2) * c, c = sqrt(K1 + 4*K2).
At K1=1, K2=0.5: c = sqrt(3).

Sphere sampling: regular (theta, phi) grid, nearest-pixel to lattice.
theta = polar angle from z-axis, phi = azimuthal from x-axis.

Cross-section:
  f^2(Omega) = r_m^2 * <|u_sc|^2> / <|u_inc|^2>
  sigma_tr = integral (1 - cos theta_s) * f^2 * dOmega
  cos theta_s = sin(theta) * cos(phi)   (scattering angle from +x)

API:
  make_wave_packet_3d(L, k0, x_start, sx, K1, K2) -> (ux0, vx0)
  make_sphere_points(L, r_m, thetas, phis) -> (iz, iy, ix)
  compute_sphere_f2(def_ux, ..., ref_ux, ..., r_m) -> f2
  integrate_sigma_3d(f2, thetas, phis) -> (sigma, sigma_tr)
  group_velocity_3d(k0, K1, K2) -> float
  estimate_n_steps_3d(...) -> int
"""

import numpy as np


def make_wave_packet_3d(L, k0, x_start, sx, K1=1.0, K2=0.5):
    """Longitudinal Gaussian wave packet propagating in +x.

    Returns (ux0, vx0): initial x-displacement and x-velocity (L, L, L).
    Uniform in y and z (plane wave front).
    Other components (uy, uz, vy, vz) are zero — pass to run_fdtd_3d.

    Dispersion: omega(k) = 2 sin(k/2) * sqrt(K1 + 4*K2).
    """
    c = np.sqrt(K1 + 4 * K2)
    omega_k = 2 * np.sin(k0 / 2) * c
    ix_arr = np.arange(L, dtype=float)
    env = np.exp(-((ix_arr - x_start) ** 2) / (2 * sx ** 2))
    ux_1d = env * np.cos(k0 * ix_arr)
    vx_1d = omega_k * env * np.sin(k0 * ix_arr)

    ux0 = np.broadcast_to(ux_1d[np.newaxis, np.newaxis, :], (L, L, L)).copy()
    vx0 = np.broadcast_to(vx_1d[np.newaxis, np.newaxis, :], (L, L, L)).copy()
    return ux0, vx0


def make_sphere_points(L, r_m, thetas, phis):
    """Sphere point indices at radius r_m from lattice center.

    Parameters
    ----------
    L : lattice size (cubic L x L x L)
    r_m : measurement radius (must be < L/2 - PML_width to stay outside PML)
    thetas : 1D array of polar angles (from z-axis)
    phis : 1D array of azimuthal angles (from x-axis)

    Returns
    -------
    iz, iy, ix : integer arrays of shape (N_theta * N_phi,)
    """
    c = (L - 1) / 2.0
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    th_flat = TH.ravel()
    ph_flat = PH.ravel()

    x = r_m * np.sin(th_flat) * np.cos(ph_flat)
    y = r_m * np.sin(th_flat) * np.sin(ph_flat)
    z = r_m * np.cos(th_flat)

    ix = np.clip(np.round(c + x).astype(int), 0, L - 1)
    iy = np.clip(np.round(c + y).astype(int), 0, L - 1)
    iz = np.clip(np.round(c + z).astype(int), 0, L - 1)
    return iz, iy, ix


def compute_sphere_f2(def_ux, def_uy, def_uz,
                      ref_ux, ref_uy, ref_uz, r_m):
    """Differential scattering amplitude on sphere points.

    f^2(Omega) = r_m^2 * <|u_sc|^2> / <|u_inc|^2>

    IMPORTANT: recordings must cover the full simulation (rec_n = n_steps),
    not just the last N steps. A short recording window causes the incident
    pulse to be captured at different sphere points at different times,
    giving up to 900x variation in inc2 and wrong sigma_tr.

    Memory: full recording at ns steps, N_pts sphere points, 3 components =
    ns * N_pts * 3 * 8 bytes. At ns=300, N_pts=312: ~2.2 MB per run.

    Parameters
    ----------
    def_ux, def_uy, def_uz : defect recordings, shape (n_rec, N_pts)
    ref_ux, ref_uy, ref_uz : reference recordings, shape (n_rec, N_pts)
    r_m : measurement radius

    Returns
    -------
    f2 : array of shape (N_pts,)
    """
    sc_ux = def_ux - ref_ux
    sc_uy = def_uy - ref_uy
    sc_uz = def_uz - ref_uz
    sc2 = np.mean(sc_ux**2 + sc_uy**2 + sc_uz**2, axis=0)
    inc2 = np.mean(ref_ux**2 + ref_uy**2 + ref_uz**2, axis=0)
    # Floor at 1e-12 * peak: prevents 0/0 at shadow points where incident
    # wave hasn't arrived. Absolute floor 1e-30 guards against zero field.
    inc2_floor = max(1e-30, 1e-12 * np.max(inc2))
    inc2 = np.maximum(inc2, inc2_floor)
    return r_m**2 * sc2 / inc2


def integrate_sigma_3d(f2, thetas, phis):
    """Integrate f^2 over the sphere to get sigma and sigma_tr.

    sigma = integral f^2 dOmega
    sigma_tr = integral (1 - cos theta_s) f^2 dOmega

    cos theta_s = sin(theta) * cos(phi): angle from +x incident direction.
    dOmega = sin(theta) * d_theta * d_phi.

    Trapezoidal in theta (endpoints halved), uniform in phi (periodic).

    Parameters
    ----------
    f2 : array of shape (N_theta * N_phi,)
    thetas : 1D array of polar angles (0 to pi)
    phis : 1D array of azimuthal angles (0 to 2*pi, periodic)

    Returns
    -------
    (sigma, sigma_tr)
    """
    N_th = len(thetas)
    N_ph = len(phis)
    assert phis[-1] < 2 * np.pi - (phis[1] - phis[0]) * 0.5, \
        "phis must not include 2*pi endpoint (use endpoint=False)"
    f2_2d = f2.reshape(N_th, N_ph)

    d_th = thetas[1] - thetas[0] if N_th > 1 else np.pi
    d_ph = phis[1] - phis[0] if N_ph > 1 else 2 * np.pi

    # Trapezoidal weights in theta (0 to pi, not periodic)
    w_th = np.ones(N_th) * d_th
    w_th[0] *= 0.5
    w_th[-1] *= 0.5

    # Solid angle grid
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    sin_th = np.sin(TH)
    cos_th_s = np.sin(TH) * np.cos(PH)

    dOmega = sin_th * w_th[:, np.newaxis] * d_ph

    sigma = np.sum(f2_2d * dOmega)
    sigma_tr = np.sum((1 - cos_th_s) * f2_2d * dOmega)
    return sigma, sigma_tr


def group_velocity_3d(k0, K1=1.0, K2=0.5):
    """Group velocity for wave along x on 3D simple cubic.

    v_g = d omega / dk = c * cos(k/2), c = sqrt(K1 + 4*K2).
    """
    c = np.sqrt(K1 + 4 * K2)
    return c * np.cos(k0 / 2)


def estimate_n_steps_3d(k0, L, x_start, sx, r_m_max, dt,
                        n_rec=100, K1=1.0, K2=0.5):
    """Estimate n_steps for wave to reach farthest sphere point + recording.

    Wave from x_start to center, scatters, propagates to r_m_max.
    """
    c = np.sqrt(K1 + 4 * K2)
    cx = (L - 1) / 2.0
    vg = group_velocity_3d(k0, K1, K2)
    t_travel = (cx - x_start + 2 * sx) / vg + r_m_max / c
    return int(t_travel / dt) + n_rec
