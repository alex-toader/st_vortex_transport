"""
elastic_3d.py — 3D elastic lattice: scalar Laplacian on simple cubic, PML, Verlet FDTD.

Simple cubic lattice with NN (K1) and NNN (K2) springs.
Scalar Laplacian: each displacement component (ux, uy, uz) independently.
Dispersion: c² = K1 + 4K2. With K1=1, K2=0.5: c = √3.

Isotropic at leading order (anisotropy < 0.1% at k=1).

Boundary: OPEN (not periodic). Sites at domain edges have fewer neighbors.
Designed for use with PML absorbing layer — without PML, edge forces are
incomplete. This matches the 2D infrastructure (elastic_lattice_2d.py).

API:
  scalar_laplacian_3d(ux, uy, uz, K1, K2) -> (fx, fy, fz)
  make_damping_3d(L, width, strength) -> gamma (L,L,L)
  run_fdtd_3d(force_fn, ux0, vx0, gamma, dt, n_steps, ...) -> dict
"""

import numpy as np


def scalar_laplacian_3d(ux, uy, uz, K1=1.0, K2=0.5):
    """Scalar Laplacian force on 3D simple cubic with NN + NNN.

    Each component independently:
      f_comp[i] = Σ_j K_{ij} (u_comp[j] - u_comp[i])

    NN: 6 bonds (±x, ±y, ±z), spring constant K1.
    NNN: 12 bonds ((±1,±1,0), (±1,0,±1), (0,±1,±1)), spring constant K2.

    Returns (fx, fy, fz).
    """
    # Process each component identically
    fx = _laplacian_component(ux, K1, K2)
    fy = _laplacian_component(uy, K1, K2)
    fz = _laplacian_component(uz, K1, K2)
    return fx, fy, fz


def _laplacian_component(u, K1, K2):
    """Scalar Laplacian for one displacement component on 3D simple cubic.

    Open boundary: edges have fewer neighbors (no wrap-around).
    Each bond is counted once as (src→dst). The diagonal z_K accumulates
    the total spring constant from all bonds touching each site.
    """
    f = np.zeros_like(u)
    z_K = np.zeros_like(u)

    # NN: ±x (axis 2), ±y (axis 1), ±z (axis 0)
    # Convention: u[iz, iy, ix]

    # NN along x
    f[:, :, :-1] += K1 * u[:, :, 1:]
    f[:, :, 1:]  += K1 * u[:, :, :-1]
    z_K[:, :, :-1] += K1
    z_K[:, :, 1:]  += K1

    # NN along y
    f[:, :-1, :] += K1 * u[:, 1:, :]
    f[:, 1:, :]  += K1 * u[:, :-1, :]
    z_K[:, :-1, :] += K1
    z_K[:, 1:, :]  += K1

    # NN along z
    f[:-1, :, :] += K1 * u[1:, :, :]
    f[1:, :, :]  += K1 * u[:-1, :, :]
    z_K[:-1, :, :] += K1
    z_K[1:, :, :]  += K1

    # NNN: 12 bonds in 3 pairs (xy, xz, yz)

    # xy pair: (±1, ±1, 0) — shifts in axes 2 and 1
    for (dy, dx) in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
        sy_src = slice(1, None) if dy > 0 else slice(None, -1)
        sy_dst = slice(None, -1) if dy > 0 else slice(1, None)
        sx_src = slice(1, None) if dx > 0 else slice(None, -1)
        sx_dst = slice(None, -1) if dx > 0 else slice(1, None)
        f[:, sy_dst, sx_dst] += K2 * u[:, sy_src, sx_src]
        z_K[:, sy_dst, sx_dst] += K2

    # xz pair: (±1, 0, ±1) — shifts in axes 2 and 0
    for (dz, dx) in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
        sz_src = slice(1, None) if dz > 0 else slice(None, -1)
        sz_dst = slice(None, -1) if dz > 0 else slice(1, None)
        sx_src = slice(1, None) if dx > 0 else slice(None, -1)
        sx_dst = slice(None, -1) if dx > 0 else slice(1, None)
        f[sz_dst, :, sx_dst] += K2 * u[sz_src, :, sx_src]
        z_K[sz_dst, :, sx_dst] += K2

    # yz pair: (0, ±1, ±1) — shifts in axes 1 and 0
    for (dz, dy) in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
        sz_src = slice(1, None) if dz > 0 else slice(None, -1)
        sz_dst = slice(None, -1) if dz > 0 else slice(1, None)
        sy_src = slice(1, None) if dy > 0 else slice(None, -1)
        sy_dst = slice(None, -1) if dy > 0 else slice(1, None)
        f[sz_dst, sy_dst, :] += K2 * u[sz_src, sy_src, :]
        z_K[sz_dst, sy_dst, :] += K2

    f -= z_K * u
    return f


def make_damping_3d(L, width, strength):
    """Spherical PML damping ramp for 3D cubic domain.

    Returns gamma array (L, L, L). Zero in interior, ramps quadratically
    to `strength` at edges.
    """
    c = (L - 1) / 2.0
    iz, iy, ix = np.mgrid[0:L, 0:L, 0:L].astype(float)
    r = np.sqrt((ix - c)**2 + (iy - c)**2 + (iz - c)**2)
    r_inner = L / 2.0 - width
    g = np.zeros((L, L, L))
    mask = r > r_inner
    g[mask] = strength * ((r[mask] - r_inner) / width) ** 2
    return g


def run_fdtd_3d(force_fn, ux0, vx0, gamma, dt, n_steps,
                rec_iz=None, rec_iy=None, rec_ix=None,
                rec_start=None, rec_n=None, m=1.0,
                uy0=None, uz0=None, vy0=None, vz0=None):
    """Velocity Verlet FDTD in 3D with optional site recording.

    Parameters
    ----------
    force_fn : callable (ux, uy, uz) -> (fx, fy, fz)
    ux0, vx0 : initial x-displacement and x-velocity (L, L, L)
    uy0, uz0 : optional initial y,z-displacement (default: zero)
    vy0, vz0 : optional initial y,z-velocity (default: zero)
    gamma : damping array (L, L, L)
    dt : time step
    n_steps : total steps
    rec_iz, rec_iy, rec_ix : integer arrays of recording site indices
    rec_start : step to begin recording (default: n_steps - rec_n)
    rec_n : number of steps to record (default: 100)
    m : mass per site

    Returns
    -------
    If no recording sites: (ux_final, uy_final, uz_final)
    If recording sites: dict with 'ux', 'uy', 'uz' arrays (rec_n, N_pts)

    Stability
    ---------
    Verlet requires dt < 2/ω_max. For K1=1, K2=0.5:
    ω²_max = 16 (at M=(π,π,0)), so dt_crit = 0.5.
    Recommended: dt ≤ 0.25 for safety margin.
    """
    # Assumes cubic grid (L×L×L)
    L = ux0.shape[0]
    ux = ux0.copy()
    uy = uy0.copy() if uy0 is not None else np.zeros((L, L, L))
    uz = uz0.copy() if uz0 is not None else np.zeros((L, L, L))
    vx = vx0.copy()
    vy = vy0.copy() if vy0 is not None else np.zeros((L, L, L))
    vz = vz0.copy() if vz0 is not None else np.zeros((L, L, L))

    ax, ay, az = force_fn(ux, uy, uz)
    ax /= m; ay /= m; az /= m

    recording = rec_iz is not None
    if recording:
        N_pts = len(rec_iz)
        if rec_n is None:
            rec_n = 100
        if rec_start is None:
            rec_start = max(0, n_steps - rec_n)
        assert n_steps >= rec_n, (
            f"n_steps={n_steps} < rec_n={rec_n}: not enough steps to fill recording buffer")
        assert rec_start + rec_n <= n_steps, (
            f"recording window [{rec_start}, {rec_start+rec_n}) extends beyond n_steps={n_steps}")
        out_ux = np.zeros((rec_n, N_pts))
        out_uy = np.zeros((rec_n, N_pts))
        out_uz = np.zeros((rec_n, N_pts))
        idx = 0

    for step in range(n_steps):
        # Verlet position update
        ux += dt * vx + 0.5 * dt**2 * ax
        uy += dt * vy + 0.5 * dt**2 * ay
        uz += dt * vz + 0.5 * dt**2 * az

        # New forces
        ax_n, ay_n, az_n = force_fn(ux, uy, uz)
        ax_n /= m; ay_n /= m; az_n /= m

        # Verlet velocity update
        vx += 0.5 * dt * (ax + ax_n)
        vy += 0.5 * dt * (ay + ay_n)
        vz += 0.5 * dt * (az + az_n)

        # PML damping: v *= 1/(1+γdt) ≈ exp(-γdt) to O(γdt).
        # Exact for small γdt; at γdt~0.1 error is ~0.5%.
        # More stable than (1-γdt) which goes negative at large γdt.
        d = 1.0 / (1.0 + gamma * dt)
        vx *= d; vy *= d; vz *= d

        ax, ay, az = ax_n, ay_n, az_n

        # Recording
        if recording and rec_start <= step < rec_start + rec_n:
            out_ux[idx] = ux[rec_iz, rec_iy, rec_ix]
            out_uy[idx] = uy[rec_iz, rec_iy, rec_ix]
            out_uz[idx] = uz[rec_iz, rec_iy, rec_ix]
            idx += 1

    if not recording:
        return ux, uy, uz

    return {'ux': out_ux, 'uy': out_uy, 'uz': out_uz}
