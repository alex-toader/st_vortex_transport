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

    Construction: f[i] = Σ_j K_ij * u[j],  z_K[i] = Σ_j K_ij.
    Net force = f - z_K * u = Σ_j K_ij (u[j] - u[i]).

    Each NEIGHBOR DIRECTION is visited once per site. For NN along x,
    the two lines (f[i] += K1*u[i+1]) and (f[i+1] += K1*u[i]) handle
    opposite neighbors of different sites, not the same bond twice.
    A bulk site has 6 NN + 12 NNN neighbors; z_K = 6*K1 + 12*K2.

    Dispersion: omega^2(k) = 2*K1*Σ(1-cos k_i) + 4*K2*Σ(1-cos k_i cos k_j).
    For wave along x: omega^2 = (2*K1 + 8*K2)(1-cos k).
    Speed: c^2 = K1 + 4*K2 = 3.0 (K1=1, K2=0.5).
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

    # NNN: 12 neighbor directions in 3 planes (xy, xz, yz), 4 per plane.
    # Each direction visits a distinct neighbor: (1,1) and (-1,-1) are
    # opposite neighbors of the SAME site, not the same bond twice.

    # xy plane: (±1, ±1, 0) — 4 neighbor directions in axes 2 and 1
    for (dy, dx) in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
        sy_src = slice(1, None) if dy > 0 else slice(None, -1)
        sy_dst = slice(None, -1) if dy > 0 else slice(1, None)
        sx_src = slice(1, None) if dx > 0 else slice(None, -1)
        sx_dst = slice(None, -1) if dx > 0 else slice(1, None)
        f[:, sy_dst, sx_dst] += K2 * u[:, sy_src, sx_src]
        z_K[:, sy_dst, sx_dst] += K2

    # xz plane: (±1, 0, ±1) — 4 neighbor directions in axes 2 and 0
    for (dz, dx) in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
        sz_src = slice(1, None) if dz > 0 else slice(None, -1)
        sz_dst = slice(None, -1) if dz > 0 else slice(1, None)
        sx_src = slice(1, None) if dx > 0 else slice(None, -1)
        sx_dst = slice(None, -1) if dx > 0 else slice(1, None)
        f[sz_dst, :, sx_dst] += K2 * u[sz_src, :, sx_src]
        z_K[sz_dst, :, sx_dst] += K2

    # yz plane: (0, ±1, ±1) — 4 neighbor directions in axes 1 and 0
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

    Spherical geometry: γ(r) ramps from 0 at r_inner to `strength` at edge.
    Cube corners (r = L√3/2) are deeper in PML than face centers (r = L/2),
    so diagonal propagation is absorbed faster. This is acceptable because
    the incident wave is a plane wave along x — scattering is measured on
    a sphere well inside r_inner. Convergence verified: L=100 vs L=120
    within 3.2% at all k (tests/convergence_scan.py).

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
    If no recording sites: tuple (ux_final, uy_final, uz_final).
    If recording sites: dict with 'ux', 'uy', 'uz' arrays (rec_n, N_pts).
    All scattering measurements use recording mode (dict return).

    Stability
    ---------
    Verlet requires dt < 2/ω_max. For K1=1, K2=0.5:
    ω²_max = 16 (at M=(π,π,0)), so dt_crit = 0.5.
    Recommended: dt ≤ 0.25 for safety margin.

    PML damping is applied after the velocity update: v *= 1/(1+γdt).
    This is first-order in γdt (error ~0.5% at γdt~0.1). Acceptable
    because the PML region is not part of the measurement domain.
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


# --- Self-tests (run at import, ~microseconds) ---

def _self_test_coordination():
    """Bulk site must have z_K = 6*K1 + 12*K2."""
    L, K1, K2 = 7, 1.0, 0.5
    u = np.zeros((L, L, L))
    f = _laplacian_component(u, K1, K2)
    # f = 0 for u=0, but we need z_K. Use u=1 everywhere:
    # f[i] = sum_j K_ij * (1 - 1) = 0. Instead, use delta function.
    u_delta = np.zeros((L, L, L))
    u_delta[L//2, L//2, L//2] = 1.0
    f_delta = _laplacian_component(u_delta, K1, K2)
    # f_delta at center = -z_K * 1 (neighbors contribute to other sites)
    z_K_bulk = -f_delta[L//2, L//2, L//2]
    expected = 6 * K1 + 12 * K2
    assert abs(z_K_bulk - expected) < 1e-10, \
        f"z_K bulk = {z_K_bulk}, expected {expected}"

def _self_test_dispersion():
    """Plane wave dispersion: c² = K1 + 4*K2 = 3.0."""
    K1, K2 = 1.0, 0.5
    L = 32
    k_test = 0.5
    ix = np.arange(L, dtype=float)
    u = np.zeros((L, L, L))
    u[:, :, :] = np.cos(k_test * ix)[np.newaxis, np.newaxis, :]
    f = _laplacian_component(u, K1, K2)
    # At bulk center: f/u = -omega^2, omega^2 = (2K1+8K2)(1-cos k)
    mid = L // 2
    ratio = f[mid, mid, mid] / u[mid, mid, mid]
    omega2_expected = (2*K1 + 8*K2) * (1 - np.cos(k_test))
    assert abs(ratio + omega2_expected) < 1e-10, \
        f"dispersion: f/u = {ratio}, expected -{omega2_expected}"

_self_test_coordination()
_self_test_dispersion()
