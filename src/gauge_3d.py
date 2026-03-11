"""
gauge_3d.py — Peierls vortex ring on 3D simple cubic lattice.

Toroidal Z2 vortex defect via Dirac disk: a flat disk in the xy-plane
bounded by the vortex ring, with Peierls rotation R(2*pi*alpha) on
bonds crossing the disk.

Option B: R(2*pi*alpha) acts on (ux, uy) only; uz unaffected.
Matches the 2D code (gauge_coupling.py) where R acts on (ux, uy).

Disk geometry: centered at (cx, cy, cz) = (L//2, L//2, L//2).

NN gauging (default): z-bonds from (cz-1, iy, ix) to (cz, iy, ix)
where (ix-cx)^2 + (iy-cy)^2 <= R_loop^2. Weight K1.

NNN gauging (gauge_nnn=True): additionally gauges NNN bonds with
dz=1 that cross the disk. These are xz bonds (dx=±1, dy=0) and
yz bonds (dx=0, dy=±1). Crossing test at bond midpoint. Weight K2.

On each gauged bond (lo = cz-1, hi = cz side):
  site lo gets: K * (R - I) @ u_hi    (R = rotation by 2*pi*alpha)
  site hi gets: K * (R^T - I) @ u_lo

At alpha=0: reduces to scalar_laplacian_3d exactly.

NNN gauging test (α=0.3, R=5, L=80, 3 k-pts):

    k     σ_NN    σ_NNN    Δ%
   0.30   40.74   54.26   +33%
   0.70    7.69   18.08  +135%
   1.30    3.14   10.79  +244%

  NNN gauging increases σ_tr by 1.3× to 3.4× (grows with k).
  Effect is massive — not a small correction.
  NN disk: 81 bonds (K1=1.0). NNN disk: 312 bonds (K2=0.5).

API:
  make_vortex_force(alpha, R_loop, L, K1, K2, gauge_nnn=False) -> force_fn
  precompute_disk_bonds(L, R_loop) -> (iy_disk, ix_disk)
  precompute_nnn_disk_bonds(L, R_loop) -> list of (iy_lo, ix_lo, iy_hi, ix_hi)
"""

import numpy as np
from elastic_3d import scalar_laplacian_3d


def precompute_disk_bonds(L, R_loop):
    """Indices of lattice sites whose NN z-bond crosses the Dirac disk.

    Returns (iy_disk, ix_disk): 1D integer arrays.
    Bond (cz-1, iy, ix) <-> (cz, iy, ix) is gauged for each pair.
    """
    cx = cy = L // 2
    iy_all, ix_all = np.mgrid[0:L, 0:L]
    mask = (ix_all - cx)**2 + (iy_all - cy)**2 <= R_loop**2
    iy_disk, ix_disk = np.where(mask)
    return iy_disk, ix_disk


def precompute_nnn_disk_bonds(L, R_loop):
    """NNN bonds crossing Dirac disk (z = cz-0.5).

    NNN bonds with dz=1 that cross the disk:
    - xz: (cz-1, iy, ix) <-> (cz, iy, ix+dx),  dx = ±1
    - yz: (cz-1, iy, ix) <-> (cz, iy+dy, ix),  dy = ±1

    Bond midpoint at (ix + dx/2, iy + dy/2) must be inside disk.

    Returns list of 4 tuples (iy_lo, ix_lo, iy_hi, ix_hi).
    Each tuple defines a set of NNN bonds of one type.
    """
    cx = cy = L // 2
    iy_all, ix_all = np.mgrid[0:L, 0:L]

    groups = []
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        iy_hi_all = iy_all + dy
        ix_hi_all = ix_all + dx

        in_bounds = ((iy_hi_all >= 0) & (iy_hi_all < L) &
                     (ix_hi_all >= 0) & (ix_hi_all < L))

        x_cross = ix_all + dx / 2.0
        y_cross = iy_all + dy / 2.0
        in_disk = (x_cross - cx)**2 + (y_cross - cy)**2 <= R_loop**2

        mask = in_bounds & in_disk
        groups.append((iy_all[mask], ix_all[mask],
                        iy_hi_all[mask], ix_hi_all[mask]))

    return groups


def make_vortex_force(alpha, R_loop, L, K1=1.0, K2=0.5, cz=None,
                      gauge_nnn=False):
    """Create force function for 3D lattice with vortex ring defect.

    Precomputes disk geometry and Peierls rotation matrix.
    Returned force_fn is ready for use with run_fdtd_3d.

    Parameters
    ----------
    alpha : float
        Vortex strength. Peierls phase = 2*pi*alpha. alpha=0.5 gives R(pi)=-I.
    R_loop : float
        Vortex ring radius (lattice units).
    L : int
        Domain size (L x L x L).
    K1, K2 : float
        NN and NNN spring constants.
    cz : int or None
        Z-layer of Dirac disk. Default: L//2.
    gauge_nnn : bool
        If True, also gauge NNN bonds crossing the disk (K2 weight).

    Returns
    -------
    force_fn : callable (ux, uy, uz) -> (fx, fy, fz)
    """
    if cz is None:
        cz = L // 2
    assert 1 <= cz < L, f"cz={cz} out of range [1, L): iz_lo=cz-1 must be valid"
    iy_disk, ix_disk = precompute_disk_bonds(L, R_loop)

    nnn_groups = precompute_nnn_disk_bonds(L, R_loop) if gauge_nnn else []

    phi = 2.0 * np.pi * (float(alpha) % 1.0)
    cm1 = np.cos(phi) - 1.0
    s_phi = np.sin(phi)
    # Snap to exact zero for half-integer alpha: sin(n*pi) is not exactly
    # 0 in IEEE 754 (e.g. sin(pi) ~ 1.2e-16). Without snap, alpha=0.5
    # gives a parasitic off-diagonal force ~1e-16 instead of exact R(pi)=-I.
    if abs(s_phi) < 1e-12:
        s_phi = 0.0
    if abs(cm1) < 1e-12:
        cm1 = 0.0
    iz_lo = cz - 1
    iz_hi = cz

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if cm1 == 0.0 and s_phi == 0.0:
            return fx, fy, fz

        # ── NN z-bonds ──
        if len(iy_disk) > 0:
            ux_hi = ux[iz_hi, iy_disk, ix_disk]
            uy_hi = uy[iz_hi, iy_disk, ix_disk]
            ux_lo = ux[iz_lo, iy_disk, ix_disk]
            uy_lo = uy[iz_lo, iy_disk, ix_disk]

            # Lower site gets K1*(R - I)*u_upper
            fx[iz_lo, iy_disk, ix_disk] += K1 * (cm1 * ux_hi - s_phi * uy_hi)
            fy[iz_lo, iy_disk, ix_disk] += K1 * (s_phi * ux_hi + cm1 * uy_hi)

            # Upper site gets K1*(R^T - I)*u_lower
            fx[iz_hi, iy_disk, ix_disk] += K1 * (cm1 * ux_lo + s_phi * uy_lo)
            fy[iz_hi, iy_disk, ix_disk] += K1 * (-s_phi * ux_lo + cm1 * uy_lo)

        # ── NNN bonds crossing disk ──
        for iy_lo_n, ix_lo_n, iy_hi_n, ix_hi_n in nnn_groups:
            if len(iy_lo_n) == 0:
                continue

            ux_hi_n = ux[iz_hi, iy_hi_n, ix_hi_n]
            uy_hi_n = uy[iz_hi, iy_hi_n, ix_hi_n]
            ux_lo_n = ux[iz_lo, iy_lo_n, ix_lo_n]
            uy_lo_n = uy[iz_lo, iy_lo_n, ix_lo_n]

            # Lower site gets K2*(R - I)*u_upper
            fx[iz_lo, iy_lo_n, ix_lo_n] += K2 * (cm1 * ux_hi_n - s_phi * uy_hi_n)
            fy[iz_lo, iy_lo_n, ix_lo_n] += K2 * (s_phi * ux_hi_n + cm1 * uy_hi_n)

            # Upper site gets K2*(R^T - I)*u_lower
            fx[iz_hi, iy_hi_n, ix_hi_n] += K2 * (cm1 * ux_lo_n + s_phi * uy_lo_n)
            fy[iz_hi, iy_hi_n, ix_hi_n] += K2 * (-s_phi * ux_lo_n + cm1 * uy_lo_n)

        return fx, fy, fz

    return force_fn


# --- Self-tests (run at import, ~milliseconds) ---

def _self_test_alpha_zero():
    """At alpha=0, gauge force = scalar Laplacian (no Peierls correction)."""
    L, R_loop = 10, 3
    rng = np.random.RandomState(0)
    ux = rng.randn(L, L, L)
    uy = rng.randn(L, L, L)
    uz = rng.randn(L, L, L)
    f_gauge = make_vortex_force(0.0, R_loop, L)
    fx_g, fy_g, fz_g = f_gauge(ux, uy, uz)
    fx_s, fy_s, fz_s = scalar_laplacian_3d(ux, uy, uz)
    assert np.max(np.abs(fx_g - fx_s)) < 1e-14, "alpha=0 gauge != scalar"
    assert np.max(np.abs(fy_g - fy_s)) < 1e-14, "alpha=0 gauge != scalar"
    assert np.max(np.abs(fz_g - fz_s)) < 1e-14, "alpha=0 gauge != scalar"


def _self_test_energy_conservation():
    """Verlet with gauge force (alpha>0, no PML) conserves energy.

    Gauge force derives from Hermitian Hamiltonian: u_lo^T K R u_hi + h.c.
    Energy = KE + PE = (1/2) Σ v² - (1/2) <u, F(u)>.
    Verlet is symplectic → energy oscillates at O(dt²), no drift.
    """
    L, k0, sx, dt = 16, 0.5, 3.0, 0.02
    alpha, R_loop = 0.30, 3
    force_fn = make_vortex_force(alpha, R_loop, L)

    ix = np.arange(L, dtype=float)
    env = np.exp(-((ix - L/2)**2) / (2 * sx**2))
    c = np.sqrt(1.0 + 4 * 0.5)
    omega_k = 2 * np.sin(k0 / 2) * c
    ux = np.broadcast_to(env * np.cos(k0 * ix), (L, L, L)).copy()
    vx = np.broadcast_to(omega_k * env * np.sin(k0 * ix), (L, L, L)).copy()
    uy = np.zeros((L, L, L)); uz = np.zeros((L, L, L))
    vy = np.zeros((L, L, L)); vz = np.zeros((L, L, L))

    def energy(ux, uy, uz, vx, vy, vz):
        ke = 0.5 * np.sum(vx**2 + vy**2 + vz**2)
        fx, fy, fz = force_fn(ux, uy, uz)
        pe = -0.5 * np.sum(ux*fx + uy*fy + uz*fz)
        return ke + pe

    e0 = energy(ux, uy, uz, vx, vy, vz)
    ax, ay, az = force_fn(ux, uy, uz)
    # 20 steps, wave stays near center (total travel ~ 1.7 sites)
    for _ in range(20):
        ux += dt * vx + 0.5 * dt**2 * ax
        uy += dt * vy + 0.5 * dt**2 * ay
        uz += dt * vz + 0.5 * dt**2 * az
        ax_n, ay_n, az_n = force_fn(ux, uy, uz)
        vx += 0.5 * dt * (ax + ax_n)
        vy += 0.5 * dt * (ay + ay_n)
        vz += 0.5 * dt * (az + az_n)
        ax, ay, az = ax_n, ay_n, az_n
    e1 = energy(ux, uy, uz, vx, vy, vz)
    drift = abs(e1 - e0) / abs(e0)
    assert drift < 1e-4, f"energy drift {drift:.2e} at alpha={alpha}"


_self_test_alpha_zero()
_self_test_energy_conservation()
