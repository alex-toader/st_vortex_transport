"""Bond geometries: disk, ring, random disk, annulus, line."""
import numpy as np


def disk_bonds(R):
    """Lattice points (dx, dy) inside disk of radius R. Returns (N,) arrays."""
    dx, dy = [], []
    for ddy in range(-R, R + 1):
        for ddx in range(-R, R + 1):
            if ddx**2 + ddy**2 <= R**2:
                dx.append(ddx)
                dy.append(ddy)
    return np.array(dx, dtype=float), np.array(dy, dtype=float)


def ring_bonds(R):
    """Perimeter-only bonds: r >= R-0.5 (outer half of last layer, not symmetric band)."""
    dx_d, dy_d = disk_bonds(R)
    r2 = dx_d**2 + dy_d**2
    mask = r2 >= (R - 0.5)**2
    return dx_d[mask], dy_d[mask]


def annulus_bonds(R, width=1):
    """Annular region: R-width < r <= R."""
    dx_d, dy_d = disk_bonds(R)
    r2 = dx_d**2 + dy_d**2
    mask = r2 > (R - width)**2
    return dx_d[mask], dy_d[mask]


def random_disk(R, N, seed=42):
    """N uniformly random points inside disk of radius R."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = R * np.sqrt(rng.uniform(0, 1, N))
    return radii * np.cos(angles), radii * np.sin(angles)


def line_bonds(N, axis='x'):
    """N bonds along a line centered at origin. Use odd N for exact centering."""
    half = N // 2
    coords = np.arange(-half, -half + N, dtype=float)
    zeros = np.zeros(N)
    if axis == 'x':
        return coords, zeros
    return zeros, coords


# ── Self-consistency ─────────────────────────────────────────────
_dx0, _dy0 = disk_bonds(0)
assert len(_dx0) == 1 and _dx0[0] == 0 and _dy0[0] == 0, \
    "disk_bonds(0) must be single point at origin"
_dx1, _dy1 = disk_bonds(1)
assert len(_dx1) == 5, f"disk_bonds(1) should have 5 points, got {len(_dx1)}"
# line_bonds symmetry
_lx, _ly = line_bonds(11, 'x')
assert len(_lx) == 11 and np.sum(_ly) == 0, "line_bonds(11) broken"
assert _lx[0] == -5 and _lx[-1] == 5, "line_bonds(11) not centered"
# ring_bonds is subset of disk_bonds
_dr, _ = ring_bonds(5)
_dd, _ = disk_bonds(5)
assert len(_dr) < len(_dd), "ring must have fewer points than disk"
