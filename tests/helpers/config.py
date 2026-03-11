"""Standard parameters for all tests."""
import numpy as np

# Spring constants (K1 = 2*K2 for cubic isotropy to O(k^6))
K1 = 1.0
K2 = 0.5

# Lattice sound speed
c_lat = np.sqrt(K1 + 4 * K2)  # sqrt(3)

# Default holonomy
ALPHA_REF = 0.30

# Default loop radius
R_LOOP = 5

# FDTD box
L_DEFAULT = 80
DW = 15       # PML width
DS = 1.5      # PML damping
DT = 0.25     # time step
sx = 8.0      # packet width
r_m = 20      # measurement sphere radius

# Angular grid
N_THETA = 13
N_PHI = 24

# Wavenumber grids
k_vals_7 = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
k_vals_13 = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5,
                       1.7, 1.9, 2.1, 2.4, 2.7, 3.0])

# BZ regularization: lower k cutoff for kappa integral (avoids k→0 where v_g→c)
EPS_LAT = 0.005

# Peierls coupling at reference alpha
V_ref = K1 * (np.cos(2 * np.pi * ALPHA_REF) - 1)

# ── Self-consistency ─────────────────────────────────────────────
assert K1 == 2 * K2, f"Cubic isotropy requires K1=2K2, got {K1}, {K2}"
assert abs(c_lat**2 - (K1 + 4 * K2)) < 1e-15, \
    f"c_lat² = {c_lat**2}, expected K1+4K2 = {K1 + 4 * K2}"
assert len(k_vals_7) == 7
assert len(k_vals_13) == 13
assert k_vals_7[0] == 0.3 and k_vals_7[-1] == 1.5
assert k_vals_13[-1] <= np.pi, \
    f"k={k_vals_13[-1]} exceeds BZ edge π={np.pi:.4f}"
