"""Lattice dispersion, group velocity, BZ grid, self-energy."""
import numpy as np
from .config import K1, K2, c_lat, EPS_LAT


def dispersion_sq(k):
    """omega^2(k) along cubic axis.

    Explicit NN + NNN form: = (2K1 + 8K2)(1-cos k) = 4c²sin²(k/2).
    """
    return 2 * K1 * (1 - np.cos(k)) + 4 * K2 * (2 - 2 * np.cos(k))


def group_velocity(k):
    """v_g = c_lat * cos(k/2)."""
    return c_lat * np.cos(k / 2)


def k_eff(k):
    """Continuum effective wavenumber: k_eff = omega/c = 2*sin(k/2)."""
    return 2 * np.sin(k / 2)


def omega_k2_grid(N_bz=64):
    """3D BZ grid of omega^2(kx,ky,kz) for self-energy computation."""
    kx_1d = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
    ck = np.cos(kx_1d)
    CX, CY, CZ = np.meshgrid(ck, ck, ck, indexing='ij')
    return (2 * K1 * (3 - CX - CY - CZ)
            + 4 * K2 * (3 - CX * CY - CY * CZ - CX * CZ))


# Cached BZ grid (computed once)
_omega_k2_cache = None


def get_omega_k2():
    """Return cached BZ grid."""
    global _omega_k2_cache
    if _omega_k2_cache is None:
        _omega_k2_cache = omega_k2_grid()
    return _omega_k2_cache


def G_00(k, eps=EPS_LAT):
    """Self-energy from BZ sum: G_00 = <1/(omega^2 - omega_k^2 + i*eps)>."""
    omega2 = dispersion_sq(k)
    return np.mean(1.0 / (omega2 - get_omega_k2() + 1j * eps))


# ── Self-consistency ─────────────────────────────────────────────
assert abs(dispersion_sq(0.0)) < 1e-15, "ω²(0) must be 0"
assert abs(group_velocity(0.0) - c_lat) < 1e-15, \
    f"v_g(0) = {group_velocity(0.0)}, expected c_lat = {c_lat}"
assert abs(k_eff(0.0)) < 1e-15, "k_eff(0) must be 0"
# ω² = 4c²sin²(k/2) at all k
_k_test = np.array([0.3, 0.9, 1.5, np.pi])
np.testing.assert_allclose(
    dispersion_sq(_k_test), 4 * c_lat**2 * np.sin(_k_test / 2)**2,
    rtol=1e-14, err_msg="ω² ≠ 4c²sin²(k/2)")
# G_00: Im < 0 (retarded causal Green function at eps > 0)
_G00_test = G_00(1.0)
assert np.imag(_G00_test) < 0, \
    f"Im(G_00(k=1)) = {np.imag(_G00_test):.4e}, expected < 0 (causality)"
