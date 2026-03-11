"""Born-level scattering: per-bond coupling, monopole/dipole, form factors."""
import numpy as np
from .config import K1, K2, c_lat, ALPHA_REF, V_ref
from .lattice import group_velocity


def peierls_coupling(alpha):
    """Peierls coupling parameters for holonomy R(2*pi*alpha).
    Returns cm1 = cos(2*pi*alpha)-1, s_phi = sin(2*pi*alpha).
    """
    phi = 2 * np.pi * alpha
    cm1 = np.cos(phi) - 1
    s_phi = np.sin(phi)
    return cm1, s_phi


def V_eff(alpha):
    """Monopole coupling K1*cm1 = K1*(cos(2πα)-1). NOT the full V in sigma_bond_born."""
    return K1 * (np.cos(2 * np.pi * alpha) - 1)


def Z_monopole(k):
    """Monopole angular integral Z_mono = 8π(1 + sin(k)/k).
    np.sinc(k/π) = sin(k)/k. Same forces at both sites of z-bond, x-incident."""
    return 8 * np.pi * (1 + np.sinc(k / np.pi))


def Z_dipole(k):
    """Dipole angular integral Z_dipo = 8π(1 - sin(k)/k).
    np.sinc(k/π) = sin(k)/k. Opposite forces at both sites of z-bond, x-incident."""
    return 8 * np.pi * (1 - np.sinc(k / np.pi))


def sigma_bond_born(k, alpha):
    """Per-bond Born cross-section: sigma = C0 * V(alpha) / v_g^2.

    V(alpha) = cm1^2 * Z_mono + s_phi^2 * Z_dipo
    C0 = K1^2 / (16 * pi^2 * c_lat^4)
    """
    cm1, s_phi = peierls_coupling(alpha)
    V = cm1**2 * Z_monopole(k) + s_phi**2 * Z_dipole(k)
    C0 = K1**2 / (16 * np.pi**2 * c_lat**4)
    vg = group_velocity(k)
    return C0 * V / vg**2


# ── Self-consistency ─────────────────────────────────────────────
# Z_mono + Z_dipo = 16π at any k
_k_check = np.array([0.3, 0.9, 1.5])
np.testing.assert_allclose(
    Z_monopole(_k_check) + Z_dipole(_k_check), 16 * np.pi,
    rtol=1e-14, err_msg="Z_mono + Z_dipo ≠ 16π")
# V_eff(ALPHA_REF) = V_ref from config
assert abs(V_eff(ALPHA_REF) - V_ref) < 1e-15, \
    f"V_eff({ALPHA_REF}) = {V_eff(ALPHA_REF)}, V_ref = {V_ref}"
# |cm1| = |s_phi| at α = 0.25 exactly
_cm1_025, _s_025 = peierls_coupling(0.25)
assert abs(abs(_cm1_025) - abs(_s_025)) < 1e-14, \
    f"|cm1|={abs(_cm1_025)}, |s_phi|={abs(_s_025)} at α=0.25"
# peierls_coupling(0) = (0, 0)
_cm1_0, _s_0 = peierls_coupling(0.0)
assert abs(_cm1_0) < 1e-15 and abs(_s_0) < 1e-15, "peierls_coupling(0) ≠ (0,0)"
# sigma_bond_born > 0 at any alpha > 0, k > 0
assert sigma_bond_born(1.0, ALPHA_REF) > 0, "sigma_bond must be positive"
