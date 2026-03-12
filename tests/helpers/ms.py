"""Multiple scattering: T=(I-VG)^{-1}V, N_eff, eigenvalues."""
import numpy as np
from .config import K1, K2, c_lat, EPS_LAT
from .lattice import dispersion_sq, k_eff, get_omega_k2
from .born import V_eff


def build_G_matrix(dx, dy, k, eps=EPS_LAT):
    """Build inter-bond Green function matrix G(r_i, r_j).

    Off-diagonal: G_ij = exp(i*k_eff*r_ij) / (4*pi*c^2*r_ij)
    Diagonal: G_00 from lattice BZ sum.
    """
    N = len(dx)
    ke = k_eff(k)
    dist = np.sqrt((dx[:, None] - dx[None, :])**2
                   + (dy[:, None] - dy[None, :])**2)
    G = np.zeros((N, N), dtype=complex)
    m = dist > 0
    G[m] = np.exp(1j * ke * dist[m]) / (4 * np.pi * c_lat**2 * dist[m])

    omega2 = dispersion_sq(k)
    G_00 = np.mean(1.0 / (omega2 - get_omega_k2() + 1j * eps))
    np.fill_diagonal(G, G_00)
    return G


def build_VG(dx, dy, k, alpha=0.30, eps=EPS_LAT):
    """Build V*G matrix for given bond positions and wavenumber."""
    V = V_eff(alpha)
    G = build_G_matrix(dx, dy, k, eps)
    return V * G


def T_matrix(dx, dy, k, alpha=0.30):
    """Full T-matrix: T = (I - V*G)^{-1} * V."""
    N = len(dx)
    V = V_eff(alpha)
    G = build_G_matrix(dx, dy, k)
    return np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))


def make_dOmega(thetas, n_phi):
    """Solid angle per (theta, phi) cell. Sum over all cells = 4π."""
    n_theta = len(thetas)
    dO = np.zeros(n_theta)
    for it in range(n_theta):
        if it == 0:
            dO[it] = 2 * np.pi * (1 - np.cos((thetas[0] + thetas[1]) / 2))
        elif it == n_theta - 1:
            dO[it] = 2 * np.pi * (np.cos((thetas[-2] + thetas[-1]) / 2) + 1)
        else:
            dO[it] = 2 * np.pi * (np.cos((thetas[it - 1] + thetas[it]) / 2)
                                   - np.cos((thetas[it] + thetas[it + 1]) / 2))
    return dO / n_phi


def _angular_sigma_tr(Tb, dx, dy, ke, thetas, phis, dOmega):
    """Vectorized angular integration of (1-cos_s)|f|² for sigma_tr."""
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    KX = ke * np.sin(TH) * np.cos(PH)
    KY = ke * np.sin(TH) * np.sin(PH)
    # Outgoing phase: k_eff from continuum Green function far field
    phase = np.exp(-1j * (KX[:, :, None] * dx + KY[:, :, None] * dy))
    F = np.einsum('ijk,k->ij', phase, Tb)
    cos_s = np.sin(TH) * np.cos(PH)
    return np.sum((1 - cos_s) * np.abs(F)**2 * dOmega[:, None]) / (4 * np.pi * c_lat**2)


def sigma_tr_ms(dx, dy, k_arr, alpha=0.30, n_theta=25, n_phi=48):
    """Compute sigma_tr from MS T-matrix at each k in k_arr.

    Uses Lippmann-Schwinger: f(k_out) = sum_j exp(-i*k_out*r_j) * (T*b)_j
    with b_j = exp(i*k_in*r_j) for +x incidence.

    Approximation: incident phase uses lattice k (exact eigenmode),
    outgoing phase uses k_eff = 2*sin(k/2) (from continuum Green function).
    Error in form factor phases: O(k³/24), ~9% at k=1.5.
    Empirically validated: MS captures 75-110% of FDTD correction (L=100 data).
    """
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dOmega = make_dOmega(thetas, n_phi)

    sigma = np.zeros(len(k_arr))
    for ik, kv in enumerate(k_arr):
        ke = 2 * np.sin(kv / 2)
        T = T_matrix(dx, dy, kv, alpha)
        # Incident phase: lattice wavenumber k (exact eigenmode)
        b = np.exp(1j * kv * dx)
        sigma[ik] = _angular_sigma_tr(T @ b, dx, dy, ke, thetas, phis, dOmega)
    return sigma


def sigma_tr_born_ms(dx, dy, k_arr, alpha=0.30, n_theta=25, n_phi=48):
    """Compute both Born and MS sigma_tr at each k.

    Returns (sigma_born, sigma_ms) arrays. Same angular integration,
    Born uses V*b, MS uses T*b.
    """
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dOmega = make_dOmega(thetas, n_phi)
    V = V_eff(alpha)

    sigma_born = np.zeros(len(k_arr))
    sigma_ms = np.zeros(len(k_arr))
    for ik, kv in enumerate(k_arr):
        ke = 2 * np.sin(kv / 2)
        T = T_matrix(dx, dy, kv, alpha)  # needed for MS; Born uses V*b directly
        b = np.exp(1j * kv * dx)
        sigma_ms[ik] = _angular_sigma_tr(T @ b, dx, dy, ke, thetas, phis, dOmega)
        sigma_born[ik] = _angular_sigma_tr(V * b, dx, dy, ke, thetas, phis, dOmega)
    return sigma_born, sigma_ms


def eigenvalues_VG(dx, dy, k, alpha=0.30):
    """Eigenvalues of V*G matrix."""
    VG = build_VG(dx, dy, k, alpha)
    return np.linalg.eigvals(VG)


# ── Self-consistency ─────────────────────────────────────────────
# G matrix is symmetric: G[i,j] = G[j,i] (isotropic Green function)
_dx_test = np.array([0.0, 1.0, 0.0, -1.0])
_dy_test = np.array([0.0, 0.0, 1.0, 0.0])
_G_test = build_G_matrix(_dx_test, _dy_test, 0.9)
assert np.max(np.abs(_G_test - _G_test.T)) < 1e-14, \
    "G matrix not symmetric"
# Im(G_00) < 0: retarded Green function (eps > 0 → pole below real axis)
assert _G_test[0, 0].imag < 0, \
    f"Im(G_00) = {_G_test[0,0].imag:.4e}, expected < 0 (causality)"
# dOmega solid angle conservation: sum over all cells = 4π
_thetas_check = np.linspace(0, np.pi, 25)
_dO_check = make_dOmega(_thetas_check, 48)
assert abs(np.sum(_dO_check) * 48 - 4*np.pi) < 1e-10, \
    f"dOmega sum = {np.sum(_dO_check)*48:.6f}, expected 4π = {4*np.pi:.6f}"
# sigma_tr_ms > 0 at any alpha > 0
_sigma_test = sigma_tr_ms(_dx_test, _dy_test, np.array([0.9]), alpha=0.30,
                          n_theta=7, n_phi=12)
assert _sigma_test[0] > 0, f"sigma_tr_ms = {_sigma_test[0]}, expected > 0"
# sigma_tr_born_ms MS component must match sigma_tr_ms exactly
_sb, _sm = sigma_tr_born_ms(_dx_test, _dy_test, np.array([0.9]), alpha=0.30,
                            n_theta=7, n_phi=12)
assert abs(_sm - _sigma_test[0]) < 1e-12, \
    f"sigma_tr_born_ms MS={_sm} != sigma_tr_ms={_sigma_test[0]}"
