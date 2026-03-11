"""§6.2 Born exponent -5/2 for filled disk.

Source: old_tests/60_born_exponent.py

Results:
  TestBornExponent:
    Angular grid converged: N=50 vs N=100 diff < 0.01 at R=3,5
    Coherent limit (discrete): N_eff(k=0.01) / N_bonds² = 0.998
    Coherent limit (continuum): N_eff(k=0.01) / (πR²)² = 0.999
    R=3: p = -2.223, R=5: p = -2.415, R=7: p = -2.371, R=9: p = -2.405
    (all in [-2.55, -2.15])
  TestBornExponentLargeR:
    R=200 (kR>15): p = -2.4878  (within 1% of -2.5)
  TestTransportDecomposition (R=10, continuum Airy):
    p_total (no transport weight) = -1.384  (≈ -3/2)
    p_transport = -2.440  (≈ -5/2)
    transport contribution = -1.056  (≈ -1)
  TestUniversality (R=5):
    lattice disk:  p = -2.415
    random disk:   p = -1.880  (ratio 0.78)
    continuum:     p = -2.432  (diff -0.018)

Analytic computations (seconds). Angular integrals + Airy form factor.
"""
import numpy as np
from scipy.special import j1
import pytest
from helpers.config import K1, K2, c_lat, k_vals_7, ALPHA_REF
from helpers.geometry import disk_bonds, random_disk
from helpers.stats import log_log_slope, cv
from helpers.lattice import k_eff


# ── Angular grid (shared) ────────────────────────────────────────

N_TH_TEST, N_PH_TEST = 100, 100


@pytest.fixture(scope="module")
def angular_grid():
    """Angular grid for Born N_eff computation."""
    thetas = np.linspace(0, np.pi, N_TH_TEST)
    phis = np.linspace(0, 2 * np.pi, N_PH_TEST, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    sin_th = np.sin(TH)
    cos_th = np.cos(TH)
    transport = 1 - sin_th * np.cos(PH)
    return TH, PH, sin_th, cos_th, transport


def _neff_from_bonds(dx, dy, k_arr, TH, PH, sin_th, cos_th, transport):
    """N_eff from Born for arbitrary bond positions (dx, dy).

    F2 = 4*cos²(qz/2) is the z-polarization factor for a z-bond scatterer.
    It appears in both numerator and denominator; since |f|² varies with angle,
    F2 does not exactly cancel (~1% effect on N_eff, <0.01 on exponent).
    Kept for physical correctness.
    """
    neff = np.zeros(len(k_arr))
    for ik, kv in enumerate(k_arr):
        b = np.exp(1j * kv * dx)
        phase_out = kv * (
            sin_th[:, :, None] * np.cos(PH)[:, :, None] * dx[None, None, :]
            + sin_th[:, :, None] * np.sin(PH)[:, :, None] * dy[None, None, :])
        a = np.exp(-1j * phase_out)
        f = np.sum(a * b[None, None, :], axis=2)
        qz = kv * cos_th
        F2 = 4 * np.cos(qz / 2)**2
        w = F2 * transport * sin_th
        neff[ik] = np.sum(np.abs(f)**2 * w) / np.sum(w)
    return neff


def neff_born_discrete(R, k_arr, TH, PH, sin_th, cos_th, transport):
    """N_eff from Born (discrete lattice disk). V cancels in ratio."""
    dx, dy = disk_bonds(R)
    return _neff_from_bonds(dx, dy, k_arr, TH, PH, sin_th, cos_th, transport)


def neff_continuum(R, k_arr, TH, PH, sin_th, cos_th, transport,
                   use_transport=True):
    """N_eff using continuum Airy disk form factor."""
    neff = np.zeros(len(k_arr))
    for ik, kv in enumerate(k_arr):
        # Momentum transfer transverse to incident (+x) direction
        qx = kv * (sin_th * np.cos(PH) - 1)
        qy = kv * sin_th * np.sin(PH)
        q_tr = np.sqrt(qx**2 + qy**2)
        F = np.where(q_tr > 1e-10,
                     2 * np.pi * R * j1(q_tr * R) / q_tr,
                     np.pi * R**2)
        qz = kv * cos_th
        F2_pol = 4 * np.cos(qz / 2)**2
        if use_transport:
            w = F2_pol * transport * sin_th
        else:
            w = F2_pol * sin_th
        neff[ik] = np.sum(np.abs(F)**2 * w) / np.sum(w)
    return neff


def neff_born_random(R, N, k_arr, TH, PH, sin_th, cos_th, transport,
                     seed=42):
    """N_eff from Born with random points inside disk."""
    dx, dy = random_disk(R, N, seed=seed)
    return _neff_from_bonds(dx, dy, k_arr, TH, PH, sin_th, cos_th, transport)


# ── Tests ─────────────────────────────────────────────────────────

class TestBornExponent:
    """Born N_eff exponent converges to -5/2."""

    def test_angular_grid_converged(self, angular_grid):
        """Exponent stable at N_TH=N_PH=100 vs 50 (diff < 0.01)."""
        thetas_50 = np.linspace(0, np.pi, 50)
        phis_50 = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        TH50, PH50 = np.meshgrid(thetas_50, phis_50, indexing='ij')
        sin50 = np.sin(TH50)
        cos50 = np.cos(TH50)
        tr50 = 1 - sin50 * np.cos(PH50)
        TH, PH, sin_th, cos_th, transport = angular_grid
        for R in [3, 5]:
            ne_50 = neff_born_discrete(R, k_vals_7, TH50, PH50, sin50, cos50, tr50)
            p_50 = log_log_slope(k_vals_7, ne_50)[0]
            ne_100 = neff_born_discrete(R, k_vals_7, TH, PH, sin_th, cos_th,
                                        transport)
            p_100 = log_log_slope(k_vals_7, ne_100)[0]
            print(f"  R={R}: N=50 p={p_50:.4f}, N=100 p={p_100:.4f}, "
                  f"diff={abs(p_100-p_50):.4f}")
            assert abs(p_100 - p_50) < 0.01, \
                f"Angular grid not converged at R={R}: diff={abs(p_100-p_50):.4f}"

    def test_neff_coherent_limit(self, angular_grid):
        """At k→0, all bonds scatter in phase: N_eff → N_bonds².

        Exact at k=0: b_j=1, phase_out=0, so f=N at all angles.
        |f|²=N² factors out of sum(|f|²·w)/sum(w) = N².
        At k=0.01 (kR=0.05): O(kR)² deviation gives ratio ≈ 0.998.
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        ne = neff_born_discrete(5, np.array([0.01]), TH, PH, sin_th, cos_th,
                                transport)
        N_bonds = len(disk_bonds(5)[0])
        ratio = ne[0] / N_bonds**2
        print(f"  k=0.01: N_eff={ne[0]:.0f}, N²={N_bonds**2}, ratio={ratio:.4f}")
        assert abs(ratio - 1.0) < 0.02, \
            f"N_eff/N² = {ratio:.4f}, expected ≈ 1 at k→0"

    def test_neff_continuum_coherent_limit(self, angular_grid):
        """Continuum Airy disk: N_eff → (πR²)² at k→0.

        At k=0: F(q=0) = πR² for all angles → |F|² = (πR²)² factors out.
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        ne = neff_continuum(5, np.array([0.01]), TH, PH, sin_th, cos_th,
                            transport)
        expected = (np.pi * 25)**2  # (πR²)²
        ratio = ne[0] / expected
        print(f"  k=0.01: N_eff={ne[0]:.0f}, (πR²)²={expected:.0f}, ratio={ratio:.4f}")
        assert abs(ratio - 1.0) < 0.02, \
            f"N_eff/(πR²)² = {ratio:.4f}, expected ≈ 1 at k→0"

    def test_born_exponent_r5(self, angular_grid):
        """Born exponent at R=5: p ∈ [-2.55, -2.30]."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        ne = neff_born_discrete(5, k_vals_7, TH, PH, sin_th, cos_th,
                                transport)
        p = log_log_slope(k_vals_7, ne)[0]
        print(f"  R=5: p = {p:.3f}")
        assert -2.55 < p < -2.30, f"p_Born(R=5) = {p:.3f} out of range"

    def test_born_exponent_in_range_all_R(self, angular_grid):
        """Born exponent ∈ [-2.55, -2.15] at R=3,5,7,9.

        R=3 is small (29 bonds, kR≈1-4) → exponent shallower than -2.5.
        Convergence to -5/2 requires kR >> 1 (verified at R=200).
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        for R in [3, 5, 7, 9]:
            ne = neff_born_discrete(R, k_vals_7, TH, PH, sin_th, cos_th,
                                    transport)
            p = log_log_slope(k_vals_7, ne)[0]
            print(f"  R={R}: p = {p:.3f}")
            assert -2.55 < p < -2.15, \
                f"p_Born(R={R}) = {p:.3f} out of range"


class TestBornExponentLargeR:
    """Large-R convergence using (delta,eta) grid at R=200."""

    def test_born_exponent_r200(self):
        """Born exponent at R=200, kR>15: p within 1% of -2.5.

        Uses forward-cone coordinates (delta, eta) for high angular resolution
        near the forward direction where the Airy disk concentrates:
          delta = polar angle from forward axis (x), range [-π/2, π/2]
          eta = azimuthal angle around forward axis, range [-π, π]
          cos(θ_s) = cos(delta)·cos(eta), jacobian = cos(delta)
        """
        R = 200
        N_d, N_e = 1000, 1000
        delta = np.linspace(-np.pi / 2, np.pi / 2, N_d)
        eta = np.linspace(-np.pi, np.pi, N_e, endpoint=False)
        D, E = np.meshgrid(delta, eta, indexing='ij')
        cos_d = np.cos(D)
        cos_e = np.cos(E)
        # cos(θ_s) = cos(delta)*cos(eta) for scattering angle θ_s
        tr = 1 - cos_d * cos_e
        jac = cos_d  # d(solid angle) = cos(delta) d(delta) d(eta)
        dd = np.pi / N_d
        de = 2 * np.pi / N_e
        W = np.sum(tr * jac) * dd * de

        k_dense = np.logspace(np.log10(0.01), np.log10(0.15), 20)
        kR = k_dense * R

        neff_tr = np.zeros(len(k_dense))
        for ik, kv in enumerate(k_dense):
            q_tr2 = kv**2 * (1 + cos_d**2 - 2 * cos_d * cos_e)
            q_tr = np.sqrt(q_tr2)
            qR = q_tr * R
            F = np.where(qR > 1e-10,
                         2 * np.pi * R * j1(qR) / qR,
                         np.pi * R**2)
            neff_tr[ik] = np.sum(np.abs(F)**2 * tr * jac) * dd * de / W

        mask = kR > 15
        p = np.polyfit(np.log(k_dense[mask]), np.log(neff_tr[mask]), 1)[0]
        print(f"  R=200 (kR>15): p = {p:.4f}")
        assert abs(p - (-2.5)) / 2.5 < 0.01, \
            f"p_Born(R=200, kR>15) = {p:.4f}, > 1% from -2.5"


class TestTransportDecomposition:
    """σ_tr = σ_total × transport_weight, exponents: -3/2 + (-1) = -5/2."""

    def test_total_exponent_minus_3_2(self, angular_grid):
        """σ_total (no transport weight) exponent ≈ -3/2."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        ne_tot = neff_continuum(10, k_vals_7, TH, PH, sin_th, cos_th,
                                transport, use_transport=False)
        p = log_log_slope(k_vals_7, ne_tot)[0]
        print(f"  p_total (no transport weight) = {p:.3f}")
        assert -1.55 < p < -1.25, f"p_total = {p:.3f}, expected ≈ -1.5"

    def test_transport_adds_minus_1(self, angular_grid):
        """Transport weight adds ≈ -1 to exponent."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        ne_tot = neff_continuum(10, k_vals_7, TH, PH, sin_th, cos_th,
                                transport, use_transport=False)
        ne_tr = neff_continuum(10, k_vals_7, TH, PH, sin_th, cos_th,
                               transport, use_transport=True)
        p_tot = log_log_slope(k_vals_7, ne_tot)[0]
        p_tr = log_log_slope(k_vals_7, ne_tr)[0]
        diff = p_tr - p_tot
        print(f"  p_transport = {p_tr:.3f}, contribution = {diff:.3f}")
        assert -1.15 < diff < -0.90, \
            f"Transport contribution = {diff:.3f}, expected ≈ -1.0"


class TestUniversality:
    """Born exponent is geometry-dependent, not lattice-specific."""

    def test_random_disk_similar_exponent(self, angular_grid):
        """Random disk gives similar Born exponent to lattice disk."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        R = 5
        ne_lat = neff_born_discrete(R, k_vals_7, TH, PH, sin_th, cos_th,
                                    transport)
        ne_rand = neff_born_random(R, 81, k_vals_7, TH, PH, sin_th,
                                   cos_th, transport, seed=42)
        p_lat = log_log_slope(k_vals_7, ne_lat)[0]
        p_rand = log_log_slope(k_vals_7, ne_rand)[0]
        ratio = p_rand / p_lat
        print(f"  lattice: p = {p_lat:.3f}, random: p = {p_rand:.3f}, ratio = {ratio:.2f}")
        assert 0.7 < ratio < 1.3, \
            f"Random/lattice exponent ratio = {ratio:.2f}"

    def test_continuum_matches_discrete(self, angular_grid):
        """Continuum Airy disk ≈ discrete lattice at R ≥ 5."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        ne_c = neff_continuum(5, k_vals_7, TH, PH, sin_th, cos_th,
                              transport)
        ne_d = neff_born_discrete(5, k_vals_7, TH, PH, sin_th, cos_th,
                                  transport)
        p_c = log_log_slope(k_vals_7, ne_c)[0]
        p_d = log_log_slope(k_vals_7, ne_d)[0]
        print(f"  continuum: p = {p_c:.3f}, discrete: p = {p_d:.3f}, diff = {p_c - p_d:.3f}")
        assert abs(p_c - p_d) < 0.1, \
            f"Continuum/discrete diff = {p_c - p_d:.3f}"

    def test_square_disk_similar_exponent(self, angular_grid):
        """Square disk gives similar Born exponent (geometry ≈ universal)."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        R = 5
        # Square disk: all points with |x|≤R and |y|≤R
        coords = [(x, y) for y in range(-R, R + 1) for x in range(-R, R + 1)]
        dx_sq = np.array([c[0] for c in coords], dtype=float)
        dy_sq = np.array([c[1] for c in coords], dtype=float)

        neff_sq = _neff_from_bonds(dx_sq, dy_sq, k_vals_7, TH, PH, sin_th,
                                   cos_th, transport)
        ne_lat = neff_born_discrete(R, k_vals_7, TH, PH, sin_th, cos_th,
                                    transport)
        p_sq = log_log_slope(k_vals_7, neff_sq)[0]
        p_round = log_log_slope(k_vals_7, ne_lat)[0]
        print(f"  square: p = {p_sq:.3f}, round: p = {p_round:.3f}")
        assert abs(p_sq - p_round) < 0.15, \
            f"Square/round exponent diff = {p_sq - p_round:.3f}"
