"""Null test: flat integrand is NN-specific, not universal across gaugings.

Source: old_tests/28_nnn_fine_scan.py, 31_flux_tube_AB.py, 32_ab_validation.py,
        39_zero_compute_analysis.py (F12)

Results:
  TestNNNNotFlat:
    NNN integrand CV = 29.0%  (threshold > 20%)
    NN integrand CV = 5.6%  (NN flatter than NNN)
  TestNNNHigherThanNN:
    σ_NNN > σ_NN at all k: 37.35 vs 28.33 (k=0.3) to 9.07 vs 2.60 (k=1.5)
  TestKappaNNvsNNN (k≤0.9):
    κ_NN: 0.022 (α=0.10) → 0.428 (α=0.30)
    κ_NNN: 0.087 (α=0.10) → 0.889 (α=0.30)
    α=0.30 k≤1.5: κ_NN=0.844 (<1), κ_NNN=2.227 (>1) — NNN overshoots
  TestABFails:
    AB integrand CV = 25.1%  (threshold > 20%)
    FDTD/AB amplitude at k=0.3: 6.5× (threshold > 3)
    FDTD/AB shape ratio CV = 31.9%  (threshold > 20%)
  TestNNPhaseArgument:
    NN z-bond dx=0: exp(ik·0) = 1 at all k  (exact)
    NNN dx=1: Re(exp(i·0.3)) = 0.955 → Re(exp(i·1.5)) = 0.071

All analytic (0 seconds).
"""
import numpy as np
import pytest
from helpers.config import k_vals_7
from helpers.stats import cv
from data.kappa_table import kappa_nn, kappa_nnn


# NNN FDTD data at α=0.30, R=5 from data/sigma_nnn.py (L=100)
from data.sigma_nnn import sigma_nnn
sigma_nnn_030 = sigma_nnn[0.30]

# NN FDTD data at α=0.30 from sigma_ring
from data.sigma_ring import sigma_ring


class TestNNNNotFlat:
    """NNN gauging does NOT produce flat integrand."""

    def test_nnn_integrand_cv_high(self):
        """NNN integrand sin²(k)·σ_NNN at α=0.30: CV > 20%."""
        integrand = np.sin(k_vals_7)**2 * sigma_nnn_030
        print(f"  NNN integrand CV = {cv(integrand):.1f}%")
        assert cv(integrand) > 20.0, \
            f"NNN integrand CV={cv(integrand):.1f}% < 20%"

    def test_nn_flatter_than_nnn(self):
        """NN integrand CV < NNN integrand CV at α=0.30."""
        integ_nn = np.sin(k_vals_7)**2 * sigma_ring[5]
        integ_nnn = np.sin(k_vals_7)**2 * sigma_nnn_030
        print(f"  NN integrand CV = {cv(integ_nn):.1f}% (NN flatter than NNN)")
        assert cv(integ_nn) < cv(integ_nnn), \
            f"NN CV={cv(integ_nn):.1f}% ≥ NNN CV={cv(integ_nnn):.1f}%"


class TestNNNHigherThanNN:
    """NNN gives consistently higher σ_tr than NN (more bonds gauged)."""

    def test_nnn_greater_than_nn(self):
        """σ_NNN > σ_NN at all k (α=0.30)."""
        print(f"  σ_NNN vs σ_NN: {sigma_nnn_030[0]:.2f} vs {sigma_ring[5][0]:.2f} (k=0.3) to {sigma_nnn_030[-1]:.2f} vs {sigma_ring[5][-1]:.2f} (k=1.5)")
        for i in range(len(k_vals_7)):
            assert sigma_nnn_030[i] > sigma_ring[5][i], \
                f"σ_NNN < σ_NN at k={k_vals_7[i]}"


class TestKappaNNvsNNN:
    """κ_NN and κ_NNN differ by factor 2-4."""

    def test_kappa_nnn_greater_than_nn(self):
        """κ_NNN > κ_NN at all α (more bonds = larger κ)."""
        for alpha in [0.10, 0.15, 0.20, 0.25, 0.30]:
            assert len(kappa_nn[alpha]) == 2, \
                f"kappa_nn[{alpha}] has {len(kappa_nn[alpha])} entries, expected 2"
            assert len(kappa_nnn[alpha]) == 2, \
                f"kappa_nnn[{alpha}] has {len(kappa_nnn[alpha])} entries, expected 2"
            k_nn_val = kappa_nn[alpha][1]   # k≤1.5
            k_nnn_val = kappa_nnn[alpha][1]  # k≤1.5
            print(f"  κ_NN={k_nn_val:.3f}, κ_NNN={k_nnn_val:.3f} at α={alpha}")
            assert k_nnn_val > k_nn_val, \
                f"κ_NNN={k_nnn_val} ≤ κ_NN={k_nn_val} at α={alpha}"

    def test_nnn_overshoots_at_030(self):
        """At α=0.30 k≤1.5: κ_NNN > 1 but κ_NN < 1.

        NNN overshoots (too many bonds gauged), while NN stays sub-diffusive.
        """
        k_nn_15 = kappa_nn[0.30][1]
        k_nnn_15 = kappa_nnn[0.30][1]
        print(f"  α=0.30 k≤1.5: κ_NN={k_nn_15:.3f} (<1), κ_NNN={k_nnn_15:.3f} (>1)")
        assert k_nn_15 < 1.0, f"κ_NN={k_nn_15} ≥ 1 at α=0.30"
        assert k_nnn_15 > 1.0, f"κ_NNN={k_nnn_15} ≤ 1 at α=0.30"

    def test_kappa_monotonic_in_alpha(self):
        """κ increases monotonically with α (both gaugings)."""
        alphas = [0.10, 0.15, 0.20, 0.25, 0.30]
        knn = [kappa_nn[a][0] for a in alphas]  # k≤0.9
        knnn = [kappa_nnn[a][0] for a in alphas]
        print(f"  κ_NN (k≤0.9): {knn[0]:.3f} (α=0.10) → {knn[-1]:.3f} (α=0.30)")
        print(f"  κ_NNN (k≤0.9): {knnn[0]:.3f} (α=0.10) → {knnn[-1]:.3f} (α=0.30)")
        for i in range(len(knn) - 1):
            assert knn[i + 1] > knn[i], \
                f"κ_NN not monotonic at α={alphas[i]}"
            assert knnn[i + 1] > knnn[i], \
                f"κ_NNN not monotonic at α={alphas[i]}"


class TestABFails:
    """Aharonov-Bohm σ=2sin²(πα)/k fails on three axes."""

    def test_ab_shape_wrong(self):
        """AB predicts σ ~ 1/k; integrand sin²(k)/k is NOT flat.

        σ_AB(k) = 2sin²(πα)/k → integrand sin²(k)·σ_AB = 2sin²(πα)·sin²(k)/k.
        CV > 20% (FDTD gives 7.4%).
        """
        alpha = 0.30
        sigma_ab = 2 * np.sin(np.pi * alpha)**2 / k_vals_7
        integrand_ab = np.sin(k_vals_7)**2 * sigma_ab
        print(f"  AB integrand CV = {cv(integrand_ab):.1f}%")
        assert cv(integrand_ab) > 20.0, \
            f"AB integrand CV={cv(integrand_ab):.1f}% < 20%"

    def test_ab_amplitude_wrong(self):
        """AB underestimates FDTD by > 3× at low k.

        σ_AB(k=0.3) ≈ 4.4, FDTD ≈ 28 → factor 6.5×.
        """
        alpha = 0.30
        sigma_ab = 2 * np.sin(np.pi * alpha)**2 / k_vals_7
        ratio = sigma_ring[5][0] / sigma_ab[0]
        print(f"  FDTD/AB at k=0.3: {ratio:.1f}×")
        assert ratio > 3.0, f"FDTD/AB = {ratio:.1f}×, expected > 3"

    def test_ab_vs_fdtd_shape_mismatch(self):
        """AB shape ∝ 1/k does not match FDTD shape at α=0.30."""
        sigma_fdtd = sigma_ring[5]
        ab_shape = 1 / k_vals_7
        # Normalize both at k=0.3
        fdtd_n = sigma_fdtd / sigma_fdtd[0]
        ab_n = ab_shape / ab_shape[0]
        ratio = fdtd_n / ab_n
        print(f"  FDTD/AB shape ratio CV = {cv(ratio):.1f}%")
        assert cv(ratio) > 20.0, \
            f"FDTD/AB shape ratio CV={cv(ratio):.1f}% < 20% — too close"


class TestNNPhaseArgument:
    """NN z-bonds have dx=0 → phase factor is k-independent.

    Documentary test — verifies the physical argument, not an implementation.
    (cf. TestStrainCouplingZero in test_5.)

    Source: old_tests/39_zero_compute_analysis.py (F12).

    NN z-bond at position (iy, ix): displacement d = (0, 0, 1).
    Phase factor exp(ik·d) for x-incident wave: exp(ik·0) = 1.
    Born vertex for this bond is k-independent → flat spectrum.

    NNN bonds with dx≠0 have exp(ik·dx) → k-dependent vertex.
    This explains WHY NN gauging is flat but NNN is not.
    """

    def test_nn_zbond_phase_is_one(self):
        """NN z-bond: exp(ik·dx) = exp(0) = 1 at all k (dx=0)."""
        dx_nn = 0  # z-bond displacement has no x-component
        print(f"  NN z-bond dx=0: exp(ik·0) = 1 at all k")
        for kv in k_vals_7:
            phase = np.exp(1j * kv * dx_nn)
            np.testing.assert_allclose(phase, 1.0, atol=1e-15)

    def test_nnn_bond_phase_varies(self):
        """NNN bond with dx=±1: exp(ik·dx) varies with k."""
        phase_arr = np.exp(1j * k_vals_7 * 1.0)  # dx = +1
        # Real part goes from cos(0.3)=0.955 to cos(1.5)=0.071
        print(f"  NNN dx=1: Re(exp(i·0.3)) = {np.real(phase_arr[0]):.3f} → Re(exp(i·1.5)) = {np.real(phase_arr[-1]):.3f}")
        assert np.real(phase_arr[-1]) < 0.1, \
            f"Re(exp(i·1.5)) = {np.real(phase_arr[-1]):.3f}, not small"
        assert np.real(phase_arr[0]) > 0.9, \
            f"Re(exp(i·0.3)) = {np.real(phase_arr[0]):.3f}, not near 1"

    def test_nn_vertex_k_independent(self):
        """NN Born vertex V·exp(ik·dx) = V·1 = V (k-independent).

        For z-bond with dx=0, the vertex seen by x-incident wave
        is V times the phase factor, which is just V (constant in k).
        """
        from helpers.born import V_eff
        V = V_eff(0.30)
        vertices = []
        for kv in k_vals_7:
            vertex = V * np.exp(1j * kv * 0)  # dx=0 for z-bond
            vertices.append(vertex)
        vertices = np.array(vertices)
        print(f"  NN Born vertex V·exp(ik·0) = V = {V:.6f} (k-independent)")
        # All should be identical
        np.testing.assert_allclose(vertices, V, atol=1e-15)
