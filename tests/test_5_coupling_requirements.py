"""Coupling requirements: flat integrand requires displacement coupling + strong α.

Source: old_tests/37_displacement_coupling.py, 39_zero_compute_analysis.py (F16)

Results:
  TestBornNotFlat:
    Born integrand CV = 68.7%  (threshold > 30%, ~19x variation over k-range)
  TestWeakCouplingNotFlat:
    V(k) CV at α=0.05 = 54.1%  (threshold > 25%)
    Integrand CV at α=0.10 = 16.4%  (threshold > 15%)
  TestAlphaThresholdRequired:
    V(k) CV: 28.2% (α=0.10), 4.6% (α=0.20), 0.0% (α=0.25), 2.7% (α=0.30), 5.2% (α=0.40)
    CV decreases to α=0.25, stays < 6% above; integrand CV: 5.6% (α=0.30) < 16.4% (α=0.10)
    α=0.50 limit: R=-I, s_phi=0, cm1=-2 (pure monopole); integrand CV=12.8% < 16.4% (α=0.10)
  TestStrainCouplingZero (documentary):
    Δu = u(z_hi) - u(z_lo) = 0 for plane wave exp(ikx) on z-bond  (exact, by construction)
    strain force dK·Δu = 0 for any dK  (exact)
    displacement force K1·cm1·u ≠ 0  (contrast, K_eff from peierls_coupling)
  TestPolarizationIndependence:
    (R-I)(R^T-I) = 2(1-cos(2πα))·I  max diff < 1e-16  (exact)
    2(1-cos(2π·0.30)) = 2.618034
    uz decoupled: (R-I)·ẑ = 0 at all α → N_pol = 2

All analytic (0 seconds).
"""
import numpy as np
import pytest
from helpers.config import K1, k_vals_7, ALPHA_REF
from helpers.born import peierls_coupling, Z_monopole, Z_dipole
from helpers.lattice import group_velocity
from helpers.stats import cv
from data.sigma_ring import sigma_alpha_nn


class TestBornNotFlat:
    """Born-level integrand is NOT flat — needs multiple scattering."""

    def test_born_integrand_cv_high(self):
        """Born σ_bond ~ V/v_g². Integrand sin²(k)·(V/v_g²) varies ~19x over k-range.

        This is why MS correction is necessary: Born alone gives CV ≈ 69%.
        """
        cm1, s_phi = peierls_coupling(0.30)
        vg = group_velocity(k_vals_7)
        V = cm1**2 * Z_monopole(k_vals_7) + s_phi**2 * Z_dipole(k_vals_7)
        born = V / vg**2
        integrand = np.sin(k_vals_7)**2 * born
        integrand_n = integrand / integrand[0]
        print(f"  Born integrand CV = {cv(integrand_n):.1f}%")
        assert cv(integrand_n) > 30.0, \
            f"Born integrand CV={cv(integrand_n):.1f}% < 30%"


class TestWeakCouplingNotFlat:
    """Weak coupling (small α) does NOT produce flat integrand."""

    def test_alpha_005_not_flat(self):
        """α=0.05: V(k) varies strongly → integrand not flat.

        No FDTD data at α=0.05 in σ_alpha_nn. Test V(k) variation instead.
        """
        cm1, s_phi = peierls_coupling(0.05)
        V = cm1**2 * Z_monopole(k_vals_7) + s_phi**2 * Z_dipole(k_vals_7)
        print(f"  V(k) CV at α=0.05 = {cv(V):.1f}%")
        assert cv(V) > 25.0, \
            f"V(k) at α=0.05: CV={cv(V):.1f}% < 25%"

    def test_alpha_010_not_flat(self):
        """α=0.10: integrand sin²(k)·σ_ring has CV > 15%."""
        sigma = sigma_alpha_nn[0.10]
        integrand = np.sin(k_vals_7)**2 * sigma
        print(f"  Integrand CV at α=0.10 = {cv(integrand):.1f}%")
        assert cv(integrand) > 15.0, \
            f"Integrand CV at α=0.10 = {cv(integrand):.1f}% < 15%"


class TestAlphaThresholdRequired:
    """Flat integrand requires α ≥ 0.20 (monopole dominance)."""

    def test_cv_decreases_with_alpha(self):
        """V(k) CV decreases to α=0.25 and stays small above."""
        alphas = [0.10, 0.20, 0.25, 0.30, 0.40]
        cvs = []
        for alpha in alphas:
            cm1, s_phi = peierls_coupling(alpha)
            V = cm1**2 * Z_monopole(k_vals_7) + s_phi**2 * Z_dipole(k_vals_7)
            cvs.append(cv(V))
        print(f"  V(k) CV: {cvs[0]:.1f}% (α=0.10), {cvs[1]:.1f}% (α=0.20), "
              f"{cvs[2]:.1f}% (α=0.25), {cvs[3]:.1f}% (α=0.30), {cvs[4]:.1f}% (α=0.40)")
        # CV decreases from α=0.10 to α=0.25
        assert cvs[0] > cvs[1] > cvs[2], \
            f"CV not decreasing: {cvs[:3]}"
        # V ≈ const for α ≥ 0.25 (CV stays small)
        for i in range(2, len(cvs)):
            assert cvs[i] < 6.0, \
                f"V not flat at α={alphas[i]}: CV={cvs[i]:.1f}%"

    def test_alpha_050_limit(self):
        """α=0.50: R=-I, s_phi=0, pure monopole. Integrand flatter than α=0.10."""
        cm1, s_phi = peierls_coupling(0.50)
        assert abs(s_phi) < 1e-12, f"s_phi={s_phi} should be 0 at α=0.50"
        assert abs(cm1 - (-2.0)) < 1e-12, f"cm1={cm1} should be -2 at α=0.50"
        integ_050 = np.sin(k_vals_7)**2 * sigma_alpha_nn[0.50]
        integ_010 = np.sin(k_vals_7)**2 * sigma_alpha_nn[0.10]
        print(f"  α=0.50 integrand CV = {cv(integ_050):.1f}% vs α=0.10 = {cv(integ_010):.1f}%")
        assert cv(integ_050) < cv(integ_010), \
            f"α=0.50 not flatter than α=0.10"

    def test_alpha_030_flatter_than_010(self):
        """Integrand at α=0.30 is flatter than at α=0.10."""
        integ_030 = np.sin(k_vals_7)**2 * sigma_alpha_nn[0.30]
        integ_010 = np.sin(k_vals_7)**2 * sigma_alpha_nn[0.10]
        print(f"  Integrand CV: {cv(integ_030):.1f}% (α=0.30) < {cv(integ_010):.1f}% (α=0.10)")
        assert cv(integ_030) < cv(integ_010), \
            f"α=0.30 CV={cv(integ_030):.1f}% ≥ α=0.10 CV={cv(integ_010):.1f}%"


class TestStrainCouplingZero:
    """Strain coupling on z-bonds gives zero for x-propagating wave.

    Documentary test — verifies the physical argument, not an implementation.
    u_lo = u_hi = exp(ik·ix) by construction (same ix), so Δu = 0 is a
    mathematical tautology. The physics is in the SETUP: z-bond endpoints
    share the same x-coordinate, so an x-propagating wave has no z-variation.

    Source: old_tests/37_displacement_coupling.py.
    At z-bond (iz_lo, iy, ix) ↔ (iz_hi, iy, ix):
      u(iz_lo) = u(iz_hi) = exp(ik·ix)
      → Δu = u_hi - u_lo = 0
      → strain force = dK·Δu = 0 for any dK.
    This is a GEOMETRIC NULL, independent of coupling strength.
    """

    def test_plane_wave_delta_u_zero(self):
        """Plane wave exp(ikx): Δu = u(z_hi) - u(z_lo) = 0 on z-bond."""
        print(f"  Δu = u(z_hi) - u(z_lo) = 0 for plane wave exp(ikx) on z-bond")
        for kv in k_vals_7:
            # Any (iy, ix) position — wave depends only on ix
            for ix in range(-5, 6):
                u_lo = np.exp(1j * kv * ix)
                u_hi = np.exp(1j * kv * ix)  # same ix, different iz
                delta_u = u_hi - u_lo
                assert abs(delta_u) < 1e-15, \
                    f"|Δu| = {abs(delta_u):.2e} at k={kv}, ix={ix}"

    def test_strain_force_zero_any_dk(self):
        """Strain force dK·(u_hi - u_lo) = 0 for any coupling dK."""
        print(f"  strain force dK·Δu = 0 for any dK")
        for dK in [-1.3, -0.5, 0.5, 1.0, 1.3]:
            for kv in [0.3, 0.9, 1.5]:
                ix = 3  # arbitrary
                delta_u = np.exp(1j * kv * ix) - np.exp(1j * kv * ix)
                force = dK * delta_u
                assert abs(force) < 1e-15, \
                    f"strain force = {abs(force):.2e} at dK={dK}, k={kv}"

    def test_displacement_coupling_nonzero(self):
        """Displacement coupling K_eff·u_neighbor ≠ 0 (contrast with strain).

        For same plane wave, displacement force = K_eff·u_hi = K_eff·exp(ikx) ≠ 0.
        This confirms displacement coupling IS active while strain is not.
        """
        cm1, _ = peierls_coupling(ALPHA_REF)
        K_eff = K1 * cm1  # = -1.309 at alpha=0.30
        print(f"  displacement force K_eff·u ≠ 0")
        for kv in [0.3, 0.9, 1.5]:
            ix = 3
            u_hi = np.exp(1j * kv * ix)
            force = K_eff * u_hi
            assert abs(force) > 1.0, \
                f"|displacement force| = {abs(force):.2e} too small"


class TestPolarizationIndependence:
    """Scattering is polarization-independent by Peierls matrix algebra.

    Source: old_tests/39_zero_compute_analysis.py (F16).

    (R-I)(R^T-I) = 2(1-cos(2πα))·I  (scalar times identity).
    This means |(R-I)·u|² = 2(1-cos(2πα))·|u|² for ANY polarization u.
    Consequence: σ_tr does not depend on incident polarization.
    """

    def test_peierls_product_is_scalar(self):
        """(R-I)(R^T-I) = 2(1-cos(2πα))·I for several α values."""
        for alpha in [0.10, 0.25, 0.30, 0.45]:
            phi = 2 * np.pi * alpha
            R = np.array([[np.cos(phi), -np.sin(phi)],
                          [np.sin(phi), np.cos(phi)]])
            I2 = np.eye(2)
            product = (R - I2) @ (R.T - I2)
            expected = 2 * (1 - np.cos(phi)) * I2
            diff = np.max(np.abs(product - expected))
            np.testing.assert_allclose(product, expected, atol=1e-14,
                                       err_msg=f"Failed at α={alpha}")
        print(f"  (R-I)(R^T-I) = 2(1-cos(2πα))·I  max diff < 1e-16")
        print(f"  2(1-cos(2π·0.30)) = {2*(1-np.cos(2*np.pi*0.30)):.6f}")

    def test_scattering_amplitude_polarization_independent(self):
        """|(R-I)·u|² = 2(1-cos)·|u|² for physical and random polarizations."""
        alpha = 0.30
        phi = 2 * np.pi * alpha
        R = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), np.cos(phi)]])
        I2 = np.eye(2)
        dR = R - I2
        expected_factor = 2 * (1 - np.cos(phi))
        # Physical polarizations: linear x, linear y, diagonal, circular
        special = [
            np.array([1, 0], dtype=complex),
            np.array([0, 1], dtype=complex),
            np.array([1, 1], dtype=complex) / np.sqrt(2),
            np.array([1, 1j], dtype=complex) / np.sqrt(2),
        ]
        rng = np.random.RandomState(42)
        random_u = [rng.randn(2) + 1j * rng.randn(2) for _ in range(6)]
        for u in special + random_u:
            amp_sq = np.sum(np.abs(dR @ u)**2)
            u_sq = np.sum(np.abs(u)**2)
            np.testing.assert_allclose(amp_sq, expected_factor * u_sq,
                                       rtol=1e-12)
        print(f"  |(R-I)·u|² = {expected_factor:.6f}·|u|² for all u")

    def test_uz_decoupled(self):
        """R acts only on (ux,uy); uz is unaffected → N_pol = 2."""
        for alpha in [0.10, 0.30, 0.50]:
            phi = 2 * np.pi * alpha
            R_full = np.eye(3)
            R_full[:2, :2] = [[np.cos(phi), -np.sin(phi)],
                              [np.sin(phi), np.cos(phi)]]
            dR = R_full - np.eye(3)
            u_z = np.array([0.0, 0.0, 1.0])
            np.testing.assert_allclose(dR @ u_z, 0, atol=1e-15,
                                       err_msg=f"uz scatters at α={alpha}")
        print(f"  (R-I)·ẑ = 0 at all α → uz decoupled")
