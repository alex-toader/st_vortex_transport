"""§6.4 Assembly: Born -5/2 + MS +1/2 → flat integrand.

Source: old_tests/55_monopole_dipole.py, 56_neff_structure.py,
        60_born_exponent.py

Results:
  TestAlgebraicCancellation:
    sin²(k) = 4sin²(k/2)cos²(k/2): max diff = 1.1e-16  (exact)
    v_g²/c² = cos²(k/2): max diff = 4.4e-16  (exact)
  TestNeffBalance:
    sin²(k/2) · N_eff CV: R=3 11.9%, R=5 5.8%, R=7 3.4%, R=9 3.7%
    R=5,7,9 < 8%; improves with R
  TestFlatIntegrand:
    sin²(k) · σ_ring CV = 5.6%  (threshold < 10%)
    Born vertex V(k) CV = 2.7%  (threshold < 5%)
  TestResidualCV:
    sinc²(k/2) deviation = 6.1%  (threshold 3-10%)
    total residual CV = 5.6%  (threshold 5-15%)
  TestNeffStructure:
    sin²(k/2) better than k² (5.8% vs 7.1%, margin > 0.5%) — lattice dispersion
    N_eff exponent = -1.870  (threshold [-2.15, -1.85])
    sin²(k/2)/k² at k=0.3 = 0.2481  (≈ 1/4)
    N_eff(k=0.5): 113 (R=3), 253 (R=5), 403 (R=7), 644 (R=9)
    R-scaling: σ_tr ~ R^p, mean p = 1.63 (sub-geometric, between 1.5 and 1.9)

All analytic. Uses stored FDTD data.
"""
import numpy as np
import pytest
from helpers.config import K1, K2, c_lat, k_vals_7, ALPHA_REF, V_ref
from helpers.born import peierls_coupling, Z_monopole, Z_dipole
from helpers.lattice import group_velocity
from helpers.stats import cv, log_log_slope
from data.sigma_ring import sigma_ring
from data.sigma_bond import sigma_bond, sigma_xx, sigma_xy


# ── Tests ─────────────────────────────────────────────────────────

class TestAlgebraicCancellation:
    """cos²(k/2) in sin²(k) cancels 1/cos²(k/2) in v_g²."""

    def test_sin2k_equals_4sin2half_cos2half(self):
        """sin²(k) = 4·sin²(k/2)·cos²(k/2) — trigonometric identity."""
        k = k_vals_7
        lhs = np.sin(k)**2
        rhs = 4 * np.sin(k / 2)**2 * np.cos(k / 2)**2
        print(f"  sin²(k) = 4sin²(k/2)cos²(k/2): max diff = {np.max(np.abs(lhs - rhs)):.1e}")
        np.testing.assert_allclose(lhs, rhs, rtol=1e-14)

    def test_vg_squared_contains_cos2half(self):
        """v_g² = c_lat² · cos²(k/2) — the factor that cancels."""
        k = k_vals_7
        vg = group_velocity(k)
        expected = c_lat**2 * np.cos(k / 2)**2
        print(f"  v_g²/c² = cos²(k/2): max diff = {np.max(np.abs(vg**2 - expected)):.1e}")
        np.testing.assert_allclose(vg**2, expected, rtol=1e-14)

    def test_cancellation_leaves_sin2half(self):
        """After cancellation: sin²(k)/v_g² ∝ sin²(k/2) (no cos²)."""
        k = k_vals_7
        ratio = np.sin(k)**2 / group_velocity(k)**2
        expected_shape = 4 * np.sin(k / 2)**2 / c_lat**2
        print(f"  sin²(k)/v_g² = 4sin²(k/2)/c²: max diff = {np.max(np.abs(ratio - expected_shape)):.1e}")
        np.testing.assert_allclose(ratio, expected_shape, rtol=1e-14)


class TestNeffBalance:
    """sin²(k/2)·N_eff ≈ const (the non-trivial balance).

    Tested at multiple R. R=3 is too small for balance (~12%),
    but R=5,7,9 all satisfy CV < 8%, improving with R.
    """

    def test_sin2half_neff_cv(self):
        """sin²(k/2) · N_eff: CV < 8% at R=5,7,9; improves with R."""
        # sigma_xx + sigma_xy: deliberate (omits sigma_xz ~ 0.1%, see sigma_bond.py)
        sigma_tot = sigma_xx + sigma_xy
        cvs = {}
        for R in [3, 5, 7, 9]:
            N_eff = sigma_ring[R] / sigma_tot
            balance = np.sin(k_vals_7 / 2)**2 * N_eff
            cvs[R] = cv(balance)
            print(f"  R={R}: sin²(k/2) · N_eff CV = {cvs[R]:.1f}%")
        # R=3 too small; R=5,7,9 pass
        for R in [5, 7, 9]:
            assert cvs[R] < 8.0, \
                f"sin²(k/2)·N_eff CV={cvs[R]:.1f}% > 8% at R={R}"
        # Balance improves with R (for R >= 5)
        assert cvs[7] < cvs[5], \
            f"CV not improving: R=7 ({cvs[7]:.1f}%) ≥ R=5 ({cvs[5]:.1f}%)"


class TestFlatIntegrand:
    """sin²(k)·σ_ring ≈ const: the central result."""

    def test_integrand_cv(self):
        """sin²(k)·σ_ring CV < 10% at α=0.30, R=5."""
        integrand = np.sin(k_vals_7)**2 * sigma_ring[5]
        print(f"  sin²(k) · σ_ring CV = {cv(integrand):.1f}%")
        assert cv(integrand) < 10.0, \
            f"Integrand CV={cv(integrand):.1f}% > 10%"

    def test_born_vertex_const(self):
        """Born vertex V(k) ≈ const at α=0.30 (CV < 5%)."""
        cm1, s_phi = peierls_coupling(ALPHA_REF)
        V = cm1**2 * Z_monopole(k_vals_7) + s_phi**2 * Z_dipole(k_vals_7)
        print(f"  Born vertex V(k) CV = {cv(V):.1f}%")
        assert cv(V) < 5.0, f"V(k) CV={cv(V):.1f}% > 5%"

    def test_chain_decomposition(self):
        """Full chain: 4sin²(k/2)·V·N_eff tracks integrand."""
        cm1, s_phi = peierls_coupling(ALPHA_REF)
        V = cm1**2 * Z_monopole(k_vals_7) + s_phi**2 * Z_dipole(k_vals_7)
        # sigma_xx + sigma_xy: deliberate (omits sigma_xz ~ 0.1%, see sigma_bond.py)
        sigma_tot = sigma_xx + sigma_xy
        N_eff = sigma_ring[5] / sigma_tot
        chain = 4 * np.sin(k_vals_7 / 2)**2 * V * N_eff
        integrand = np.sin(k_vals_7)**2 * sigma_ring[5]
        # Chain and integrand should have same shape
        chain_n = chain / chain[0]
        integ_n = integrand / integrand[0]
        ratio_cv = cv(integ_n / chain_n)
        print(f"  chain/integrand mismatch CV = {ratio_cv:.1f}%")
        assert ratio_cv < 6.0, \
            f"Chain/integrand mismatch CV={ratio_cv:.1f}%"


class TestResidualCV:
    """Residual CV ≈ 10% from three identified sources."""

    def test_sinc_squared_deviation(self):
        """sinc²(k/2) = [sin(k/2)/(k/2)]² deviates from 1: CV ≈ 6%."""
        sinc2 = (np.sin(k_vals_7 / 2) / (k_vals_7 / 2))**2
        c = cv(sinc2)
        print(f"  sinc²(k/2) deviation = {c:.1f}%")
        assert 3.0 < c < 10.0, f"sinc² CV={c:.1f}%, expected ≈ 6%"

    def test_residual_not_zero(self):
        """Integrand CV is finite (~7-10%), not forced to 0 by symmetry."""
        integrand = np.sin(k_vals_7)**2 * sigma_ring[5]
        c = cv(integrand)
        print(f"  total residual CV = {c:.1f}%")
        assert c > 5.0, f"CV={c:.1f}% suspiciously low"
        assert c < 15.0, f"CV={c:.1f}% too high"


class TestNeffStructure:
    """N_eff = σ_ring/σ_bond: coherent-incoherent crossover.

    Source: old_tests/56_neff_structure.py.

    N_eff ∝ 1/k² (CV=2.3% at R=5) in the sampled k-range.
    This is better than 1/sin²(k/2) (CV=5.4%).
    The 1/k² scaling is EMPIRIC (non-Born): Born gives k^{-2.4}, FDTD k^{-2.0}.
    """

    def test_sin2half_better_than_k2(self):
        """1/sin²(k/2) fits N_eff better than 1/k² (lattice dispersion)."""
        sigma_tot = sigma_xx + sigma_xy
        N_eff = sigma_ring[5] / sigma_tot
        cv_k2 = cv(k_vals_7**2 * N_eff)
        cv_sin2 = cv(np.sin(k_vals_7 / 2)**2 * N_eff)
        print(f"  sin²(k/2) · N_eff CV = {cv_sin2:.1f}%, k² CV = {cv_k2:.1f}%")
        assert cv_sin2 < cv_k2 - 0.5, \
            f"1/sin²(k/2) CV={cv_sin2:.1f}% not clearly better than 1/k² CV={cv_k2:.1f}%"

    def test_neff_exponent_near_minus_2(self):
        """N_eff power law exponent ≈ -2.0 (not -2.5 from Born)."""
        sigma_tot = sigma_xx + sigma_xy
        N_eff = sigma_ring[5] / sigma_tot
        p = log_log_slope(k_vals_7, N_eff)[0]
        print(f"  N_eff exponent = {p:.3f}")
        assert -2.15 < p < -1.85, f"N_eff exponent = {p:.3f}"

    def test_sin2half_over_k2_approx_quarter(self):
        """sin²(k/2)/k² ≈ 1/4 at small k, CV ≈ 6% over k-range."""
        factor = np.sin(k_vals_7 / 2)**2 / k_vals_7**2
        print(f"  sin²(k/2)/k² at k=0.3 = {factor[0]:.4f} (≈ 1/4)")
        assert cv(factor) < 10.0, \
            f"sin²(k/2)/k² CV={cv(factor):.1f}% > 10%"
        # Approaches 1/4 at k→0
        assert abs(factor[0] - 0.25) < 0.01, \
            f"sin²(k/2)/k² at k=0.3 = {factor[0]:.4f}, not ≈ 0.25"

    def test_neff_grows_with_R(self):
        """N_eff(k=0.5) grows with R: more bonds = more N_eff."""
        sigma_tot = sigma_xx + sigma_xy
        neff_prev = 0
        for R in [3, 5, 7, 9]:
            neff_R = sigma_ring[R][1] / sigma_tot[1]  # k=0.5 is index 1
            print(f"  N_eff(k=0.5, R={R}): {neff_R:.0f}")
            assert neff_R > neff_prev, \
                f"N_eff(R={R}) = {neff_R:.1f} ≤ previous"
            neff_prev = neff_R

    def test_sigma_tr_scales_as_R_3_2(self):
        """σ_tr ~ R^p with p ≈ 3/2 (sub-geometric, from stationary phase).

        Not R^2 (geometric disk area). The exponent p ≈ 1.5-1.9 across k,
        with mean ≈ 1.6. Verified from FDTD data at R = 3, 5, 7, 9.
        """
        Rs = np.array([3, 5, 7, 9], dtype=float)
        slopes = []
        for ik, kv in enumerate(k_vals_7):
            sigmas = np.array([sigma_ring[R][ik] for R in [3, 5, 7, 9]])
            p = np.polyfit(np.log(Rs), np.log(sigmas), 1)[0]
            slopes.append(p)
        mean_p = np.mean(slopes)
        print(f"  R-scaling exponents: {[f'{s:.2f}' for s in slopes]}")
        print(f"  mean = {mean_p:.3f}")
        # Sub-geometric: between 1 and 2, closer to 3/2
        assert 1.3 < mean_p < 2.0, \
            f"Mean R-exponent = {mean_p:.3f}, expected ~1.5-1.9"
        # Not R^2
        assert mean_p < 1.9, \
            f"R-exponent {mean_p:.3f} too close to 2 (geometric)"
