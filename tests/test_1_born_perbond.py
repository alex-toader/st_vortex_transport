"""§6.1 Per-bond Born mechanism: σ_bond = C₀·V/v_g².

Source: old_tests/55_monopole_dipole.py, 35_diagonal_offdiag.py,
        36_decomp_alpha_scan.py

Results:
  TestMonopoleDipole:
    Z_mono CV = 5.9%  (threshold < 7%)
    Z_dipo ratio k=1.5/k=0.3 = 22.4×  (threshold > 20)
    Z_mono + Z_dipo = 16π = 50.2655  (exact)
    Z_dipo → 0, Z_mono → 16π at k → 0
  TestBornShape (FDTD/Born shape ratio CV):
    Born formula C0·V/v_g² verified at α=0.25 (exact, rtol=1e-12)
    FDTD/Born ratio = 2.28–2.47  (MS enhancement, threshold > 2×)
    σ_tot = 2.8%  (threshold < 6%)
    σ_xx  = 0.6%  (threshold < 6%)
    σ_xy  = 14.7% (threshold < 20%, normalized at k=0.9)
  TestAlphaThreshold:
    |cm1| = |s_phi| = 1.000000 at α = 0.25  (exact)
    V(k) CV: 0.00% at α=0.25, 2.7% at α=0.30, 28.2% at α=0.10
    CV(V) asymmetric: steep drop below α=0.25, gentle rise above (< 6%)
  TestBornNormalization:
    1/v_g² range = 1.83×  (threshold < 2)
    1/sin²(k) range = 11.4×  (threshold > 10)
  TestDiagonalOffdiagDecomposition:
    |cm1|/|s_phi|: 0.325 (α=0.10), 1.000 (α=0.25), 1.376 (α=0.30)
    K_eff(0.30) = -1.309  (threshold < -1.0)
    V_diag CV = 5.9%  (threshold < 7%)
    V_offdiag CV = 72.7%  (threshold > 50%)

All analytic (0 seconds). Uses stored FDTD data for comparison.
"""
import numpy as np
import pytest
from helpers.config import K1, k_vals_7, c_lat
from helpers.born import peierls_coupling, V_eff, Z_monopole, Z_dipole, sigma_bond_born
from helpers.lattice import group_velocity
from helpers.stats import cv
from data.sigma_bond import sigma_bond, sigma_xx, sigma_xy, k_vals


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def alpha():
    return 0.30


@pytest.fixture
def born_shapes(alpha):
    """Born σ_xx, σ_xy, σ_tot shapes (unnormalized) at α=0.30."""
    cm1, s_phi = peierls_coupling(alpha)
    vg = group_velocity(k_vals_7)
    Z_mono = Z_monopole(k_vals_7)
    Z_dipo = Z_dipole(k_vals_7)
    born_xx = cm1**2 * Z_mono / vg**2
    born_xy = s_phi**2 * Z_dipo / vg**2
    born_tot = born_xx + born_xy
    return born_xx, born_xy, born_tot


# ── Tests ─────────────────────────────────────────────────────────

class TestMonopoleDipole:
    """Z_mono and Z_dipo angular integrals."""

    def test_z_mono_approximately_constant(self):
        """Z_mono = 8π(1+sinc(k)) varies slowly — CV < 7%."""
        Z = Z_monopole(k_vals_7)
        print(f"  Z_mono CV = {cv(Z):.1f}%")
        assert cv(Z) < 7.0, f"Z_mono CV={cv(Z):.1f}% > 7%"

    def test_z_dipo_grows_strongly(self):
        """Z_dipo = 8π(1−sinc(k)) grows > 20× from k=0.3 to k=1.5."""
        Z = Z_dipole(k_vals_7)
        ratio = Z[-1] / Z[0]
        print(f"  Z_dipo ratio k=1.5/k=0.3 = {ratio:.1f}x")
        assert ratio > 20, f"Z_dipo ratio={ratio:.1f} < 20"

    def test_z_mono_plus_dipo(self):
        """Z_mono + Z_dipo = 16π (completeness)."""
        Z_m = Z_monopole(k_vals_7)
        Z_d = Z_dipole(k_vals_7)
        expected = 16 * np.pi
        print(f"  Z_mono + Z_dipo = {(Z_m + Z_d)[0]:.4f}, 16π = {expected:.4f}")
        np.testing.assert_allclose(Z_m + Z_d, expected, rtol=1e-10)

    def test_z_limits_at_small_k(self):
        """Z_dipo → 0 and Z_mono → 16π as k → 0."""
        k_small = np.array([0.001])
        assert Z_dipole(k_small)[0] < 0.01, "Z_dipo must vanish at k→0"
        assert abs(Z_monopole(k_small)[0] - 16 * np.pi) < 0.01, \
            "Z_mono must → 16π at k→0"


class TestBornShape:
    """Per-bond Born shape matches FDTD at ~5% level."""

    def test_born_formula_at_025(self):
        """At α=0.25: V = cm1²·16π (since |cm1|=|s_phi|, Z_mono+Z_dipo=16π).

        Independent check that sigma_bond_born = C0·V/v_g² is numerically correct.
        """
        cm1, _ = peierls_coupling(0.25)
        C0 = K1**2 / (16 * np.pi**2 * c_lat**4)
        vg = group_velocity(k_vals_7)
        expected = C0 * 16 * np.pi * cm1**2 / vg**2
        born = sigma_bond_born(k_vals_7, 0.25)
        np.testing.assert_allclose(born, expected, rtol=1e-12)

    def test_born_underestimates_fdtd(self):
        """FDTD > Born at all k: MS enhancement factor ~2.3×.

        Born captures the shape but underestimates the amplitude.
        The T-matrix (MS) provides the missing enhancement.
        """
        born = sigma_bond_born(k_vals_7, 0.30)
        ratio = sigma_bond / born
        for i, k in enumerate(k_vals_7):
            print(f"    k={k:.1f}: FDTD={sigma_bond[i]:.4f}, Born={born[i]:.4f}, "
                  f"ratio={ratio[i]:.3f}")
        print(f"  FDTD/Born ratio: {ratio.min():.2f}–{ratio.max():.2f}")
        assert np.all(sigma_bond > born), \
            "Born should underestimate FDTD (MS enhancement missing)"
        assert ratio.min() > 2.0, f"MS enhancement < 2× at some k"

    def test_sigma_tot_shape_cv(self, born_shapes):
        """FDTD/Born shape ratio for σ_tot: CV < 6%."""
        _, _, born_tot = born_shapes
        sigma_tot_fdtd = sigma_xx + sigma_xy
        # Normalize both to k=0.3
        born_n = born_tot / born_tot[0]
        fdtd_n = sigma_tot_fdtd / sigma_tot_fdtd[0]
        ratio = fdtd_n / born_n
        print(f"  σ_tot shape CV = {cv(ratio):.1f}%")
        assert cv(ratio) < 6.0, f"σ_tot shape CV={cv(ratio):.1f}% > 6%"

    def test_sigma_xx_shape_cv(self, born_shapes):
        """FDTD/Born shape ratio for σ_xx: CV < 6%."""
        born_xx, _, _ = born_shapes
        born_n = born_xx / born_xx[0]
        fdtd_n = sigma_xx / sigma_xx[0]
        ratio = fdtd_n / born_n
        print(f"  σ_xx shape CV = {cv(ratio):.1f}%")
        assert cv(ratio) < 6.0, f"σ_xx shape CV={cv(ratio):.1f}% > 6%"

    def test_sigma_xy_shape_cv(self, born_shapes):
        """FDTD/Born shape ratio for σ_xy: CV < 20% (normalized at k=0.9)."""
        _, born_xy, _ = born_shapes
        # Normalize at k=0.9 (index 3) because Z_dipo ≈ 0 at k=0.3
        born_n = born_xy / born_xy[3]
        fdtd_n = sigma_xy / sigma_xy[3]
        ratio = fdtd_n / born_n
        print(f"  σ_xy shape CV = {cv(ratio):.1f}%")
        assert cv(ratio) < 20.0, f"σ_xy shape CV={cv(ratio):.1f}% > 20%"


class TestAlphaThreshold:
    """α = 0.25 threshold: |cm1| = |s_phi|."""

    def test_cm1_equals_sphi_at_025(self):
        """|cm1| = |s_phi| exactly at α = 0.25."""
        cm1, s_phi = peierls_coupling(0.25)
        print(f"  |cm1| = {abs(cm1):.6f}, |s_phi| = {abs(s_phi):.6f} at α=0.25")
        np.testing.assert_allclose(abs(cm1), abs(s_phi), rtol=1e-10)

    def test_v_const_above_threshold(self):
        """V(k) ≈ const at α = 0.30 (CV < 5%)."""
        cm1, s_phi = peierls_coupling(0.30)
        V = cm1**2 * Z_monopole(k_vals_7) + s_phi**2 * Z_dipole(k_vals_7)
        print(f"  V(k) CV = {cv(V):.1f}% at α=0.30")
        assert cv(V) < 5.0, f"V CV={cv(V):.1f}% > 5%"

    def test_v_varies_below_threshold(self):
        """V(k) varies significantly at α = 0.10 (CV > 20%)."""
        cm1, s_phi = peierls_coupling(0.10)
        V = cm1**2 * Z_monopole(k_vals_7) + s_phi**2 * Z_dipole(k_vals_7)
        print(f"  V(k) CV = {cv(V):.1f}% at α=0.10")
        assert cv(V) > 20.0, f"V CV={cv(V):.1f}% < 20% at α=0.10"

    def test_cv_minimum_at_025(self):
        """CV(V) is minimized at α = 0.25 (exactly 0%)."""
        cm1, s_phi = peierls_coupling(0.25)
        V = cm1**2 * Z_monopole(k_vals_7) + s_phi**2 * Z_dipole(k_vals_7)
        print(f"  V(k) CV = {cv(V):.2f}% at α=0.25")
        assert cv(V) < 0.5, f"CV(V) at α=0.25 = {cv(V):.2f}% ≠ 0"

    def test_cv_v_alpha_scan(self):
        """CV(V) asymmetric: steep drop below α=0.25, gentle rise above.

        α < 0.25: dipole dominates, Z_dipo varies strongly → CV large (28% at 0.10).
        α = 0.25: |cm1|=|s_phi|, V ∝ Z_mono+Z_dipo = 16π = const → CV = 0.
        α > 0.25: monopole dominates, Z_mono varies slowly → CV small (~3-6%).
        """
        alphas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        cvs = []
        for a in alphas:
            cm1, s_phi = peierls_coupling(a)
            V = cm1**2 * Z_monopole(k_vals_7) + s_phi**2 * Z_dipole(k_vals_7)
            cvs.append(cv(V))
        # Minimum at α=0.25
        i_min = np.argmin(cvs)
        print(f"  CV(V) min at α={alphas[i_min]:.2f} ({cvs[i_min]:.1f}%)")
        assert alphas[i_min] == 0.25, f"CV(V) min at α={alphas[i_min]}, not 0.25"
        # Decreasing below crossover
        for i in range(3):  # 0.10 → 0.15 → 0.20 → 0.25
            assert cvs[i] > cvs[i + 1], \
                f"CV not decreasing: α={alphas[i]}({cvs[i]:.1f}%) → α={alphas[i+1]}({cvs[i+1]:.1f}%)"
        # Rising above crossover (gentle: 0% → 2.7% → 4.3% → 5.2% → ...)
        for i in range(3, len(alphas) - 1):  # 0.25 → 0.30 → ... → 0.50
            assert cvs[i + 1] >= cvs[i], \
                f"CV not rising: α={alphas[i]}({cvs[i]:.1f}%) → α={alphas[i+1]}({cvs[i+1]:.1f}%)"
        # All above α=0.20: CV < 6%
        for i in range(2, len(alphas)):
            assert cvs[i] < 6.0, f"CV(V)={cvs[i]:.1f}% > 6% at α={alphas[i]}"


class TestBornNormalization:
    """v_g normalization (not 1/sin²(k))."""

    def test_vg_varies_less_than_sin2k(self):
        """1/v_g² varies < 2× while 1/sin²(k) varies > 10×."""
        vg = group_velocity(k_vals_7)
        inv_vg2 = 1 / vg**2
        inv_sin2k = 1 / np.sin(k_vals_7)**2
        range_vg = inv_vg2[-1] / inv_vg2[0]
        range_sin = inv_sin2k.max() / inv_sin2k.min()
        print(f"  1/v_g² range = {range_vg:.2f}x, 1/sin²(k) range = {range_sin:.1f}x")
        assert range_vg < 2.0, f"1/v_g² range={range_vg:.1f} > 2"
        assert range_sin > 10.0, f"1/sin²(k) range={range_sin:.1f} < 10"


class TestDiagonalOffdiagDecomposition:
    """Peierls correction R-I decomposes into cm1·I (diagonal) + s·J (offdiag).

    Source: old_tests/35_diagonal_offdiag.py, 36_decomp_alpha_scan.py.

    cm1 = cos(2πα)-1 (diagonal, displacement-like coupling)
    s_phi = sin(2πα) (off-diagonal, rotation/mixing)

    Key results:
    - Diagonal dominates at α≥0.25 (|cm1| > |s_phi|)
    - Diagonal coupling is displacement-like (force ∝ u_neighbor, no self-energy)
    - Off-diagonal is a weaker scatterer (σ_offdiag/σ_diag < 1 at strong α)
    - File 35 FDTD: diagonal CV=8.8%, offdiag CV=23.7% at α=0.30
    - File 36: diagonal only flat at α≥0.30, NOT at weak coupling
    """

    def test_cm1_dominates_at_030(self):
        """|cm1| > |s_phi| at α=0.30: diagonal part is larger."""
        cm1, s_phi = peierls_coupling(0.30)
        ratio = abs(cm1) / abs(s_phi)
        print(f"  |cm1|/|s_phi| = {ratio:.3f} at α=0.30")
        assert ratio > 1.3, f"|cm1|/|s_phi| = {ratio:.2f} at α=0.30"

    def test_sphi_dominates_at_010(self):
        """|s_phi| > |cm1| at α=0.10: off-diagonal dominates at weak coupling."""
        cm1, s_phi = peierls_coupling(0.10)
        ratio = abs(cm1) / abs(s_phi)
        print(f"  |cm1|/|s_phi| = {ratio:.3f} at α=0.10")
        assert ratio < 0.5, f"|cm1|/|s_phi| = {ratio:.2f} at α=0.10"

    def test_crossover_at_025(self):
        """|cm1| = |s_phi| exactly at α=0.25 — the crossover point."""
        cm1, s_phi = peierls_coupling(0.25)
        print(f"  |cm1|/|s_phi| = {abs(cm1)/abs(s_phi):.6f} at α=0.25")
        np.testing.assert_allclose(abs(cm1), abs(s_phi), rtol=1e-10)

    def test_ratio_grows_with_alpha(self):
        """|cm1|/|s_phi| grows monotonically for α ∈ [0.05, 0.45]."""
        ratios = []
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
            cm1, s_phi = peierls_coupling(alpha)
            ratios.append(abs(cm1) / abs(s_phi))
        for i in range(len(ratios) - 1):
            assert ratios[i + 1] > ratios[i], \
                f"|cm1|/|s_phi| not growing at α step {i}"

    def test_diagonal_is_displacement_coupling(self):
        """Diagonal Peierls gauge correction = K1·cm1·u_neighbor.

        Normal spring: f = K1·(u_neighbor - u_self).
        Gauged bond:   f = K1·R·u_neighbor - K1·u_self.
        Gauge correction = K1·(R-I)·u_neighbor = K1·cm1·u_neighbor (diagonal part).
        The self-energy term (-K1·u_self) is unchanged by the gauge.
        At α=0.30: K1·cm1 = K1·(cos(2π·0.3)-1) ≈ -1.31.
        """
        cm1, _ = peierls_coupling(0.30)
        K_eff = K1 * cm1
        # Effective coupling is negative and strong
        print(f"  K_eff(0.30) = {K_eff:.3f}")
        assert K_eff < -1.0, f"K_eff = {K_eff:.3f}, expected < -1.0"
        # Born vertex for diagonal-only: V_diag = cm1² · Z_mono
        # Z_mono ≈ const → V_diag ≈ const (flat vertex)
        V_diag = cm1**2 * Z_monopole(k_vals_7)
        print(f"  V_diag CV = {cv(V_diag):.1f}%")
        assert cv(V_diag) < 7.0, f"V_diag CV={cv(V_diag):.1f}% > 7%"

    def test_offdiag_vertex_not_const(self):
        """Off-diagonal vertex V_offdiag = s_phi²·Z_dipo varies strongly.

        Z_dipo grows >20× from k=0.3 to k=1.5 → V_offdiag is NOT flat.
        This is why off-diagonal coupling does not produce flat integrand.
        """
        _, s_phi = peierls_coupling(0.30)
        V_offdiag = s_phi**2 * Z_dipole(k_vals_7)
        print(f"  V_offdiag CV = {cv(V_offdiag):.1f}%")
        assert cv(V_offdiag) > 50.0, \
            f"V_offdiag CV={cv(V_offdiag):.1f}% < 50%"


