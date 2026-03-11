"""Mechanism elimination: flatness is NOT from T-matrix, resonance, or compensation.

Source: old_tests/52_tmatrix_single_bond.py, 54_vectorial_tmatrix.py,
        53_polarization_decomposition.py, 59_eigenvalue_analysis.py,
        61_pair_sum_geometry.py

Results:
  TestScalarTMatrixFails:
    max |DK·G_anti| = 0.194  (threshold < 0.25, Born regime)
    max |T/DK - 1| = 0.240  (threshold < 0.25, <25% correction)
  TestVectorialTMatrixFails:
    |ΔK eigenvalue| = 1.618, scalar |DK| = 1.309, ratio = 1.236 (α=0.50: degenerate -2)
    max |eig·G_anti| = 0.240  (threshold < 0.5, still Born)
  TestNotResonance (R=5):
    |λ_max|: 0.603 (k=0.3) → 0.335 (k=1.5)  (all < 0.7)
    |λ_max| at k=0.3: 0.308 (R=3), 0.603 (R=5), 0.773 (R=7), 0.861 (R=9)  [R≤9 only]
  TestNoCompensation:
    σ_xy ratio k=1.5/k=0.3 = 68.0×  (threshold > 40)
    σ_xx max/min = 1.54×  (threshold < 2.0)
    σ_xy/σ_xx at k=0.3 = 0.0043  (threshold < 0.01)
  TestGeometryDependence:
    disk R=5: p = 0.354, line N=11: p = 0.099, annulus: p = 0.158
    single bond: p = 0.039  (threshold < 0.05)
    R=9: disk p=0.411 (253 bonds) >> annulus p=0.166 (56 bonds) — interior matters
    disk grows: 0.308 (R=3) → 0.354 (R=5) → 0.380 (R=7)
    line grows slowly: 0.099 (N=11) → 0.112 (N=21) → 0.127 (N=31)  [spread < 0.15]
  TestEigenvalueNeffFlat:
    Tr(AA†) slope: 0.027 (R=3), 0.027 (R=5), 0.027 (R=7), 0.028 (R=9)
    All ~0 (NOT -2). Tr(AA†) ≈ 1.4N (Born regime, threshold < 2N).

Analytic (seconds). BZ integrals + matrix computations.
"""
import numpy as np
import pytest
from helpers.config import K1, K2, c_lat, k_vals_7, ALPHA_REF, EPS_LAT, V_ref
from helpers.geometry import disk_bonds, line_bonds, annulus_bonds
from helpers.lattice import dispersion_sq, k_eff, get_omega_k2
from helpers.born import V_eff
from helpers.ms import build_G_matrix, eigenvalues_VG
from helpers.stats import log_log_slope
from data.sigma_bond import sigma_xx, sigma_xy


def _compute_enh_exponent(dx, dy):
    """Enhancement exponent from T-matrix: |Tb|²/|Vb|² power law in k.

    Uses V_ref = K1*(cos(2π·ALPHA_REF)-1) from config.
    """
    N = len(dx)
    V = V_ref  # tied to ALPHA_REF=0.30
    enh = np.zeros(len(k_vals_7))
    for ik, kv in enumerate(k_vals_7):
        G = build_G_matrix(dx, dy, kv)
        T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
        b = np.exp(1j * kv * dx)
        Tb = T @ b
        Vb = V * b
        enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
    return log_log_slope(k_vals_7, enh)[0]


class TestScalarTMatrixFails:
    """Scalar T-matrix for single z-bond is in Born regime.

    Related: TestVectorialTMatrixFails (2×2 channel), TestNotResonance (multi-bond).
    All three confirm Born regime from different angles:
      scalar |DK·G_anti| < 0.25 < vectorial |eig·G| < 0.5 < multi-bond |λ_max| < 0.9.
    """

    def test_dk_g_anti_small(self):
        """|DK·G_anti| ≤ 0.25 at all k (α=0.30) → Born regime.

        G_anti = BZ integral of (1-cos(kz))/(ω²-ω²(k)+iε).
        DK = K1·cm1.
        """
        cm1 = np.cos(2 * np.pi * ALPHA_REF) - 1
        DK = K1 * cm1
        omega_k2 = get_omega_k2()
        N_bz = 64
        kz_1d = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
        CZ_1d = np.cos(kz_1d)
        CZ_3d = np.broadcast_to(CZ_1d[None, None, :],
                                omega_k2.shape)
        assert CZ_3d.shape == omega_k2.shape, \
            f"CZ_3d {CZ_3d.shape} != omega_k2 {omega_k2.shape}"
        max_prod = 0
        for kv in k_vals_7:
            omega2 = dispersion_sq(kv)
            denom = omega2 - omega_k2 + 1j * EPS_LAT
            G_anti = np.mean((1 - CZ_3d) / denom)
            product = abs(DK * G_anti)
            max_prod = max(max_prod, product)
            assert product < 0.25, \
                f"|DK·G_anti| = {product:.3f} at k={kv}"
        print(f"  max |DK·G_anti| = {max_prod:.3f}")

    def test_scalar_tmatrix_is_born(self):
        """|DK·G_anti| < 0.25 means T ≈ DK (Born, <20% correction)."""
        # This is an interpretation test: if |DK·G| << 1, then
        # T = DK/(1-DK·G) ≈ DK (Born). The scalar channel doesn't explain
        # the non-Born behavior of σ_bond.
        cm1 = np.cos(2 * np.pi * ALPHA_REF) - 1
        DK = K1 * cm1
        omega_k2 = get_omega_k2()
        N_bz = 64
        kz_1d = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
        CZ_3d = np.broadcast_to(
            np.cos(kz_1d)[None, None, :], omega_k2.shape)
        max_ratio = 0
        for kv in k_vals_7:
            omega2 = dispersion_sq(kv)
            G_anti = np.mean((1 - CZ_3d) / (omega2 - omega_k2 + 1j * EPS_LAT))
            T_scalar = DK / (1 - DK * G_anti)
            correction = abs(T_scalar / DK - 1)
            max_ratio = max(max_ratio, correction)
        print(f"  max |T/DK - 1| = {max_ratio:.3f}")
        assert max_ratio < 0.25, \
            f"Max |T/DK - 1| = {max_ratio:.3f} > 25% — not Born?"


class TestVectorialTMatrixFails:
    """Vectorial 2×2 T-matrix also fails to explain σ_xx ≈ const.

    Source: old_tests/54_vectorial_tmatrix.py.

    ΔK = K1·[[cm1, -s_phi], [s_phi, cm1]] is a 2×2 matrix.
    Eigenvalues: K1·(cm1 ± i·s_phi) = K1·(e^{±i·2πα} - 1).
    |eigenvalue| = K1·√(2(1-cos(2πα))) > K1·|cm1| (vectorial > scalar).
    But still Born regime (|eig·G| < 1) → T ≈ ΔK.
    """

    def test_dk_eigenvalues(self):
        """ΔK eigenvalues are K1·(cm1 ± i·s_phi).

        At α=0.50: s_phi=0, eigenvalues degenerate to K1·cm1 = -2 (scalar).
        """
        for alpha in [0.10, 0.25, 0.30, 0.50]:
            phi = 2 * np.pi * alpha
            cm1 = np.cos(phi) - 1
            s_phi = np.sin(phi)
            DK = K1 * np.array([[cm1, -s_phi], [s_phi, cm1]])
            eigs = np.linalg.eigvals(DK)
            expected = np.sort_complex(
                np.array([K1 * (cm1 + 1j * s_phi),
                          K1 * (cm1 - 1j * s_phi)]))
            eigs_sorted = np.sort_complex(eigs)
            np.testing.assert_allclose(eigs_sorted, expected, atol=1e-14,
                                       err_msg=f"Failed at α={alpha}")

    def test_vectorial_stronger_than_scalar(self):
        """|ΔK eigenvalue| > |scalar DK| = K1·|cm1| at α=0.30."""
        phi = 2 * np.pi * 0.30
        cm1 = np.cos(phi) - 1
        s_phi = np.sin(phi)
        eig_mag = K1 * np.sqrt(cm1**2 + s_phi**2)
        scalar_mag = K1 * abs(cm1)
        ratio = eig_mag / scalar_mag
        print(f"  |ΔK eigenvalue| = {eig_mag:.3f}, scalar |DK| = {scalar_mag:.3f}, ratio = {ratio:.3f}")
        assert ratio > 1.2, \
            f"Vectorial/scalar ratio = {ratio:.3f}, expected > 1.2"

    def test_vectorial_still_born(self):
        """Vectorial |eig·G_anti| < 0.5 at all k → still Born regime.

        Uses same BZ integral as TestScalarTMatrixFails but with
        vectorial eigenvalue (factor ~1.24× larger).
        """
        phi = 2 * np.pi * ALPHA_REF
        cm1 = np.cos(phi) - 1
        s_phi = np.sin(phi)
        eig_mag = K1 * np.sqrt(cm1**2 + s_phi**2)
        omega_k2 = get_omega_k2()
        N_bz = 64
        kz_1d = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
        CZ_3d = np.broadcast_to(
            np.cos(kz_1d)[None, None, :], omega_k2.shape)
        max_prod = 0
        for kv in k_vals_7:
            omega2 = dispersion_sq(kv)
            G_anti = np.mean(
                (1 - CZ_3d) / (omega2 - omega_k2 + 1j * EPS_LAT))
            product = eig_mag * abs(G_anti)
            max_prod = max(max_prod, product)
            assert product < 0.5, \
                f"|eig·G_anti| = {product:.3f} at k={kv} — not Born?"
        print(f"  max |eig·G_anti| = {max_prod:.3f}")


class TestNotResonance:
    """System is far from resonance at α=0.30."""

    def test_lambda_max_far_from_1(self):
        """|λ_max| of VG is 0.3-0.6, far from 1."""
        dx, dy = disk_bonds(5)
        lam_first = None
        lam_last = None
        for ik, kv in enumerate(k_vals_7):
            eigs = eigenvalues_VG(dx, dy, kv, ALPHA_REF)
            lam_max = np.max(np.abs(eigs))
            if ik == 0:
                lam_first = lam_max
            if ik == len(k_vals_7) - 1:
                lam_last = lam_max
            assert lam_max < 0.7, \
                f"|λ_max| = {lam_max:.3f} at k={kv} — too close to 1"
            assert lam_max > 0.1, \
                f"|λ_max| = {lam_max:.3f} at k={kv} — too small"
        print(f"  |λ_max|: {lam_first:.3f} (k={k_vals_7[0]}) → {lam_last:.3f} (k={k_vals_7[-1]})")

    @pytest.mark.parametrize("R", [3, 5, 7, 9])
    def test_no_resonance_any_R(self, R):
        """No resonance: |λ_max| < 0.9. Validated empirically for R≤9.

        R=9 gives 0.861 at k=0.3 — larger R may approach resonance.
        """
        dx, dy = disk_bonds(R)
        eigs = eigenvalues_VG(dx, dy, 0.3, ALPHA_REF)
        lam_max = np.max(np.abs(eigs))
        print(f"  |λ_max| at k=0.3, R={R}: {lam_max:.3f}")
        assert lam_max < 0.9, f"|λ_max| = {lam_max:.3f} at R={R}"


class TestNoCompensation:
    """σ_xx and σ_xy do NOT compensate each other."""

    def test_sigma_xy_grows(self):
        """σ_xy grows > 40× from k=0.3 to k=1.5 (not constant)."""
        ratio = sigma_xy[-1] / sigma_xy[0]
        print(f"  σ_xy ratio k=1.5/k=0.3 = {ratio:.1f}×")
        assert ratio > 40, f"σ_xy ratio = {ratio:.1f}"

    def test_sigma_xx_does_not_decrease_much(self):
        """σ_xx varies < 2× (doesn't decrease to compensate σ_xy)."""
        ratio = sigma_xx.max() / sigma_xx.min()
        print(f"  σ_xx max/min = {ratio:.2f}×")
        assert ratio < 2.0, f"σ_xx max/min = {ratio:.2f}"

    def test_cross_pol_fraction_small_at_low_k(self):
        """σ_xy / σ_xx < 1% at k=0.3 (no compensation at low k)."""
        frac = sigma_xy[0] / sigma_xx[0]
        print(f"  σ_xy/σ_xx at k=0.3 = {frac:.4f}")
        assert frac < 0.01, f"σ_xy/σ_xx = {frac:.4f} at k=0.3"

    def test_cross_pol_grows_with_k(self):
        """σ_xy / σ_xx grows monotonically with k."""
        frac = sigma_xy / sigma_xx
        for i in range(len(frac) - 1):
            assert frac[i + 1] > frac[i], \
                f"σ_xy/σ_xx not growing: {frac[i]:.4f} → {frac[i+1]:.4f}"


class TestGeometryDependence:
    """MS shift depends on geometry: disk >> line, annulus (file 59 Part F).

    The +1/2 shift is specific to FILLED 2D disk. Line and annulus give
    much smaller shifts. Disk interior (filled area) matters, not perimeter.
    """

    def test_disk_shift_larger_than_line(self):
        """Disk R=5 gives larger MS shift than 1D line (11 bonds)."""
        dx_d, dy_d = disk_bonds(5)
        dx_l, dy_l = line_bonds(11, axis='x')
        p_disk = _compute_enh_exponent(dx_d, dy_d)
        p_line = _compute_enh_exponent(dx_l, dy_l)
        print(f"  disk R=5: p = {p_disk:.3f}, line N=11: p = {p_line:.3f}")
        assert p_disk > p_line + 0.1, \
            f"Disk shift {p_disk:.3f} not much larger than line {p_line:.3f}"

    def test_disk_shift_larger_than_annulus(self):
        """Disk R=5 gives larger MS shift than annulus R=4-5."""
        dx_d, dy_d = disk_bonds(5)
        dx_a, dy_a = annulus_bonds(5, width=1)
        p_disk = _compute_enh_exponent(dx_d, dy_d)
        p_ann = _compute_enh_exponent(dx_a, dy_a)
        print(f"  annulus: p = {p_ann:.3f}")
        assert p_disk > p_ann + 0.05, \
            f"Disk shift {p_disk:.3f} not larger than annulus {p_ann:.3f}"

    def test_disk_interior_matters(self):
        """R=9: disk (253 bonds) >> annulus (56 bonds). Interior drives shift."""
        dx_d, dy_d = disk_bonds(9)
        dx_a, dy_a = annulus_bonds(9, width=1)
        p_disk = _compute_enh_exponent(dx_d, dy_d)
        p_ann = _compute_enh_exponent(dx_a, dy_a)
        print(f"  R=9 disk: p = {p_disk:.3f} ({len(dx_d)} bonds), "
              f"annulus: p = {p_ann:.3f} ({len(dx_a)} bonds)")
        assert p_disk > p_ann + 0.1, \
            f"Disk R=9 shift {p_disk:.3f} not much larger than annulus {p_ann:.3f}"

    def test_single_bond_no_shift(self):
        """Single bond: MS shift ≈ 0 (no inter-bond propagation)."""
        dx_1 = np.array([0.0])
        dy_1 = np.array([0.0])
        p_1 = _compute_enh_exponent(dx_1, dy_1)
        print(f"  single bond: p = {p_1:.3f}")
        assert abs(p_1) < 0.05, \
            f"Single bond shift {p_1:.3f} ≠ 0"

    def test_disk_enhancement_grows_with_R(self):
        """Disk enhancement exponent grows with R (file 61 Part B).

        +1/2 requires filled 2D disk at large R. Line/annulus saturate.
        """
        p_list = []
        for R in [3, 5, 7]:
            dx_d, dy_d = disk_bonds(R)
            p_list.append(_compute_enh_exponent(dx_d, dy_d))
        print(f"  disk grows: {p_list[0]:.3f} (R=3) → {p_list[1]:.3f} (R=5) → {p_list[2]:.3f} (R=7)")
        # Should grow with R
        for i in range(len(p_list) - 1):
            assert p_list[i + 1] > p_list[i], \
                f"Disk enh not growing: R={[3,5,7][i]} p={p_list[i]:.3f}"

    def test_line_enhancement_saturates(self):
        """1D line enhancement grows much slower than disk (spread < 0.15)."""
        p_list = []
        for N in [11, 21, 31]:
            dx_l, dy_l = line_bonds(N, axis='x')
            p_list.append(_compute_enh_exponent(dx_l, dy_l))
        print(f"  line saturates: {p_list[0]:.3f} (N=11) → {p_list[1]:.3f} (N=21) → {p_list[2]:.3f} (N=31)")
        # Should be roughly constant (no growth)
        spread = max(p_list) - min(p_list)
        assert spread < 0.15, \
            f"Line enh varies too much: {p_list}, spread={spread:.3f}"


class TestEigenvalueNeffFlat:
    """Tr(AA†) = Σ 1/|1-λ_i|² does NOT scale as k^{-2}.

    If k^{-2} were structural in VG, then the resolvent amplification
    A = (I-VG)^{-1} would show N_eff ∝ k^{-2}.

    Instead Tr(AA†) ≈ const (slope ≈ 0) because system is in Born regime
    (max|λ| < 0.9) and 1/|1-λ|² ≈ 1 + 2Re(λ) + ... ≈ 1.

    The actual k^{-2} in N_eff = σ_ring/σ_bond is emergent from
    coherent phase sums + transport weighting, not from VG eigenvalues.
    """

    @pytest.mark.parametrize("R", [3, 5, 7, 9])
    def test_eigenvalue_neff_not_k_minus_2(self, R):
        """Tr(AA†) slope ≈ 0 (not -2) at all R."""
        dx, dy = disk_bonds(R)
        neff = []
        for kv in k_vals_7:
            eigs = eigenvalues_VG(dx, dy, kv, ALPHA_REF)
            neff.append(np.sum(1.0 / np.abs(1 - eigs)**2))
        slope, _ = log_log_slope(k_vals_7, np.array(neff))
        print(f"  Tr(AA†) slope R={R}: {slope:.3f}")
        assert abs(slope) < 0.5, \
            f"Eigenvalue N_eff slope = {slope:.3f} at R={R}, expected ~0"

    def test_eigenvalue_neff_approximately_N(self):
        """Tr(AA†) ≈ N_bonds (Born regime: 1/|1-λ|² ≈ 1)."""
        for R in [3, 5, 7]:
            dx, dy = disk_bonds(R)
            N = len(dx)
            neff_vals = []
            for kv in k_vals_7:
                eigs = eigenvalues_VG(dx, dy, kv, ALPHA_REF)
                neff_vals.append(np.sum(1.0 / np.abs(1 - eigs)**2))
            mean_neff = np.mean(neff_vals)
            print(f"  Tr(AA†) ≈ {mean_neff:.1f}, N = {N} at R={R} (ratio {mean_neff/N:.1f})")
            # Ratio ≈ 1.4 at all R. Tighten from 3x to 2x.
            assert 0.8 * N < mean_neff < 2.0 * N, \
                f"Tr(AA†) = {mean_neff:.0f}, N = {N} at R={R}"
