"""§9 Forward decoherence mechanism: microscopic decomposition.

Deep tests of the forward decoherence mechanism discovered during paper
writing (internal/70_forward_decoherence_investigation.md). Goes beyond
test_3's TestForwardDecoherence (basic suppression/tilt/R-monotonicity)
to test the microscopic structure.

Results:
  TestAngularUniformity (F2, 4 tests):
    MS/Born ratio: mean=0.537, CV=4.5% at k=0.3, R=5.
    Max ratio = 0.622 < 1 everywhere. MS rescales uniformly.
    k=1.5: CV=11.7%, ratio 0.92–1.69 (uniformity is low-k specific).
    Transport/total ratio diff = 2.8% at k=0.3 (uniform → weight-independent).
  TestTMatrixDecomposition (F3, 5 tests):
    |T_jj/V|² = 1.347 (diagonal enhances). Forward: diag=-90.4, offdiag=+38.4.
    Cross term = -137.0 (destructive). |cross|/diagonal = 0.73.
    α=0.05: |cross|/diag = 2.8% vs α=0.30: 73% (26× ratio).
  TestTransportShiftRIndependence (F4, 6 tests):
    Shift: +0.410 (R=3), +0.448 (R=5), +0.402 (R=7), +0.414 (R=9). CV=4.2%.
    Integrand CV: 15.7% (R=3), 12.5% (R=5), 12.4% (R=7), 13.2% (R=9).
    Random disk: shift=0.272 (CV=9.4%, 5 seeds). Positive, not lattice-specific.
    CV at 15 k-points: 12.6% (same as 7 pts — flatness not sampling artifact).
    α=0.40 shift > 2× α=0.20 shift (shift ∝ |V|).
  TestGeometryDependence (F7, 3 tests):
    C_disk=0.270 > C_annulus=0.121 > C_line=0.075. Disk/line ratio = 3.6×.
  TestCFullVsCStatic (F8, 4 tests):
    C_full < C_static at all R. Ratio 1.44–1.56 (phases reduce range).
    Physical r_cut=3 converges (|Δp/p|=0.6%). Static r_cut=3→5 grows 29%.
  TestMeanFieldCoupling (F9, 3 tests):
    |V·⟨ΣG⟩|: 0.667 (k=0.3) → 0.157 (k=1.5). Power law p = -0.94.
    Spectral radius: 0.603 (k=0.3), 0.335 (k=0.9,1.5). All < 1.
  TestPowerDecomposition (F10, 5 tests):
    cross/born = 2 × diag_amp × offdiag_amp × cos(Φ).
    Exponents sum: +0.01 + (-0.48) + (-0.16) = -0.63 ✓.
    offdiag_amp ≈ -0.50 (stable, from N_coop × incoh).
    Crossover k (enh=1): 0.61 (R=3) → 0.76 (R=5) → 1.46 (R=9), grows with R.
  TestBornSeriesAlternation (F11, 5 tests):
    |G_ij|² is k-independent. cancel_ratio: 0.37 (k=0.3) vs 0.93 (k=1.5).
    VG coherence: 0.83 (k=0.3) vs 0.20 (k=1.5).
    Eigenvalue |λ| CV: 41% (k=0.3, clustered) → 24% (k=1.5, spread).
  TestCosPhiConvergence (F12, 2 tests):
    |cos(Φ)| exponent: -0.35 (R=3) → -0.05 (R=9) → 0.
  TestCross12SignChange (F13, 5 tests):
    cross_12 = 2Re⟨VG*·VG²⟩: -11% at k=0.3, +79% at k=1.5.
    Path excess phases: frac(k_eff·pe > π) = 7.5% (k=0.3) → 68% (k=1.5).

Analytic (seconds). Matrix computations.
"""
import numpy as np
import pytest
from helpers.config import K1, K2, c_lat, k_vals_7, ALPHA_REF, V_ref
from helpers.geometry import disk_bonds, annulus_bonds, line_bonds
from helpers.lattice import k_eff
from helpers.born import V_eff, sigma_bond_born
from helpers.ms import build_G_matrix, build_VG, T_matrix, eigenvalues_VG
from helpers.stats import cv, log_log_slope


# ── Angular grid (same as test_3) ────────────────────────────────

N_TH, N_PH = 100, 100


@pytest.fixture(scope="module")
def angular_grid():
    thetas = np.linspace(0, np.pi, N_TH)
    phis = np.linspace(0, 2 * np.pi, N_PH, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    sin_th = np.sin(TH)
    cos_th = np.cos(TH)
    transport = 1 - sin_th * np.cos(PH)
    return TH, PH, sin_th, cos_th, transport


# ── Helpers ───────────────────────────────────────────────────────

def _ms_born_ratio_angular(dx, dy, k, alpha, TH, PH, sin_th, cos_th):
    """MS/Born power ratio at each angular direction.

    Returns (N_TH, N_PH) array of |f_MS|²/|f_Born|² at each (θ,φ).
    """
    N = len(dx)
    V = V_eff(alpha)
    G = build_G_matrix(dx, dy, k)
    T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
    b = np.exp(1j * k * dx)
    Tb = T @ b
    Vb = V * b

    phase_out = k * (
        sin_th[:, :, None] * np.cos(PH)[:, :, None] * dx[None, None, :]
        + sin_th[:, :, None] * np.sin(PH)[:, :, None] * dy[None, None, :])
    a = np.exp(-1j * phase_out)
    f_ms = np.sum(a * Tb[None, None, :], axis=2)
    f_born = np.sum(a * Vb[None, None, :], axis=2)

    # Avoid division by zero at nodes
    born_pow = np.abs(f_born)**2
    mask = born_pow > 1e-20 * np.max(born_pow)
    ratio = np.full_like(born_pow, np.nan)
    ratio[mask] = np.abs(f_ms[mask])**2 / born_pow[mask]
    return ratio


def _compute_neff_transport(R, k_arr, alpha, TH, PH, sin_th, cos_th,
                            transport):
    """N_eff (Born and MS) via angular integration with transport weight.

    Same as test_3's compute_neff_born_ms but as a module-level function.
    """
    dx, dy = disk_bonds(R)
    N = len(dx)
    V = V_eff(alpha)

    neff_b = np.zeros(len(k_arr))
    neff_m = np.zeros(len(k_arr))

    for ik, kv in enumerate(k_arr):
        G = build_G_matrix(dx, dy, kv)
        T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
        b = np.exp(1j * kv * dx)
        Tb = T @ b
        Vb = V * b

        phase_out = kv * (
            sin_th[:, :, None] * np.cos(PH)[:, :, None] * dx[None, None, :]
            + sin_th[:, :, None] * np.sin(PH)[:, :, None]
            * dy[None, None, :])
        a = np.exp(-1j * phase_out)
        f_ms = np.sum(a * Tb[None, None, :], axis=2)
        f_born = np.sum(a * Vb[None, None, :], axis=2)

        qz = kv * cos_th
        F2 = 4 * np.cos(qz / 2)**2
        w = F2 * transport * sin_th
        s_single = np.abs(V)**2 * np.sum(w)
        neff_b[ik] = np.sum(np.abs(f_born)**2 * w) / s_single
        neff_m[ik] = np.sum(np.abs(f_ms)**2 * w) / s_single

    return neff_b, neff_m


def _enhancement_exponent(dx, dy, k_arr, alpha=ALPHA_REF):
    """Exponent of Σ|Tb|²/Σ|Vb|² (total power, NO angular/transport weight).

    Returns (slope, enhancement_array).
    """
    V = V_eff(alpha)
    N = len(dx)
    enh = np.zeros(len(k_arr))
    for ik, kv in enumerate(k_arr):
        G = build_G_matrix(dx, dy, kv)
        b = np.exp(1j * kv * dx)
        Vb = V * b
        T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
        Tb = T @ b
        enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
    p, _ = log_log_slope(k_arr, enh)
    return p, enh


# ── Tests ─────────────────────────────────────────────────────────

class TestAngularUniformity:
    """F2: MS/Born ratio is constant at all angles.

    MS does NOT reshape the angular cone. It suppresses the amplitude
    UNIFORMLY at all angles. Transport-weighted 50%/90% angles are
    identical for Born and MS.

    Ref: internal/70_forward_decoherence_investigation.md, F2.
    """

    def test_ratio_cv_below_5pct(self, angular_grid):
        """MS/Born power ratio has CV < 5% across all angles at k=0.3, R=5.

        Investigation found CV = 0.539–0.554 range (< 3% variation).
        Threshold at 5% to allow angular grid discretization.
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        dx, dy = disk_bonds(5)
        ratio = _ms_born_ratio_angular(dx, dy, 0.3, ALPHA_REF,
                                       TH, PH, sin_th, cos_th)
        valid = ratio[~np.isnan(ratio)]
        ratio_cv = np.std(valid) / np.mean(valid) * 100
        print(f"  MS/Born angular ratio: mean={np.mean(valid):.3f}, "
              f"CV={ratio_cv:.1f}%")
        assert ratio_cv < 5.0, \
            f"Angular ratio CV = {ratio_cv:.1f}%, expected < 5%"

    def test_ratio_below_1_everywhere(self, angular_grid):
        """MS/Born < 1 at ALL angles at k=0.3 (uniform suppression).

        Not just forward suppression — MS suppresses everywhere equally.
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        dx, dy = disk_bonds(5)
        ratio = _ms_born_ratio_angular(dx, dy, 0.3, ALPHA_REF,
                                       TH, PH, sin_th, cos_th)
        valid = ratio[~np.isnan(ratio)]
        max_ratio = np.max(valid)
        print(f"  Max angular ratio at k=0.3: {max_ratio:.3f}")
        assert max_ratio < 1.0, \
            f"MS/Born ratio > 1 at some angle: max = {max_ratio:.3f}"

    def test_uniformity_breaks_at_high_k(self, angular_grid):
        """MS/Born ratio CV > 10% at k=1.5 (uniformity is low-k specific).

        At k=1.5, Born pattern has angular nodes where |f_Born|→0.
        Away from nodes (1% mask), ratio varies 0.92–1.69 (CV=11.7%).
        Confirms decoherence mechanism is k-specific.
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        dx, dy = disk_bonds(5)
        N = len(dx)
        V = V_eff(ALPHA_REF)
        kv = 1.5
        G = build_G_matrix(dx, dy, kv)
        T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
        b = np.exp(1j * kv * dx)
        Tb = T @ b
        Vb = V * b
        phase_out = kv * (
            sin_th[:, :, None] * np.cos(PH)[:, :, None] * dx[None, None, :]
            + sin_th[:, :, None] * np.sin(PH)[:, :, None]
            * dy[None, None, :])
        a = np.exp(-1j * phase_out)
        f_ms = np.sum(a * Tb[None, None, :], axis=2)
        f_born = np.sum(a * Vb[None, None, :], axis=2)
        born_pow = np.abs(f_born)**2
        ms_pow = np.abs(f_ms)**2
        # 1% mask to exclude Born angular nodes
        mask = born_pow > 0.01 * np.max(born_pow)
        ratio = ms_pow[mask] / born_pow[mask]
        ratio_cv = np.std(ratio) / np.mean(ratio) * 100
        print(f"  k=1.5: mean={np.mean(ratio):.3f}, CV={ratio_cv:.1f}%, "
              f"N_valid={np.sum(mask)}/{mask.size}")
        assert ratio_cv > 10.0, \
            f"CV = {ratio_cv:.1f}% at k=1.5, expected > 10% (non-uniform)"

    def test_transport_equals_total_at_low_k(self, angular_grid):
        """Transport-weighted and total power MS/Born ratios match at k=0.3.

        If MS rescales uniformly at all angles, then the transport weight
        (1-cosθ_s) doesn't change the ratio. Confirms the F4 transport
        shift follows directly from uniform suppression.
        Difference: 2.8% at k=0.3.
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        dx, dy = disk_bonds(5)
        N = len(dx)
        V = V_eff(ALPHA_REF)
        kv = 0.3
        G = build_G_matrix(dx, dy, kv)
        T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
        b = np.exp(1j * kv * dx)
        Tb = T @ b
        Vb = V * b
        phase_out = kv * (
            sin_th[:, :, None] * np.cos(PH)[:, :, None] * dx[None, None, :]
            + sin_th[:, :, None] * np.sin(PH)[:, :, None]
            * dy[None, None, :])
        a = np.exp(-1j * phase_out)
        f_ms = np.sum(a * Tb[None, None, :], axis=2)
        f_born = np.sum(a * Vb[None, None, :], axis=2)
        ms_pow = np.abs(f_ms)**2
        born_pow = np.abs(f_born)**2
        qz = kv * cos_th
        F2 = 4 * np.cos(qz / 2)**2
        # Total power ratio
        w_tot = F2 * sin_th
        ratio_tot = np.sum(ms_pow * w_tot) / np.sum(born_pow * w_tot)
        # Transport-weighted ratio
        w_tr = F2 * transport * sin_th
        ratio_tr = np.sum(ms_pow * w_tr) / np.sum(born_pow * w_tr)
        diff = abs(ratio_tr - ratio_tot) / ratio_tot * 100
        print(f"  ratio_tot={ratio_tot:.4f}, ratio_tr={ratio_tr:.4f}, "
              f"diff={diff:.1f}%")
        assert diff < 5.0, \
            f"Transport/total ratio diff = {diff:.1f}%, expected < 5%"


class TestTMatrixDecomposition:
    """F3: Microscopic mechanism — diagonal enhances, off-diagonal interferes.

    T = T_diag + T_offdiag. The diagonal (self-energy) enhances each
    bond. The off-diagonal (rescattering) produces amplitudes with
    OPPOSITE sign to Born, causing destructive interference.

    At low k, phases are aligned so rescattering adds coherently →
    strong cancellation. At high k, phases oscillate → cancellation
    averages out → Born regime.

    Ref: internal/70_forward_decoherence_investigation.md, F3.
    """

    def test_diagonal_enhances(self):
        """Self-energy: |T_jj|² > |V|² (diagonal enhances each bond).

        T_jj = V/(1-V·G₀₀) ≈ -1.52 vs V ≈ -1.31 at α=0.30.
        Enhancement ratio |T_jj/V|² ≈ 1.35.
        """
        dx, dy = disk_bonds(5)
        k = 0.3
        T = T_matrix(dx, dy, k, ALPHA_REF)
        V = V_eff(ALPHA_REF)
        T_diag = np.diag(T)
        enh_diag = np.mean(np.abs(T_diag)**2) / abs(V)**2
        print(f"  |T_jj/V|² = {enh_diag:.3f} (expected > 1)")
        assert enh_diag > 1.0, \
            f"|T_jj/V|² = {enh_diag:.3f}, expected > 1"

    def test_offdiag_opposes_diagonal(self):
        """Off-diagonal rescattering opposes diagonal in forward direction.

        Forward sum: Σ (T_offdiag · b)_j has OPPOSITE sign to Σ (T_diag · b)_j.
        """
        dx, dy = disk_bonds(5)
        k = 0.3
        V = V_eff(ALPHA_REF)
        N = len(dx)
        T = T_matrix(dx, dy, k, ALPHA_REF)
        b = np.exp(1j * k * dx)

        T_diag_mat = np.diag(np.diag(T))
        T_offdiag_mat = T - T_diag_mat

        fwd_diag = np.sum(T_diag_mat @ b)
        fwd_offdiag = np.sum(T_offdiag_mat @ b)

        # They should have opposite signs (real parts)
        product = fwd_diag.real * fwd_offdiag.real
        print(f"  Forward diag: {fwd_diag.real:.1f}, "
              f"offdiag: {fwd_offdiag.real:.1f}, "
              f"product: {product:.1f}")
        assert product < 0, \
            f"Diag and offdiag not opposing: product = {product:.1f}"

    def test_cross_term_negative(self):
        """Cross term 2·Re(Σ T_diag*·T_offdiag) is negative (destructive).

        At k=0.3, R=5: cross term is ~73% of diagonal power.
        """
        dx, dy = disk_bonds(5)
        k = 0.3
        V = V_eff(ALPHA_REF)
        N = len(dx)
        T = T_matrix(dx, dy, k, ALPHA_REF)
        b = np.exp(1j * k * dx)

        Tb = T @ b
        T_diag_b = np.diag(T) * b
        T_offdiag_b = Tb - T_diag_b

        total_pow = np.sum(np.abs(Tb)**2)
        diag_pow = np.sum(np.abs(T_diag_b)**2)
        offdiag_pow = np.sum(np.abs(T_offdiag_b)**2)
        cross = total_pow - diag_pow - offdiag_pow  # = 2*Re(Σ d*·o)

        print(f"  diag={diag_pow:.1f}, offdiag={offdiag_pow:.1f}, "
              f"cross={cross:.1f}")
        assert cross < 0, \
            f"Cross term = {cross:.1f}, expected < 0 (destructive)"

    def test_cross_fraction_large_at_low_k(self):
        """Cross term is > 50% of diagonal at k=0.3 (strong cancellation).

        Investigation found |cross|/diagonal = 73% at k=0.3, R=5.
        """
        dx, dy = disk_bonds(5)
        k = 0.3
        T = T_matrix(dx, dy, k, ALPHA_REF)
        b = np.exp(1j * k * dx)

        Tb = T @ b
        T_diag_b = np.diag(T) * b
        T_offdiag_b = Tb - T_diag_b

        diag_pow = np.sum(np.abs(T_diag_b)**2)
        total_pow = np.sum(np.abs(Tb)**2)
        cross = total_pow - diag_pow - np.sum(np.abs(T_offdiag_b)**2)
        frac = abs(cross) / diag_pow

        print(f"  |cross|/diagonal = {frac:.2f} at k=0.3 (expected > 0.5)")
        assert frac > 0.50, \
            f"|cross|/diagonal = {frac:.2f}, expected > 0.50"

    def test_cross_fraction_scales_with_alpha(self):
        """Cross fraction at α=0.05 is < 10× cross fraction at α=0.30.

        At weak coupling: |cross|/diag = 2.8% (α=0.05) vs 73% (α=0.30).
        Links to test_5's TestWeakCouplingNotFlat: weak coupling means
        weak rescattering → weak cross → no MS correction → no flatness.
        """
        dx, dy = disk_bonds(5)
        k = 0.3
        fracs = {}
        for alpha in [0.05, 0.30]:
            T = T_matrix(dx, dy, k, alpha)
            b = np.exp(1j * k * dx)
            Tb = T @ b
            T_diag_b = np.diag(T) * b
            T_offdiag_b = Tb - T_diag_b
            diag_pow = np.sum(np.abs(T_diag_b)**2)
            total_pow = np.sum(np.abs(Tb)**2)
            cross = total_pow - diag_pow - np.sum(np.abs(T_offdiag_b)**2)
            fracs[alpha] = abs(cross) / diag_pow
            print(f"  α={alpha}: |cross|/diag = {fracs[alpha]:.3f}")
        ratio = fracs[0.30] / fracs[0.05]
        print(f"  Ratio α=0.30/α=0.05 = {ratio:.1f}×")
        assert ratio > 10, \
            f"Cross fraction ratio = {ratio:.1f}, expected > 10"


class TestTransportShiftRIndependence:
    """F4: Transport-weighted exponent shift is R-INDEPENDENT.

    With CORRECT transport weight (1 - sinθ cosφ = 1 - cosθ_scattering),
    the MS exponent shift is +0.448 ± 0.03, constant across R=3..15.

    Total power suppression grows with R (F1), but the transport
    cross-section shift is constant because transport weight (1-cosθ_s)
    filters out the forward cone where Born/MS differ most.

    Ref: internal/70_forward_decoherence_investigation.md, F4.
    """

    def test_shift_constant_across_R(self, angular_grid):
        """Shift = p_MS - p_Born is constant (CV < 15%) across R=3,5,7,9.

        Investigation: shift = +0.41, +0.45, +0.40, +0.41 at R=3,5,7,9.
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        shifts = []
        for R in [3, 5, 7, 9]:
            nb, nm = _compute_neff_transport(R, k_vals_7, ALPHA_REF,
                                             TH, PH, sin_th, cos_th,
                                             transport)
            p_b = log_log_slope(k_vals_7, nb)[0]
            p_m = log_log_slope(k_vals_7, nm)[0]
            shift = p_m - p_b
            shifts.append(shift)
            print(f"  R={R}: p_Born={p_b:.2f}, p_MS={p_m:.2f}, "
                  f"shift={shift:.3f}")
        shifts = np.array(shifts)
        shift_cv = cv(shifts)
        print(f"  Shift CV = {shift_cv:.1f}%")
        assert shift_cv < 15.0, \
            f"Shift CV = {shift_cv:.1f}%, expected < 15% (R-independent)"

    def test_shift_positive_and_near_half(self, angular_grid):
        """Shift ∈ [+0.3, +0.6] at R=5 (close to +1/2).

        Investigation: shift = +0.448 at R=5, α=0.30.
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        nb, nm = _compute_neff_transport(5, k_vals_7, ALPHA_REF,
                                         TH, PH, sin_th, cos_th,
                                         transport)
        p_b = log_log_slope(k_vals_7, nb)[0]
        p_m = log_log_slope(k_vals_7, nm)[0]
        shift = p_m - p_b
        print(f"  Shift = {shift:.3f} (expected ≈ 0.45)")
        assert 0.3 < shift < 0.6, \
            f"Shift = {shift:.3f}, expected ∈ [0.3, 0.6]"

    def test_integrand_cv_stable_across_R(self, angular_grid):
        """MS integrand sin²(k)·σ_bond·N_eff has CV < 20% at all R=3..9.

        Must include σ_bond = C₀·V/v_g² for the cos²(k/2) cancellation.
        Investigation: CV ≈ 12% at all R. The flat integrand works
        at ALL R, not just R=5.
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        sb = np.array([sigma_bond_born(k, ALPHA_REF) for k in k_vals_7])
        cvs = []
        for R in [3, 5, 7, 9]:
            _, nm = _compute_neff_transport(R, k_vals_7, ALPHA_REF,
                                            TH, PH, sin_th, cos_th,
                                            transport)
            integrand = np.sin(k_vals_7)**2 * sb * nm
            integrand_n = integrand / np.mean(integrand)
            c = cv(integrand_n)
            cvs.append(c)
            print(f"  R={R}: integrand CV = {c:.1f}%")
        assert all(c < 20.0 for c in cvs), \
            f"Integrand CV > 20% at some R: {cvs}"

    def test_shift_positive_for_random_disk(self, angular_grid):
        """Random disk gives positive transport shift (not lattice-specific).

        Lattice shift ≈ 0.45, random disk shift ≈ 0.27 (weaker but positive).
        CV < 15% across 5 seeds. Links to test_3's TestSeedIndependence.
        """
        from helpers.geometry import random_disk
        TH, PH, sin_th, cos_th, transport = angular_grid
        V = V_eff(ALPHA_REF)
        N_lat = len(disk_bonds(5)[0])
        shifts = []
        for seed in range(5):
            dx, dy = random_disk(5, N_lat, seed=seed)
            N = len(dx)
            neff_b = np.zeros(len(k_vals_7))
            neff_m = np.zeros(len(k_vals_7))
            for ik, kv in enumerate(k_vals_7):
                G = build_G_matrix(dx, dy, kv)
                T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
                b = np.exp(1j * kv * dx)
                Tb = T @ b
                Vb = V * b
                phase_out = kv * (
                    sin_th[:, :, None] * np.cos(PH)[:, :, None]
                    * dx[None, None, :]
                    + sin_th[:, :, None] * np.sin(PH)[:, :, None]
                    * dy[None, None, :])
                a = np.exp(-1j * phase_out)
                f_ms = np.sum(a * Tb[None, None, :], axis=2)
                f_born = np.sum(a * Vb[None, None, :], axis=2)
                qz = kv * cos_th
                F2 = 4 * np.cos(qz / 2)**2
                w = F2 * transport * sin_th
                s_single = np.abs(V)**2 * np.sum(w)
                neff_b[ik] = np.sum(np.abs(f_born)**2 * w) / s_single
                neff_m[ik] = np.sum(np.abs(f_ms)**2 * w) / s_single
            shift = log_log_slope(k_vals_7, neff_m)[0] \
                - log_log_slope(k_vals_7, neff_b)[0]
            shifts.append(shift)
            print(f"  seed={seed}: shift={shift:.3f}")
        shifts = np.array(shifts)
        print(f"  Mean={np.mean(shifts):.3f}, CV={cv(shifts):.1f}%")
        assert np.all(shifts > 0), \
            f"Negative shift at some seed: {shifts}"
        assert cv(shifts) < 15.0, \
            f"Shift CV = {cv(shifts):.1f}%, expected < 15%"

    def test_integrand_cv_stable_at_higher_resolution(self, angular_grid):
        """Integrand CV at 15 k-points matches 7 k-points (< 20%).

        Flatness is not an artifact of sparse k-sampling.
        CV = 12.5% at 7 pts, 12.6% at 15 pts — essentially identical.
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        V = V_eff(ALPHA_REF)
        dx, dy = disk_bonds(5)
        N = len(dx)
        k_dense = np.linspace(0.3, 1.5, 15)
        sb_d = np.array([sigma_bond_born(k, ALPHA_REF) for k in k_dense])
        neff_m = np.zeros(len(k_dense))
        for ik, kv in enumerate(k_dense):
            G = build_G_matrix(dx, dy, kv)
            T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
            b = np.exp(1j * kv * dx)
            Tb = T @ b
            phase_out = kv * (
                sin_th[:, :, None] * np.cos(PH)[:, :, None]
                * dx[None, None, :]
                + sin_th[:, :, None] * np.sin(PH)[:, :, None]
                * dy[None, None, :])
            a = np.exp(-1j * phase_out)
            f_ms = np.sum(a * Tb[None, None, :], axis=2)
            qz = kv * cos_th
            F2 = 4 * np.cos(qz / 2)**2
            w = F2 * transport * sin_th
            s_single = np.abs(V)**2 * np.sum(w)
            neff_m[ik] = np.sum(np.abs(f_ms)**2 * w) / s_single
        integrand = np.sin(k_dense)**2 * sb_d * neff_m
        cv_15 = cv(integrand / np.mean(integrand))
        print(f"  CV at 15 k-points: {cv_15:.1f}%")
        assert cv_15 < 20.0, \
            f"CV = {cv_15:.1f}% at 15 k-points, expected < 20%"

    def test_shift_proportional_to_V(self, angular_grid):
        """Transport shift Δp = C·|V| with C ≈ 0.334, CV < 10%.

        p_Born is α-independent (V² cancels in N_eff). The shift comes
        entirely from p_MS. C = shift/|V| is nearly constant across
        α = 0.15–0.40: C = 0.316–0.351 (CV = 3.7%).
        Directly citeable as Δp = C|V|.
        """
        TH, PH, sin_th, cos_th, transport = angular_grid
        C_vals = []
        for alpha in [0.15, 0.20, 0.25, 0.30, 0.40]:
            nb, nm = _compute_neff_transport(5, k_vals_7, alpha,
                                             TH, PH, sin_th, cos_th,
                                             transport)
            shift = log_log_slope(k_vals_7, nm)[0] \
                - log_log_slope(k_vals_7, nb)[0]
            V = V_eff(alpha)
            C = shift / abs(V)
            C_vals.append(C)
            print(f"  α={alpha}: shift={shift:+.3f}, |V|={abs(V):.3f}, "
                  f"C={C:.3f}")
        C_arr = np.array(C_vals)
        C_cv = cv(C_arr)
        print(f"  C mean={np.mean(C_arr):.3f}, CV={C_cv:.1f}%")
        assert C_cv < 10.0, \
            f"C = shift/|V| CV = {C_cv:.1f}%, expected < 10%"


class TestGeometryDependence:
    """F7: C coefficient depends on geometry — local coordination matters.

    C_disk ≈ 0.34, C_annulus ≈ 0.12, C_line ≈ 0.09.
    Disk C is ~3.5× larger than line C. Roughly proportional to the
    number of neighbors within cooperation range (r≤3).

    Ref: internal/70_forward_decoherence_investigation.md, F7.
    """

    def test_disk_larger_than_line(self):
        """Disk C > line C by at least factor 2.

        Investigation: disk C = 0.342, line C = 0.092 (ratio 3.7×).
        Uses fixed line length L=9 (matching investigation), not N_disk.
        """
        R = 5
        dx_d, dy_d = disk_bonds(R)
        p_d, _ = _enhancement_exponent(dx_d, dy_d, k_vals_7)

        dx_l, dy_l = line_bonds(9)
        p_l, _ = _enhancement_exponent(dx_l, dy_l, k_vals_7)

        V = abs(V_eff(ALPHA_REF))
        C_disk = p_d / V
        C_line = p_l / V

        print(f"  C_disk = {C_disk:.3f}, C_line = {C_line:.3f}, "
              f"ratio = {C_disk/C_line:.1f}×")
        assert C_disk > 2 * C_line, \
            f"C_disk/C_line = {C_disk/C_line:.1f}, expected > 2"

    def test_disk_larger_than_annulus(self):
        """Disk C > annulus C (interior bonds contribute to cooperation).

        Investigation: disk C = 0.342, annulus C = 0.109 at R=5.
        """
        R = 5
        dx_d, dy_d = disk_bonds(R)
        p_d, _ = _enhancement_exponent(dx_d, dy_d, k_vals_7)

        dx_a, dy_a = annulus_bonds(R, width=1)
        p_a, _ = _enhancement_exponent(dx_a, dy_a, k_vals_7)

        V = abs(V_eff(ALPHA_REF))
        C_disk = p_d / V
        C_ann = p_a / V

        print(f"  C_disk = {C_disk:.3f}, C_annulus = {C_ann:.3f}, "
              f"ratio = {C_disk/C_ann:.1f}×")
        assert C_disk > C_ann, \
            f"C_disk = {C_disk:.3f} ≤ C_annulus = {C_ann:.3f}"

    def test_ordering_disk_annulus_line(self):
        """C_disk > C_annulus > C_line (coordination hierarchy).

        Uses fixed line length L=9 (matching investigation).
        """
        R = 5
        V = abs(V_eff(ALPHA_REF))

        dx_d, dy_d = disk_bonds(R)
        C_disk = _enhancement_exponent(dx_d, dy_d, k_vals_7)[0] / V

        dx_a, dy_a = annulus_bonds(R, width=1)
        C_ann = _enhancement_exponent(dx_a, dy_a, k_vals_7)[0] / V

        dx_l, dy_l = line_bonds(9)
        C_line = _enhancement_exponent(dx_l, dy_l, k_vals_7)[0] / V

        print(f"  C_disk={C_disk:.3f} > C_ann={C_ann:.3f} > "
              f"C_line={C_line:.3f}")
        assert C_disk > C_ann > C_line, \
            f"Ordering violated: {C_disk:.3f}, {C_ann:.3f}, {C_line:.3f}"


class TestCFullVsCStatic:
    """F8: Phase oscillation makes C universal (R-independent).

    Transport-weighted C (physical phases): stable at 0.32 ± 0.02.
    Total power C (physical): C_full < C_static (phases reduce range).

    The oscillating phases limit the cooperation range to r ≈ 3,
    making C a local property. Without phases, distant bonds keep
    contributing.

    The R-independence test of C_full is covered by
    TestTransportShiftRIndependence. Here we test the comparison
    between physical (exp(ikr)/r) and static (1/r) Green functions.

    Ref: internal/70_forward_decoherence_investigation.md, F8.
    """

    @staticmethod
    def _enh_exp_custom(dx, dy, k_arr, use_phase=True, alpha=ALPHA_REF):
        """Enhancement exponent with or without oscillating phases.

        Matches test_3's _enh_exp_custom: includes G₀₀ from BZ sum.
        use_phase only affects the OFF-DIAGONAL propagator. G₀₀ stays
        complex (from BZ sum, with Im < 0) in both modes — "static"
        means no off-diagonal phase oscillation, not fully real G.
        Conclusion C_full < C_static is valid since only the off-diagonal
        differs between branches.
        """
        from helpers.lattice import dispersion_sq, get_omega_k2
        from helpers.config import EPS_LAT
        N = len(dx)
        V = V_eff(alpha)
        omega_k2 = get_omega_k2()
        dist = np.sqrt((dx[:, None] - dx[None, :])**2
                       + (dy[:, None] - dy[None, :])**2)
        enh = np.zeros(len(k_arr))
        for ik, kv in enumerate(k_arr):
            ke = k_eff(kv)
            G = np.zeros((N, N), dtype=complex)
            m = dist > 0
            if use_phase:
                G[m] = np.exp(1j * ke * dist[m]) / (
                    4 * np.pi * c_lat**2 * dist[m])
            else:
                G[m] = 1.0 / (4 * np.pi * c_lat**2 * dist[m])
            omega2 = dispersion_sq(kv)
            G_00 = np.mean(1.0 / (omega2 - omega_k2 + 1j * EPS_LAT))
            np.fill_diagonal(G, G_00)

            T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
            b = np.exp(1j * kv * dx)
            Tb = T @ b
            Vb = V * b
            enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
        return log_log_slope(k_arr, enh)[0]

    def test_c_full_smaller_than_static(self):
        """C_full < C_static at all R (phases reduce effective range).

        Oscillating phases cancel distant contributions → smaller
        enhancement → smaller C.
        """
        V = abs(V_eff(ALPHA_REF))
        for R in [3, 5, 7, 9]:
            dx, dy = disk_bonds(R)
            C_full = self._enh_exp_custom(dx, dy, k_vals_7,
                                          use_phase=True) / V
            C_stat = self._enh_exp_custom(dx, dy, k_vals_7,
                                          use_phase=False) / V
            print(f"  R={R}: C_full={C_full:.3f}, C_static={C_stat:.3f}")
            assert C_full < C_stat, \
                f"R={R}: C_full={C_full:.3f} ≥ C_static={C_stat:.3f}"

    def test_static_ratio_significant_at_all_R(self):
        """C_static/C_full > 1.3 at all R (phases significantly reduce C).

        The ratio quantifies how much distant-bond cancellation the
        oscillating phases provide. Significant at all tested R.
        """
        V = abs(V_eff(ALPHA_REF))
        for R in [3, 5, 7, 9]:
            dx, dy = disk_bonds(R)
            C_full = self._enh_exp_custom(dx, dy, k_vals_7,
                                          use_phase=True) / V
            C_stat = self._enh_exp_custom(dx, dy, k_vals_7,
                                          use_phase=False) / V
            ratio = C_stat / C_full
            print(f"  R={R}: C_static/C_full = {ratio:.2f}")
            assert ratio > 1.3, \
                f"R={R}: ratio = {ratio:.2f}, expected > 1.3"

    def test_physical_converges_with_range(self):
        """Physical G: r_cut=3 gives enhancement within 5% of full range.

        Cooperation range is ~3 lattice spacings with oscillating phases.
        """
        dx, dy = disk_bonds(5)
        p_near = self._enh_exp_custom_rcut(dx, dy, k_vals_7,
                                           use_phase=True, r_cut=3)
        p_full = self._enh_exp_custom_rcut(dx, dy, k_vals_7,
                                           use_phase=True, r_cut=100)
        rel = abs(p_near - p_full) / abs(p_full)
        print(f"  r_cut=3: p={p_near:.3f}, full: p={p_full:.3f}, "
              f"|Δp/p|={rel:.1%}")
        assert rel < 0.05, \
            f"|Δp/p| = {rel:.3f}, r_cut=3: {p_near:.3f}, full: {p_full:.3f}"

    def test_static_does_not_converge(self):
        """Static G: enhancement at r_cut=5 > r_cut=3 by > 15%.

        Without phases, distant bonds keep contributing → no convergence.
        """
        dx, dy = disk_bonds(5)
        p_3 = self._enh_exp_custom_rcut(dx, dy, k_vals_7,
                                        use_phase=False, r_cut=3)
        p_5 = self._enh_exp_custom_rcut(dx, dy, k_vals_7,
                                        use_phase=False, r_cut=5)
        growth = (p_5 - p_3) / abs(p_3)
        print(f"  static r_cut=3: {p_3:.3f}, r_cut=5: {p_5:.3f}, "
              f"growth={growth:.1%}")
        assert growth > 0.15, \
            f"Static growth = {growth:.1%}, expected > 15%"

    @staticmethod
    def _enh_exp_custom_rcut(dx, dy, k_arr, use_phase=True, r_cut=100,
                             alpha=ALPHA_REF):
        """Enhancement exponent with range cutoff."""
        from helpers.lattice import dispersion_sq, get_omega_k2
        from helpers.config import EPS_LAT
        N = len(dx)
        V = V_eff(alpha)
        omega_k2 = get_omega_k2()
        dist = np.sqrt((dx[:, None] - dx[None, :])**2
                       + (dy[:, None] - dy[None, :])**2)
        enh = np.zeros(len(k_arr))
        for ik, kv in enumerate(k_arr):
            ke = k_eff(kv)
            G = np.zeros((N, N), dtype=complex)
            m = (dist > 0) & (dist <= r_cut)
            if use_phase:
                G[m] = np.exp(1j * ke * dist[m]) / (
                    4 * np.pi * c_lat**2 * dist[m])
            else:
                G[m] = 1.0 / (4 * np.pi * c_lat**2 * dist[m])
            omega2 = dispersion_sq(kv)
            G_00 = np.mean(1.0 / (omega2 - omega_k2 + 1j * EPS_LAT))
            np.fill_diagonal(G, G_00)

            T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
            b = np.exp(1j * kv * dx)
            Tb = T @ b
            Vb = V * b
            enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
        return log_log_slope(k_arr, enh)[0]


class TestMeanFieldCoupling:
    """F9: Mean-field coupling explains k-dependent suppression.

    |V·⟨Σ G_ij⟩| decreases with k: strong at k=0.3 (coupling ≈ 0.67),
    weak at k=1.5 (coupling ≈ 0.16). The oscillating phases cancel
    more at high k, reducing the effective coupling.

    This k-dependent coupling → k-dependent suppression → exponent shift.

    All spectral radii < 1: system is NOT resonant.

    Ref: internal/70_forward_decoherence_investigation.md, F9.
    """

    def test_coupling_decreases_with_k(self):
        """Mean-field coupling at k=0.3 > coupling at k=1.5.

        Investigation: 0.67 (k=0.3) → 0.16 (k=1.5).
        """
        dx, dy = disk_bonds(5)
        V = V_eff(ALPHA_REF)
        N = len(dx)

        couplings = []
        for kv in [0.3, 0.9, 1.5]:
            G = build_G_matrix(dx, dy, kv)
            # Mean off-diagonal G sum per bond
            G_sum = np.mean(np.sum(G, axis=1) - np.diag(G))
            coupling = abs(V * G_sum)
            couplings.append(coupling)
            print(f"  k={kv}: |V·⟨ΣG⟩| = {coupling:.3f}")

        assert couplings[0] > couplings[-1], \
            f"Coupling not decreasing: {couplings[0]:.3f} → {couplings[-1]:.3f}"

    def test_no_resonance(self):
        """All eigenvalues of V·G have |λ| < 1 (no resonance).

        Investigation: spectral radius = 0.60 at k=0.3, R=5.
        """
        dx, dy = disk_bonds(5)
        for kv in [0.3, 0.9, 1.5]:
            eigs = eigenvalues_VG(dx, dy, kv, ALPHA_REF)
            spec_radius = np.max(np.abs(eigs))
            print(f"  k={kv}: spectral radius = {spec_radius:.3f}")
            assert spec_radius < 1.0, \
                f"Resonance at k={kv}: |λ_max| = {spec_radius:.3f} ≥ 1"

    def test_coupling_power_law(self):
        """Coupling decreases as ~k^{-p} with p ∈ [0.5, 1.5].

        Investigation: p ≈ 0.9.
        """
        dx, dy = disk_bonds(5)
        V = V_eff(ALPHA_REF)

        couplings = []
        for kv in k_vals_7:
            G = build_G_matrix(dx, dy, kv)
            G_sum = np.mean(np.sum(G, axis=1) - np.diag(G))
            couplings.append(abs(V * G_sum))

        p, _ = log_log_slope(k_vals_7, np.array(couplings))
        print(f"  Coupling power law: p = {p:.2f}")
        assert -1.5 < p < -0.5, \
            f"Coupling exponent p = {p:.2f}, expected ∈ [-1.5, -0.5]"


class TestPowerDecomposition:
    """F10: Total power decomposition and exponent chain.

    total_pow = diag_pow + offdiag_pow + cross.
    cross/born = 2 × (diag_amp/√born) × (offdiag_amp/√born) × cos(Φ).
    Exponents sum: diag_amp (+0.01) + offdiag_amp (-0.48) + cos (-0.16) = -0.63.
    offdiag_amp ≈ √(N_coop × incoh) ≈ k^{-1/2} at moderate R.

    Ref: internal/70_forward_decoherence_investigation.md, F10.
    """

    @staticmethod
    def _decompose(dx, dy, k_arr, alpha=ALPHA_REF):
        """Compute diag_amp, offdiag_amp, cos(Φ), cross, enh at each k."""
        V = V_eff(alpha)
        N = len(dx)
        result = {'diag_amp': [], 'offdiag_amp': [], 'cos_phi': [],
                  'cross': [], 'enh': []}
        for kv in k_arr:
            b = np.exp(1j * kv * dx)
            G = build_G_matrix(dx, dy, kv)
            T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
            Tb = T @ b
            Vb = V * b

            diag_part = np.diag(T) * b
            offdiag_part = Tb - diag_part

            born_pow = np.sum(np.abs(Vb)**2)
            diag_pow = np.sum(np.abs(diag_part)**2)
            offdiag_pow = np.sum(np.abs(offdiag_part)**2)
            cross = 2 * np.real(np.sum(np.conj(diag_part) * offdiag_part))

            diag_amp = np.sqrt(diag_pow / born_pow)
            offdiag_amp = np.sqrt(offdiag_pow / born_pow)
            cos_phi = cross / (2 * np.sqrt(diag_pow) * np.sqrt(offdiag_pow))

            result['diag_amp'].append(diag_amp)
            result['offdiag_amp'].append(offdiag_amp)
            result['cos_phi'].append(cos_phi)
            result['cross'].append(abs(cross / born_pow))
            result['enh'].append((diag_pow + offdiag_pow + cross) / born_pow)
        return {k: np.array(v) for k, v in result.items()}

    def test_exponents_sum_correctly(self):
        """cross exponent = diag_amp + offdiag_amp + cos(Φ) exponents (±0.05).

        cross/born = 2 × diag_amp × offdiag_amp × cos(Φ), so their log-log
        slopes must add.
        """
        dx, dy = disk_bonds(5)
        d = self._decompose(dx, dy, k_vals_7)
        p_da = log_log_slope(k_vals_7, d['diag_amp'])[0]
        p_oa = log_log_slope(k_vals_7, d['offdiag_amp'])[0]
        p_cp = log_log_slope(k_vals_7, np.abs(d['cos_phi']))[0]
        p_cr = log_log_slope(k_vals_7, d['cross'])[0]
        predicted = p_da + p_oa + p_cp
        err = abs(predicted - p_cr)
        print(f"  diag_amp={p_da:+.3f}, offdiag_amp={p_oa:+.3f}, "
              f"cos(Φ)={p_cp:+.3f}")
        print(f"  Sum={predicted:+.3f}, cross={p_cr:+.3f}, error={err:.3f}")
        assert err < 0.05, \
            f"Exponent sum {predicted:.3f} ≠ cross {p_cr:.3f} (Δ={err:.3f})"

    def test_offdiag_amp_near_minus_half(self):
        """offdiag_amp exponent ∈ [-0.6, -0.35] at R=5.

        offdiag_amp = √(N_coop × incoh) ≈ k^{-1/2} at moderate R.
        """
        dx, dy = disk_bonds(5)
        d = self._decompose(dx, dy, k_vals_7)
        p = log_log_slope(k_vals_7, d['offdiag_amp'])[0]
        print(f"  offdiag_amp exponent: {p:+.3f} (expected ≈ -0.48)")
        assert -0.6 < p < -0.35, \
            f"offdiag_amp exponent {p:.3f} outside [-0.6, -0.35]"

    def test_enh_positive_slope(self):
        """Enhancement exponent > 0 (MS flattens the spectrum).

        enh = Σ|Tb|²/Σ|Vb|² grows with k → positive exponent → shift.
        """
        dx, dy = disk_bonds(5)
        d = self._decompose(dx, dy, k_vals_7)
        p = log_log_slope(k_vals_7, d['enh'])[0]
        print(f"  enh exponent: {p:+.3f} (expected ≈ +0.35)")
        assert p > 0.15, \
            f"enh exponent {p:.3f}, expected > 0.15"

    def test_cross_negative_at_all_k(self):
        """Cross term cos(Φ) < 0 at all k (always destructive).

        The off-diagonal rescattering always opposes the diagonal.
        """
        dx, dy = disk_bonds(5)
        d = self._decompose(dx, dy, k_vals_7)
        print(f"  cos(Φ) range: [{np.min(d['cos_phi']):.3f}, "
              f"{np.max(d['cos_phi']):.3f}]")
        assert np.all(d['cos_phi'] < 0), \
            f"cos(Φ) ≥ 0 at some k: {d['cos_phi']}"

    def test_crossover_k_grows_with_R(self):
        """Crossover k (where enh=1) grows with R.

        Small disk: crossover at k≈0.6 (suppression only at low k).
        Large disk: crossover at k≈1.5 (suppression extends to high k).
        Crossover monotonically increases: more bonds → wider suppression.
        """
        k_fine = np.linspace(0.3, 1.5, 50)
        crossovers = []
        for R in [3, 5, 7, 9]:
            dx, dy = disk_bonds(R)
            _, enh = _enhancement_exponent(dx, dy, k_fine)
            cross_k = None
            for i in range(len(enh) - 1):
                if enh[i] < 1 and enh[i + 1] >= 1:
                    cross_k = k_fine[i] + (k_fine[i + 1] - k_fine[i]) * \
                        (1 - enh[i]) / (enh[i + 1] - enh[i])
                    break
            crossovers.append(cross_k)
            print(f"  R={R}: crossover_k = "
                  f"{'%.3f' % cross_k if cross_k else '>1.5'}")
        # R=3, R=5, R=7 must have crossover in [0.3, 1.5]
        for i, R in enumerate([3, 5, 7]):
            assert crossovers[i] is not None, \
                f"No crossover at R={R} in [0.3, 1.5]"
        # Monotonicity: each found crossover must be larger than previous
        found = [(R, ck) for R, ck in zip([3, 5, 7, 9], crossovers)
                 if ck is not None]
        for i in range(len(found) - 1):
            assert found[i][1] < found[i + 1][1], \
                f"Crossover not growing: R={found[i][0]}→{found[i][1]:.3f}, " \
                f"R={found[i+1][0]}→{found[i+1][1]:.3f}"


class TestBornSeriesAlternation:
    """F11: Born series alternation drives incoh k-dependence.

    |G_ij|² is k-independent. The k-dependence of incoh (per-element
    |T_ij|²) comes from the resolvent structure: at low k, VG elements
    are coherent and Born series terms cancel; at high k, they add
    constructively.

    cancel_ratio = <|R_ij|> / <Σ|(VG)^n_ij|> measures this:
    0.37 at k=0.3 (strong cancellation) vs 2.0 at k=1.5.

    Ref: internal/70_forward_decoherence_investigation.md, F11.
    """

    def test_G_incoh_k_independent(self):
        """|G_ij|² off-diagonal is k-independent (exponent ≈ 0).

        The bare Green function magnitude 1/(4πc²r) has no k-dependence;
        only the phase exp(ikr) depends on k. So |G_ij|² = 1/(4πc²r)² =
        const for each pair.
        """
        dx, dy = disk_bonds(5)
        N = len(dx)
        mask = ~np.eye(N, dtype=bool)
        incoh_G = []
        for kv in k_vals_7:
            G = build_G_matrix(dx, dy, kv)
            incoh_G.append(np.mean(np.abs(G[mask])**2))
        p = log_log_slope(k_vals_7, np.array(incoh_G))[0]
        print(f"  |G_ij|² exponent: {p:+.4f} (expected ≈ 0)")
        assert abs(p) < 0.05, \
            f"|G_ij|² exponent = {p:.4f}, expected ≈ 0"

    def test_cancel_ratio_increases_with_k(self):
        """Cancel ratio grows from low k to high k.

        At low k: strong cancellation (ratio < 0.5).
        At high k: constructive addition (ratio > 0.8).
        """
        dx, dy = disk_bonds(5)
        N = len(dx)
        V = V_eff(ALPHA_REF)
        mask = ~np.eye(N, dtype=bool)

        ratios = []
        for kv in [0.3, 1.5]:
            G = build_G_matrix(dx, dy, kv)
            VG = V * G
            resolvent = np.linalg.inv(np.eye(N) - VG)
            R_mean = np.mean(np.abs(resolvent[mask]))

            VG_power = np.eye(N, dtype=complex)
            sum_abs = np.zeros(N * N).reshape(N, N)
            for n in range(12):
                VG_power = VG_power @ VG
                sum_abs += np.abs(VG_power)
            SA_mean = np.mean(sum_abs[mask])
            ratios.append(R_mean / SA_mean)

        print(f"  cancel_ratio: k=0.3 → {ratios[0]:.3f}, "
              f"k=1.5 → {ratios[1]:.3f}")
        assert ratios[0] < ratios[1], \
            f"Cancel ratio not increasing: {ratios[0]:.3f} → {ratios[1]:.3f}"
        assert ratios[0] < 0.5, \
            f"Low-k ratio {ratios[0]:.3f} not < 0.5 (weak cancellation)"

    def test_VG_coherence_decreases_with_k(self):
        """VG off-diagonal coherence drops from high (k=0.3) to low (k=1.5).

        Coherence = |mean(VG_offdiag)| / mean(|VG_offdiag|).
        At low k, phases aligned → coherence ≈ 0.8.
        At high k, phases random → coherence ≈ 0.2.
        """
        dx, dy = disk_bonds(5)
        N = len(dx)
        V = V_eff(ALPHA_REF)
        mask = ~np.eye(N, dtype=bool)

        cohs = []
        for kv in [0.3, 1.5]:
            G = build_G_matrix(dx, dy, kv)
            VG = V * G
            offdiag = VG[mask]
            coh = np.abs(np.mean(offdiag)) / np.mean(np.abs(offdiag))
            cohs.append(coh)

        print(f"  VG coherence: k=0.3 → {cohs[0]:.3f}, k=1.5 → {cohs[1]:.3f}")
        assert cohs[0] > 2 * cohs[1], \
            f"Coherence ratio {cohs[0]/cohs[1]:.1f}×, expected > 2×"

    def test_incoh_grows_with_k(self):
        """Per-element |T_ij|² grows with k (positive exponent).

        incoh exponent ≈ +0.48 at R=5. This is the k-dependence from
        the resolvent structure, not from |G_ij|² (which is constant).
        """
        dx, dy = disk_bonds(5)
        N = len(dx)
        V = V_eff(ALPHA_REF)
        mask = ~np.eye(N, dtype=bool)

        incoh_vals = []
        for kv in k_vals_7:
            G = build_G_matrix(dx, dy, kv)
            T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
            incoh_vals.append(np.mean(np.abs(T[mask])**2))

        p = log_log_slope(k_vals_7, np.array(incoh_vals))[0]
        print(f"  incoh exponent: {p:+.3f} (expected ≈ +0.48)")
        assert p > 0.3, \
            f"incoh exponent {p:.3f}, expected > 0.3"

    def test_eigenvalue_clustering_decreases_with_k(self):
        """VG eigenvalue magnitudes cluster at low k, spread at high k.

        At k=0.3: one dominant eigenvalue (|λ|_max ≈ 0.60), rest smaller,
        |λ| CV ≈ 41%. At k=1.5: eigenvalues spread more uniformly,
        |λ| CV ≈ 24%. The decreasing CV reflects phase decoherence:
        at low k, all bonds contribute coherently to one dominant mode.
        """
        dx, dy = disk_bonds(5)
        N = len(dx)
        V = V_eff(ALPHA_REF)

        cvs = []
        for kv in [0.3, 1.5]:
            G = build_G_matrix(dx, dy, kv)
            eigs = np.linalg.eigvals(V * G)
            mag = np.abs(eigs)
            c = np.std(mag) / np.mean(mag)
            cvs.append(c)
            print(f"  k={kv}: |λ| CV={100*c:.1f}%, "
                  f"max={np.max(mag):.3f}, min={np.min(mag):.4f}")

        assert cvs[0] > cvs[1], \
            f"|λ| CV at k=0.3 ({cvs[0]:.3f}) not > k=1.5 ({cvs[1]:.3f})"
        assert cvs[0] > 0.30, \
            f"|λ| CV at k=0.3 = {cvs[0]:.3f}, expected > 0.30 (clustered)"


class TestCosPhiConvergence:
    """F12: cos(Φ) exponent → 0 at large R.

    The effective phase between diagonal and off-diagonal N-vectors
    becomes k-independent at large R. This simplifies the cross term
    chain: cross ≈ diag_amp × offdiag_amp (with constant prefactor).

    |cos(Φ)| exponent: -0.35 (R=3) → -0.16 (R=5) → -0.05 (R=9).

    Ref: internal/70_forward_decoherence_investigation.md, F12.
    """

    def test_cos_phi_exponent_magnitude_decreases_with_R(self):
        """|cos(Φ)| exponent approaches 0 as R grows.

        At R=3 the phase has significant k-dependence; by R=9 it's
        nearly constant. Tests that |p_cosΦ(R=9)| < |p_cosΦ(R=3)|.
        """
        V = V_eff(ALPHA_REF)
        exponents = []
        for R in [3, 9]:
            dx, dy = disk_bonds(R)
            N = len(dx)
            cos_vals = []
            for kv in k_vals_7:
                b = np.exp(1j * kv * dx)
                G = build_G_matrix(dx, dy, kv)
                T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
                Tb = T @ b
                d = np.diag(T) * b
                o = Tb - d
                dp = np.sum(np.abs(d)**2)
                op = np.sum(np.abs(o)**2)
                cr = 2 * np.real(np.sum(np.conj(d) * o))
                cos_vals.append(abs(cr / (2 * np.sqrt(dp) * np.sqrt(op))))
            p = log_log_slope(k_vals_7, np.array(cos_vals))[0]
            exponents.append(p)
            print(f"  R={R}: |cos(Φ)| exponent = {p:+.3f}")

        assert abs(exponents[1]) < abs(exponents[0]), \
            f"|p_cosΦ| at R=9 ({abs(exponents[1]):.3f}) not < R=3 " \
            f"({abs(exponents[0]):.3f})"

    def test_cos_phi_exponent_small_at_R9(self):
        """|cos(Φ)| exponent magnitude < 0.15 at R=9.

        By R=9, the phase is nearly k-independent.
        """
        V = V_eff(ALPHA_REF)
        dx, dy = disk_bonds(9)
        N = len(dx)
        cos_vals = []
        for kv in k_vals_7:
            b = np.exp(1j * kv * dx)
            G = build_G_matrix(dx, dy, kv)
            T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
            Tb = T @ b
            d = np.diag(T) * b
            o = Tb - d
            dp = np.sum(np.abs(d)**2)
            op = np.sum(np.abs(o)**2)
            cr = 2 * np.real(np.sum(np.conj(d) * o))
            cos_vals.append(abs(cr / (2 * np.sqrt(dp) * np.sqrt(op))))
        p = log_log_slope(k_vals_7, np.array(cos_vals))[0]
        print(f"  R=9: |cos(Φ)| exponent = {p:+.3f} (expected ≈ 0)")
        assert abs(p) < 0.15, \
            f"|cos(Φ)| exponent at R=9 = {abs(p):.3f}, expected < 0.15"


class TestCross12SignChange:
    """F13: cross_12 = 2Re<VG*·VG²> switches sign with k.

    The leading Born correction to <|R_ij|²> is the cross_12 term.
    It involves the 3-point correlator G_ij*G_il*G_lj with phase
    k_eff × (r_il + r_lj - r_ij) = k_eff × path_excess.

    At low k: cross_12 < 0 (destructive, suppresses incoh).
    At high k: cross_12 > 0 (constructive, enhances incoh).

    This sign change is the microscopic origin of incoh's k-dependence.

    Ref: internal/70_forward_decoherence_investigation.md, F13.
    """

    @staticmethod
    def _cross12(dx, dy, kv, alpha=ALPHA_REF):
        """Compute cross_12 = 2Re<VG*·VG²> and |VG|² (off-diagonal)."""
        N = len(dx)
        V = V_eff(alpha)
        G = build_G_matrix(dx, dy, kv)
        VG = V * G
        VG2 = VG @ VG
        mask = ~np.eye(N, dtype=bool)
        term1 = np.mean(np.abs(VG[mask])**2)
        cross12 = 2 * np.mean(np.real(VG[mask].conj() * VG2[mask]))
        return cross12, term1

    def test_cross12_negative_at_low_k(self):
        """cross_12 < 0 at k=0.3 for R=5 (destructive Born correction).

        The path excess phases are small → systematic alternation →
        destructive interference.
        """
        dx, dy = disk_bonds(5)
        c12, t1 = self._cross12(dx, dy, 0.3)
        ratio = c12 / t1
        print(f"  cross_12/|VG|² = {ratio:+.3f} at k=0.3 (expected < 0)")
        assert c12 < 0, \
            f"cross_12 = {c12:.6f} ≥ 0 at k=0.3"

    def test_cross12_positive_at_high_k(self):
        """cross_12 > 0 at k=1.5 for R=5 (constructive Born correction).

        Phases randomize → no alternation → constructive addition.
        """
        dx, dy = disk_bonds(5)
        c12, t1 = self._cross12(dx, dy, 1.5)
        ratio = c12 / t1
        print(f"  cross_12/|VG|² = {ratio:+.3f} at k=1.5 (expected > 0)")
        assert c12 > 0, \
            f"cross_12 = {c12:.6f} ≤ 0 at k=1.5"

    def test_cross12_sign_changes(self):
        """cross_12 changes sign between k=0.3 and k=1.5 at R ≥ 5.

        At R=3, cross_12 is positive at all k (too small for destructive
        phase). The sign change appears only for R ≥ 5.
        """
        for R in [5, 7, 9]:
            dx, dy = disk_bonds(R)
            c_low, _ = self._cross12(dx, dy, 0.3)
            c_high, _ = self._cross12(dx, dy, 1.5)
            print(f"  R={R}: cross_12(k=0.3)={c_low:+.6f}, "
                  f"cross_12(k=1.5)={c_high:+.6f}")
            assert c_low < 0 < c_high, \
                f"R={R}: no sign change: low={c_low:.6f}, high={c_high:.6f}"

    def test_cross12_fraction_large_at_high_k(self):
        """cross_12/|VG|² > 0.5 at k=1.5 (dominant correction).

        The Born correction is a large fraction of the first-order term,
        showing that higher-order rescattering is significant.
        """
        dx, dy = disk_bonds(5)
        c12, t1 = self._cross12(dx, dy, 1.5)
        ratio = c12 / t1
        print(f"  cross_12/|VG|² = {ratio:.3f} at k=1.5 (expected > 0.5)")
        assert ratio > 0.5, \
            f"cross_12/|VG|² = {ratio:.3f}, expected > 0.5"

    def test_path_excess_phase_fraction_grows(self):
        """Fraction of triplet paths with phase > π grows from low to high k.

        cross_12 involves 3-point correlator with phase k_eff × path_excess
        where path_excess = r_il + r_lj - r_ij ≥ 0 (triangle inequality).
        At low k: 7.5% of phases exceed π → coherent, negative cross_12.
        At high k: 68% exceed π → destructive for coherence, positive cross_12.
        This geometric phase distribution is the microscopic origin of the
        cross_12 sign change.
        """
        from helpers.lattice import k_eff as ke_func
        dx, dy = disk_bonds(5)
        N = len(dx)
        dist = np.sqrt((dx[:, None] - dx[None, :])**2
                       + (dy[:, None] - dy[None, :])**2)

        # pe[i,j,l] = dist[i,l] + dist[l,j] - dist[i,j]  (axes: i,j,l)
        pe_3d = (dist[:, None, :]       # dist[i,l]: (N,1,N)
                 + dist[None, :, :]      # dist[j,l]: (1,N,N) — symmetric so = dist[l,j]
                 - dist[:, :, None])     # dist[i,j]: (N,N,1)
        ii = np.arange(N)
        mask = ((ii[:, None, None] != ii[None, :, None])    # i≠j
                & (ii[:, None, None] != ii[None, None, :])  # i≠l
                & (ii[None, :, None] != ii[None, None, :])) # j≠l
        pe = pe_3d[mask]

        fracs = []
        for kv in [0.3, 1.5]:
            ke = ke_func(kv)
            frac_gt_pi = np.mean(ke * pe > np.pi)
            fracs.append(frac_gt_pi)
            print(f"  k={kv}: frac(k_eff*pe > π) = {100*frac_gt_pi:.1f}%")

        assert fracs[1] > fracs[0], \
            f"Phase fraction not growing: {fracs[0]:.3f} → {fracs[1]:.3f}"
        assert fracs[0] < 0.15, \
            f"Low-k phase fraction {fracs[0]:.3f} too high (expected < 0.15)"
        assert fracs[1] > 0.50, \
            f"High-k phase fraction {fracs[1]:.3f} too low (expected > 0.50)"
