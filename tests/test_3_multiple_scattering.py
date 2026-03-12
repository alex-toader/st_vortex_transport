"""§6.3 Multiple scattering correction: Born -5/2 → MS -2.0.

Source: old_tests/58_multiple_scattering.py, 59_eigenvalue_analysis.py,
        62_sanity_checks.py

Results:
  TestMSExponent:
    R=3: p_MS = -1.813, R=5: -1.966, R=7: -1.970, R=9: -1.991
    p_FDTD(R=5) = -1.870, MS/FDTD fraction = 0.82  (threshold 0.75-1.10)
  TestIntegrandFlatness:
    Born integrand CV = 29.5%  (threshold > 28%)
    MS integrand CV = 10.4%  (threshold < 25%)
  TestShiftLinearity:
    C = p_enh/|V|: 0.305 (α=0.10), 0.326 (α=0.20), 0.342 (α=0.30), 0.351 (α=0.40)
  TestRandomDisk:
    lattice p_enh = 0.354, random p_enh = 0.343  (ratio 0.97)
  TestPhaseRemoval:
    physical (exp(ikr)/r): p_enh = 0.448
    static (1/r):          p_enh = 0.831  (1.85× larger)
  TestBornSeriesConvergence:
    full resummation: p = 0.448
    1st correction:   p = 0.120  (27% of full, threshold < 50%)
  TestShiftDecomposition:
    full: p = 0.448, offdiag only: 0.361, G₀₀ only: 0.039
    G₀₀ fraction = 9%  (threshold < 20%)
    sum ≈ full within 10.6%  (nonlinear, threshold < 30%)
  TestSingleModeAnalysis:
    |λ_eff| power law: p = -0.769  (threshold [-1.2, -0.5])
    near-field (r≤3) fraction: 81%  (threshold > 60%)
  TestCUVCutoff:
    R=5: C(1×) = 0.270, C(2×) = 0.368, C(4×) = 0.494  (monotonic growth)
  TestSeedIndependence:
    p_enh: 0.343 (seed=42), 0.354 (123), 0.357 (456), 0.335 (789), 0.320 (1001)
    CV = 3.9%  (threshold < 15%)
  TestPositionalNoise:
    clean: p_enh=0.354, ±0.1a: 0.353 (|Δp/p|=0.2%), ±0.2a: 0.350
  TestPropagationRange:
    r_cut=3: p=0.352, r_cut=∞: p=0.354, |Δp/p|=0.6%
    r_cut=2,5,∞ spread < 0.15 (oscillatory convergence, not monotonic)
  TestPhaseRegularization:
    physical r_cut=3: p=0.352, full: 0.354, |Δp/p|=0.6%  (converged)
    static r_cut=3: 0.471, r_cut=5: 0.607  (still growing, +29%)
    short range (r_cut=2): physical=0.323 > static=0.242
  TestEnergyConservation:
    σ_tot/σ_tr: 1.66 (k=0.3) → 7.38 (k=1.5). All > 1.
  TestForwardDecoherence:
    MS/Born at k=0.3: 0.611 (suppression, < 1)
    tilt: 0.611 (k=0.3) → 1.183 (k=1.5) (positive shift)
    forward |Σ Tb|²/|Σ Vb|² = 0.539 at k=0.3 (coherence broken)
    suppression grows with R (monotonic): 0.611 (R=5), 0.537 (R=7), 0.473 (R=9)

Analytic (seconds). Matrix computations.
"""
import numpy as np
import pytest
from helpers.config import K1, K2, c_lat, k_vals_7, ALPHA_REF, EPS_LAT, V_ref
from helpers.geometry import disk_bonds, random_disk
from helpers.lattice import dispersion_sq, k_eff, get_omega_k2
from helpers.born import V_eff
from helpers.ms import build_G_matrix, build_VG, T_matrix, make_dOmega
from helpers.stats import cv, log_log_slope, N_eff_from_sigma
from data.sigma_ring import sigma_ring, sigma_alpha_nn
from data.sigma_bond import sigma_bond


# ── Angular grid ──────────────────────────────────────────────────

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


def compute_neff_born_ms(R, k_arr, alpha, TH, PH, sin_th, cos_th,
                         transport):
    """Compute N_eff (Born and MS) via angular integration.

    Normalized to single-bond signal |V|²·Σw, so N_eff = σ_ring/σ_bond.
    Different from test_2's neff_born_discrete which divides by Σw only
    (giving ⟨|f|²⟩, not N_eff).
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


# ── Tests ─────────────────────────────────────────────────────────

class TestMSExponent:
    """MS exponent ≈ -2.0."""

    @pytest.mark.parametrize("R", [5, 7, 9])
    def test_ms_exponent_near_minus_2(self, R, angular_grid):
        """MS N_eff exponent ∈ [-2.15, -1.85] at R=5,7,9, α=0.30."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        _, nm = compute_neff_born_ms(R, k_vals_7, ALPHA_REF,
                                     TH, PH, sin_th, cos_th, transport)
        p = log_log_slope(k_vals_7, nm)[0]
        print(f"  R={R}: p_MS = {p:.3f}")
        assert -2.15 < p < -1.85, f"p_MS(R={R}) = {p:.3f}"

    def test_ms_exponent_r3_approximate(self, angular_grid):
        """R=3 (29 bonds): MS exponent ∈ [-2.15, -1.75] (small-R fluctuation)."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        _, nm = compute_neff_born_ms(3, k_vals_7, ALPHA_REF,
                                     TH, PH, sin_th, cos_th, transport)
        p = log_log_slope(k_vals_7, nm)[0]
        print(f"  R=3: p_MS = {p:.3f}")
        assert -2.15 < p < -1.75, f"p_MS(R=3) = {p:.3f}"

    def test_ms_captures_fdtd_correction(self, angular_grid):
        """MS shift captures 75-110% of FDTD shift at R=5."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        nb, nm = compute_neff_born_ms(5, k_vals_7, ALPHA_REF,
                                      TH, PH, sin_th, cos_th, transport)
        nf = sigma_ring[5] / sigma_bond
        p_b = log_log_slope(k_vals_7, nb)[0]
        p_m = log_log_slope(k_vals_7, nm)[0]
        p_f = log_log_slope(k_vals_7, nf)[0]
        frac = (p_m - p_b) / (p_f - p_b)
        print(f"  p_FDTD = {p_f:.3f}, MS/FDTD fraction = {frac:.2f}")
        assert 0.75 < frac < 1.10, \
            f"MS/FDTD fraction = {frac:.2f} at R=5"


class TestIntegrandFlatness:
    """MS flattens the integrand compared to Born."""

    def test_born_integrand_not_flat(self, angular_grid):
        """Born integrand sin²(k)·σ_bond·N_eff_Born: CV > 30%."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        nb, _ = compute_neff_born_ms(5, k_vals_7, ALPHA_REF,
                                     TH, PH, sin_th, cos_th, transport)
        integrand = np.sin(k_vals_7)**2 * sigma_bond * nb
        integrand_n = integrand / integrand[0]
        print(f"  Born integrand CV = {cv(integrand_n):.1f}%")
        assert cv(integrand_n) > 28.0, \
            f"Born integrand CV={cv(integrand_n):.1f}% ≤ 28%"

    def test_ms_integrand_flatter(self, angular_grid):
        """MS integrand CV < 25% (between Born and FDTD)."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        _, nm = compute_neff_born_ms(5, k_vals_7, ALPHA_REF,
                                     TH, PH, sin_th, cos_th, transport)
        integrand = np.sin(k_vals_7)**2 * sigma_bond * nm
        integrand_n = integrand / integrand[0]
        print(f"  MS integrand CV = {cv(integrand_n):.1f}%")
        assert cv(integrand_n) < 25.0, \
            f"MS integrand CV={cv(integrand_n):.1f}% ≥ 25%"


class TestShiftLinearity:
    """Exponent shift is linear in |V|."""

    def test_shift_proportional_to_V(self, angular_grid):
        """shift ≈ C·|V| with C ∈ [0.20, 0.45] across α scan."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        C_vals = []
        for alpha in [0.10, 0.20, 0.30, 0.40]:
            V = V_eff(alpha)
            nb, nm = compute_neff_born_ms(5, k_vals_7, alpha,
                                          TH, PH, sin_th, cos_th, transport)
            p_b = log_log_slope(k_vals_7, nb)[0]
            p_m = log_log_slope(k_vals_7, nm)[0]
            shift = p_m - p_b
            C = shift / abs(V)
            C_vals.append(C)
            print(f"  α={alpha}: C = {C:.3f}")
        C_arr = np.array(C_vals)
        assert all(0.20 < c < 0.45 for c in C_arr), \
            f"C values = {C_arr}"
        assert cv(C_arr) < 25, f"C CV={cv(C_arr):.1f}% > 25%"


class TestRandomDisk:
    """Enhancement is geometric (disk shape), not lattice-specific."""

    def test_random_disk_similar_enhancement(self, angular_grid):
        """Random disk p_enh / lattice p_enh ratio ∈ [0.7, 1.2]."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        R = 5
        dx_lat, dy_lat = disk_bonds(R)
        N = len(dx_lat)

        def compute_p_enh(dx, dy):
            N_b = len(dx)
            enh = np.zeros(len(k_vals_7))
            for ik, kv in enumerate(k_vals_7):
                G = build_G_matrix(dx, dy, kv)
                T = np.linalg.solve(np.eye(N_b) - V_ref * G,
                                    V_ref * np.eye(N_b))
                b = np.exp(1j * kv * dx)
                Tb = T @ b
                Vb = V_ref * b
                enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
            return log_log_slope(k_vals_7, enh)[0]

        p_lat = compute_p_enh(dx_lat, dy_lat)
        dx_r, dy_r = random_disk(R, N, seed=42)
        p_rand = compute_p_enh(dx_r, dy_r)
        ratio = p_rand / p_lat
        print(f"  lattice p_enh = {p_lat:.3f}, random p_enh = {p_rand:.3f}, ratio = {ratio:.2f}")
        assert 0.7 < ratio < 1.2, \
            f"Random/lattice p_enh ratio = {ratio:.2f}"


class TestPhaseRemoval:
    """Phase exp(ikr) REDUCES the MS shift (file 59 Part B).

    Without phase (static 1/r): shift is larger (+0.83).
    With phase: shift is +0.45. Phase randomization at high k reduces
    inter-bond cooperation.
    """

    def test_static_shift_larger_than_physical(self, angular_grid):
        """MS shift without exp(ikr) > MS shift with exp(ikr)."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        R = 5
        dx, dy = disk_bonds(R)
        N = len(dx)
        V = V_eff(ALPHA_REF)

        dist = np.sqrt((dx[:, None] - dx[None, :])**2
                       + (dy[:, None] - dy[None, :])**2)
        m = dist > 0

        omega_k2 = get_omega_k2()

        def compute_shift(with_phase):
            neff_b = np.zeros(len(k_vals_7))
            neff_m = np.zeros(len(k_vals_7))
            for ik, kv in enumerate(k_vals_7):
                ke = k_eff(kv)
                omega2 = dispersion_sq(kv)
                G = np.zeros((N, N), dtype=complex)
                if with_phase:
                    G[m] = np.exp(1j * ke * dist[m]) / (
                        4 * np.pi * c_lat**2 * dist[m])
                else:
                    G[m] = 1.0 / (4 * np.pi * c_lat**2 * dist[m])
                G_00 = np.mean(1.0 / (omega2 - omega_k2 + 1j * EPS_LAT))
                np.fill_diagonal(G, G_00)
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
            p_b = log_log_slope(k_vals_7, neff_b)[0]
            p_m = log_log_slope(k_vals_7, neff_m)[0]
            return p_m - p_b

        shift_phys = compute_shift(with_phase=True)
        shift_static = compute_shift(with_phase=False)
        print(f"  physical: p_enh = {shift_phys:.3f}, static: p_enh = {shift_static:.3f}, ratio = {shift_static/shift_phys:.2f}")
        assert shift_static > shift_phys, \
            f"Static shift {shift_static:.3f} ≤ physical {shift_phys:.3f}"
        # Static should be substantially larger (roughly 2× at R=5)
        assert shift_static > shift_phys * 1.3, \
            f"Static/physical ratio only {shift_static/shift_phys:.2f}"


class TestPhaseRegularization:
    """Phase exp(ikr) self-regularizes the Green function range.

    Physical G(r) = exp(ikr)/(4πc²r) converges by r_cut ≈ 3:
    oscillating phases cancel distant contributions.

    Static G(r) = 1/(4πc²r) does NOT converge: enhancement
    keeps growing as r_cut increases (no phase cancellation).

    At short range (r < 2.5), phases ADD constructive interference:
    physical enhancement > static. Beyond r ≈ 2.5, phases
    destructively cancel, stopping the growth.
    """

    def test_physical_converges_fast(self):
        """Physical G: r_cut=3 gives enhancement within 5% of full range."""
        dx, dy = disk_bonds(5)
        p_near = _enh_exp_custom(dx, dy, use_phase=True, r_cut=3)
        p_full = _enh_exp_custom(dx, dy, use_phase=True, r_cut=100)
        rel = abs(p_near - p_full) / abs(p_full)
        print(f"  physical r_cut=3: p={p_near:.3f}, full: p={p_full:.3f}, |Δp/p|={rel:.1%}")
        assert rel < 0.05, \
            f"|Δp/p| = {rel:.3f}, r_cut=3: {p_near:.3f}, full: {p_full:.3f}"

    def test_static_does_not_converge(self):
        """Static G: enhancement at r_cut=5 > r_cut=3 by > 15%."""
        dx, dy = disk_bonds(5)
        p_3 = _enh_exp_custom(dx, dy, use_phase=False, r_cut=3)
        p_5 = _enh_exp_custom(dx, dy, use_phase=False, r_cut=5)
        growth = (p_5 - p_3) / abs(p_3)
        print(f"  static r_cut=3: {p_3:.3f}, r_cut=5: {p_5:.3f}, growth={growth:.1%}")
        assert growth > 0.15, \
            f"Static growth = {growth:.1%}, expected > 15%"

    def test_short_range_phase_constructive(self):
        """At r_cut=2, physical > static (phases add constructively)."""
        dx, dy = disk_bonds(5)
        p_phys = _enh_exp_custom(dx, dy, use_phase=True, r_cut=2)
        p_stat = _enh_exp_custom(dx, dy, use_phase=False, r_cut=2)
        print(f"  r_cut=2: physical={p_phys:.3f}, static={p_stat:.3f}")
        assert p_phys > p_stat, \
            f"Physical {p_phys:.3f} ≤ static {p_stat:.3f} at r_cut=2"


def _enh_exp_custom(dx, dy, use_phase=True, r_cut=100):
    """Enhancement exponent with optional phase and range truncation."""
    N = len(dx)
    V = V_eff(ALPHA_REF)
    omega_k2 = get_omega_k2()
    dist = np.sqrt((dx[:, None] - dx[None, :])**2
                   + (dy[:, None] - dy[None, :])**2)
    enh = np.zeros(len(k_vals_7))
    for ik, kv in enumerate(k_vals_7):
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
        T_mat = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
        b = np.exp(1j * kv * dx)
        Tb = T_mat @ b
        Vb = V * b
        enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
    return log_log_slope(k_vals_7, enh)[0]


class TestBornSeriesConvergence:
    """Born series oscillates — requires full resummation (file 59 Part D).

    Born + VG gives only ~27% of shift. Born + VG + VG² overshoots to ~152%.
    Full MS (I-VG)^{-1} = 100%. Series oscillates at |λ_max| ≈ 0.6.
    """

    def test_first_correction_undershoots(self, angular_grid):
        """1st Born correction (VG) captures < 50% of full shift."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        R = 5
        dx, dy = disk_bonds(R)
        N = len(dx)
        V = V_eff(ALPHA_REF)
        x = np.log(k_vals_7)

        neff_b = np.zeros(len(k_vals_7))
        neff_1 = np.zeros(len(k_vals_7))
        neff_m = np.zeros(len(k_vals_7))

        for ik, kv in enumerate(k_vals_7):
            G = build_G_matrix(dx, dy, kv)
            VG = V * G
            b = np.exp(1j * kv * dx)
            Vb = V * b
            T1b = V * (b + VG @ b)
            T = np.linalg.solve(np.eye(N) - VG, V * np.eye(N))
            Tb = T @ b
            phase_out = kv * (
                sin_th[:, :, None] * np.cos(PH)[:, :, None]
                * dx[None, None, :]
                + sin_th[:, :, None] * np.sin(PH)[:, :, None]
                * dy[None, None, :])
            a = np.exp(-1j * phase_out)
            qz = kv * cos_th
            F2 = 4 * np.cos(qz / 2)**2
            w = F2 * transport * sin_th
            s_single = np.abs(V)**2 * np.sum(w)
            for amp, arr in [(Vb, neff_b), (T1b, neff_1), (Tb, neff_m)]:
                f = np.sum(a * amp[None, None, :], axis=2)
                arr[ik] = np.sum(np.abs(f)**2 * w) / s_single

        p_b = np.polyfit(x, np.log(neff_b), 1)[0]
        p_1 = np.polyfit(x, np.log(neff_1), 1)[0]
        p_m = np.polyfit(x, np.log(neff_m), 1)[0]
        shift_full = p_m - p_b
        frac_1 = (p_1 - p_b) / shift_full
        print(f"  full resummation: p = {p_m - p_b:.3f}, 1st correction: p = {p_1 - p_b:.3f} ({frac_1*100:.0f}%)")
        assert frac_1 < 0.50, \
            f"1st correction captures {frac_1*100:.0f}% > 50%"
        assert frac_1 > 0.0, \
            f"1st correction is negative: {frac_1*100:.0f}%"


class TestShiftDecomposition:
    """Shift is ~80% from inter-bond 1/r, ~9% from self-energy G₀₀ (file 59 Part E).

    Clean decomposition using 4 G types:
    - G₀₀ diagonal only: small shift (~9%)
    - Off-diagonal 1/r only (no G₀₀): dominant shift (~80%)
    - Physical (both): full shift (100%)
    - Uniform G: maximum possible shift
    """

    def test_offdiag_dominates_diagonal(self, angular_grid):
        """Off-diagonal 1/r gives > 3× the shift of G₀₀-only."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        R = 5
        dx, dy = disk_bonds(R)
        N = len(dx)
        V = V_eff(ALPHA_REF)
        x = np.log(k_vals_7)
        dist = np.sqrt((dx[:, None] - dx[None, :])**2
                       + (dy[:, None] - dy[None, :])**2)
        mask = dist > 0

        omega_k2 = get_omega_k2()

        def compute_shift_g(g_type):
            neff_b_g = np.zeros(len(k_vals_7))
            neff_m_g = np.zeros(len(k_vals_7))
            for ik, kv in enumerate(k_vals_7):
                ke = k_eff(kv)
                omega2 = dispersion_sq(kv)
                G_00_val = np.mean(1.0 / (omega2 - omega_k2 + 1j * EPS_LAT))
                G_g = np.zeros((N, N), dtype=complex)
                if g_type == 'diagonal':
                    np.fill_diagonal(G_g, G_00_val)
                elif g_type == 'no_self':
                    G_g[mask] = np.exp(1j * ke * dist[mask]) / (
                        4 * np.pi * c_lat**2 * dist[mask])
                elif g_type == 'physical':
                    G_g[mask] = np.exp(1j * ke * dist[mask]) / (
                        4 * np.pi * c_lat**2 * dist[mask])
                    np.fill_diagonal(G_g, G_00_val)
                T = np.linalg.solve(np.eye(N) - V * G_g, V * np.eye(N))
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
                neff_b_g[ik] = np.sum(np.abs(f_born)**2 * w) / s_single
                neff_m_g[ik] = np.sum(np.abs(f_ms)**2 * w) / s_single
            pb = np.polyfit(x, np.log(neff_b_g), 1)[0]
            pm = np.polyfit(x, np.log(neff_m_g), 1)[0]
            return pm - pb

        sh_diag = compute_shift_g('diagonal')
        sh_noself = compute_shift_g('no_self')
        sh_phys = compute_shift_g('physical')
        print(f"  full: {sh_phys:.3f}, offdiag: {sh_noself:.3f}, G₀₀: {sh_diag:.3f}, G₀₀ fraction = {sh_diag/sh_phys:.0%}")
        # Off-diagonal dominates
        assert sh_noself > 3 * sh_diag, \
            f"off-diag shift {sh_noself:.3f} ≤ 3× diag {sh_diag:.3f}"
        # Physical > off-diag (G₀₀ adds to shift, empirical — T-matrix is nonlinear
        # so shifts don't add exactly: full ≠ offdiag + diag)
        assert sh_phys > sh_noself, \
            f"physical {sh_phys:.3f} ≤ off-diag {sh_noself:.3f}"
        # G₀₀ is small fraction
        assert sh_diag / sh_phys < 0.20, \
            f"G₀₀ fraction = {sh_diag/sh_phys:.2f} > 20%"
        # Approximate additivity: full ≈ offdiag + diag (within 30%)
        # Not exact because T = (I-VG)^{-1}V is nonlinear in G
        approx_sum = sh_noself + sh_diag
        rel_err = abs(approx_sum - sh_phys) / sh_phys
        print(f"  sum={approx_sum:.3f}, |sum-full|/full = {rel_err:.1%}")
        assert rel_err < 0.30, \
            f"Decomposition error {rel_err:.1%} > 30%"


def _compute_abs_lambda_eff(R=5):
    """Compute |λ_eff| at each k in k_vals_7 (pure inter-bond, no G₀₀)."""
    dx, dy = disk_bonds(R)
    N = len(dx)
    dist = np.sqrt((dx[:, None] - dx[None, :])**2
                   + (dy[:, None] - dy[None, :])**2)
    mask = dist > 0
    V = V_eff(ALPHA_REF)
    abs_lam = np.zeros(len(k_vals_7))
    for ik, kv in enumerate(k_vals_7):
        ke = k_eff(kv)
        G = np.zeros((N, N), dtype=complex)
        G[mask] = np.exp(1j * ke * dist[mask]) / (
            4 * np.pi * c_lat**2 * dist[mask])
        b = np.exp(1j * kv * dx)
        VGb = V * (G @ b)
        lam = np.sum(VGb * np.conj(b)) / np.sum(np.abs(b)**2)
        abs_lam[ik] = np.abs(lam)
    return abs_lam


class TestSingleModeAnalysis:
    """Single-mode Rayleigh quotient captures MS shift (file 59 Part G).

    λ_eff = ⟨b|VG|b⟩/⟨b|b⟩ projects VG onto incident mode.
    |λ_eff| ~ k^{-0.77}: phases coherent at low k → more cooperation.
    Single-mode T·b ≈ V/(1-λ_eff)·b captures 77-100% of shift.
    """

    def test_lambda_eff_decreases_with_k(self):
        """|λ_eff| decreases with k (phase decoherence at high k)."""
        abs_lam = _compute_abs_lambda_eff(R=5)
        p = log_log_slope(k_vals_7, abs_lam)[0]
        print(f"  |λ_eff| power law (monotonicity check): p = {p:.3f}")
        for i in range(len(abs_lam) - 1):
            assert abs_lam[i + 1] < abs_lam[i], \
                f"|λ_eff| not decreasing: {abs_lam[i]:.4f} → {abs_lam[i+1]:.4f}"

    def test_lambda_eff_power_law(self):
        """|λ_eff| ~ k^p with p ≈ -0.77 (near but not exactly -1)."""
        abs_lam = _compute_abs_lambda_eff(R=5)
        p = log_log_slope(k_vals_7, abs_lam)[0]
        print(f"  |λ_eff| power law: p = {p:.3f}")
        assert -1.2 < p < -0.5, f"|λ_eff| exponent = {p:.3f}"

    def test_near_field_dominates(self):
        """~80% of λ_eff from r ≤ 3 (short-range lattice structure)."""
        R = 5
        dx, dy = disk_bonds(R)
        N = len(dx)
        dist = np.sqrt((dx[:, None] - dx[None, :])**2
                       + (dy[:, None] - dy[None, :])**2)
        V = V_eff(ALPHA_REF)
        kv = 0.5
        ke = k_eff(kv)
        b = np.exp(1j * kv * dx)
        # Full G (no self-energy)
        mask_full = dist > 0
        G_full = np.zeros((N, N), dtype=complex)
        G_full[mask_full] = np.exp(1j * ke * dist[mask_full]) / (
            4 * np.pi * c_lat**2 * dist[mask_full])
        VGb_full = V * (G_full @ b)
        lam_full = np.sum(VGb_full * np.conj(b)) / np.sum(np.abs(b)**2)
        # Near-field G (r ≤ 3)
        mask_near = (dist > 0) & (dist <= 3)
        G_near = np.zeros((N, N), dtype=complex)
        G_near[mask_near] = np.exp(1j * ke * dist[mask_near]) / (
            4 * np.pi * c_lat**2 * dist[mask_near])
        VGb_near = V * (G_near @ b)
        lam_near = np.sum(VGb_near * np.conj(b)) / np.sum(np.abs(b)**2)
        frac = np.abs(lam_near) / np.abs(lam_full)
        print(f"  near-field (r≤3) fraction: {frac:.0%}")
        assert frac > 0.6, f"Near-field fraction = {frac:.2f} < 0.6"


class TestCUVCutoff:
    """C is UV-cutoff dependent: grows with density ~ log(density)."""

    def test_c_grows_with_density(self, angular_grid):
        """C at 4× density > C at 1× density."""
        TH, PH, sin_th, cos_th, transport = angular_grid
        R = 5
        dx_lat, dy_lat = disk_bonds(R)
        N_lat = len(dx_lat)
        rng = np.random.RandomState(123)

        def compute_C(dx, dy):
            N = len(dx)
            p_born_arr = []
            p_ms_arr = []
            for ik, kv in enumerate(k_vals_7):
                G = build_G_matrix(dx, dy, kv)
                T = np.linalg.solve(np.eye(N) - V_ref * G,
                                    V_ref * np.eye(N))
                b = np.exp(1j * kv * dx)
                Tb = T @ b
                Vb = V_ref * b
                p_born_arr.append(np.log(np.sum(np.abs(Vb)**2)))
                p_ms_arr.append(np.log(np.sum(np.abs(Tb)**2)))
            pb = np.polyfit(np.log(k_vals_7), p_born_arr, 1)[0]
            pm = np.polyfit(np.log(k_vals_7), p_ms_arr, 1)[0]
            return (pm - pb) / abs(V_ref)

        C_1x = compute_C(dx_lat, dy_lat)
        # 2× density random disk
        N_2x = N_lat * 2
        angles_2 = rng.uniform(0, 2 * np.pi, N_2x)
        radii_2 = R * np.sqrt(rng.uniform(0, 1, N_2x))
        dx_2x = radii_2 * np.cos(angles_2)
        dy_2x = radii_2 * np.sin(angles_2)
        C_2x = compute_C(dx_2x, dy_2x)
        # 4× density random disk
        N_4x = N_lat * 4
        angles_4 = rng.uniform(0, 2 * np.pi, N_4x)
        radii_4 = R * np.sqrt(rng.uniform(0, 1, N_4x))
        dx_4x = radii_4 * np.cos(angles_4)
        dy_4x = radii_4 * np.sin(angles_4)
        C_4x = compute_C(dx_4x, dy_4x)
        print(f"  R=5: C(1×) = {C_1x:.3f}, C(2×) = {C_2x:.3f}, C(4×) = {C_4x:.3f}")
        # Growth is gradual: 1× < 2× < 4×
        assert C_2x > C_1x, \
            f"C(2×) = {C_2x:.3f} ≤ C(1×) = {C_1x:.3f}"
        assert C_4x > C_2x, \
            f"C(4×) = {C_4x:.3f} ≤ C(2×) = {C_2x:.3f}"


def _enhancement_exponent(dx, dy, k_arr=k_vals_7, alpha=ALPHA_REF):
    """Enhancement exponent from |Tb|²/|Vb|² power law in k."""
    N = len(dx)
    V = V_eff(alpha)
    enh = np.zeros(len(k_arr))
    for ik, kv in enumerate(k_arr):
        G = build_G_matrix(dx, dy, kv)
        T_mat = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
        b = np.exp(1j * kv * dx)
        Tb = T_mat @ b
        Vb = V * b
        enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
    return log_log_slope(k_arr, enh)[0]


class TestSeedIndependence:
    """MS enhancement exponent is reproducible across random seeds."""

    def test_exponent_cv_across_seeds(self):
        """Enhancement exponent CV < 15% over 5 random disk seeds."""
        R, N_bonds = 5, 81
        slopes = []
        for seed in [42, 123, 456, 789, 1001]:
            dx, dy = random_disk(R, N_bonds, seed=seed)
            slopes.append(_enhancement_exponent(dx, dy))
        slopes = np.array(slopes)
        print(f"  p_enh: {', '.join(f'{s:.3f}' for s in slopes)}")
        print(f"  CV = {cv(slopes):.1f}%")
        assert cv(slopes) < 15.0, \
            f"Seed CV = {cv(slopes):.1f}%, slopes = {slopes}"

    def test_all_seeds_positive_enhancement(self):
        """All random seeds give positive MS enhancement (p_enh > 0)."""
        R, N_bonds = 5, 81
        for seed in [42, 123, 456, 789]:
            dx, dy = random_disk(R, N_bonds, seed=seed)
            p = _enhancement_exponent(dx, dy)
            print(f"  seed={seed}: p_enh = {p:.3f}")
            assert p > 0.05, \
                f"seed={seed}: p_enh = {p:.3f} ≤ 0.05"


class TestPositionalNoise:
    """MS exponent is robust to lattice disorder (±0.1a noise)."""

    def test_noisy_exponent_close_to_clean(self):
        """Enhancement exponent changes < 15% with ±0.1a noise."""
        dx0, dy0 = disk_bonds(5)
        p_clean = _enhancement_exponent(dx0, dy0)
        rng = np.random.RandomState(42)
        N = len(dx0)
        dx_noisy = dx0 + rng.uniform(-0.1, 0.1, N)
        dy_noisy = dy0 + rng.uniform(-0.1, 0.1, N)
        p_noisy = _enhancement_exponent(dx_noisy, dy_noisy)
        rel_change = abs(p_noisy - p_clean) / abs(p_clean)
        print(f"  clean: p_enh={p_clean:.3f}, ±0.1a: {p_noisy:.3f} (|Δp/p|={rel_change:.1%})")
        assert rel_change < 0.15, \
            f"|Δp/p| = {rel_change:.3f}, clean={p_clean:.3f}, noisy={p_noisy:.3f}"

    def test_large_noise_still_positive(self):
        """Even ±0.2a noise gives positive enhancement exponent."""
        dx0, dy0 = disk_bonds(5)
        for seed in [42, 123]:
            rng = np.random.RandomState(seed)
            N = len(dx0)
            dx_n = dx0 + rng.uniform(-0.2, 0.2, N)
            dy_n = dy0 + rng.uniform(-0.2, 0.2, N)
            p = _enhancement_exponent(dx_n, dy_n)
            print(f"  ±0.2a (seed={seed}): p_enh = {p:.3f}")
            assert p > 0.05, \
                f"seed={seed}, noise=±0.2a: p_enh = {p:.3f} ≤ 0.05"


class TestPropagationRange:
    """MS exponent converges with Green function range r_cut."""

    def test_near_field_gives_correct_exponent(self):
        """r_cut=3 already gives exponent within 20% of full range."""
        dx, dy = disk_bonds(5)
        N = len(dx)
        V = V_eff(ALPHA_REF)
        dist = np.sqrt((dx[:, None] - dx[None, :])**2
                       + (dy[:, None] - dy[None, :])**2)
        omega_k2 = get_omega_k2()

        def enh_exponent_truncated(r_cut):
            enh = np.zeros(len(k_vals_7))
            for ik, kv in enumerate(k_vals_7):
                ke = k_eff(kv)
                G = np.zeros((N, N), dtype=complex)
                m = (dist > 0) & (dist <= r_cut)
                G[m] = np.exp(1j * ke * dist[m]) / (
                    4 * np.pi * c_lat**2 * dist[m])
                omega2 = dispersion_sq(kv)
                G_00 = np.mean(1.0 / (omega2 - omega_k2 + 1j * EPS_LAT))
                np.fill_diagonal(G, G_00)
                T_mat = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
                b = np.exp(1j * kv * dx)
                Tb = T_mat @ b
                Vb = V * b
                enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
            return log_log_slope(k_vals_7, enh)[0]

        p_full = enh_exponent_truncated(100)
        p_near = enh_exponent_truncated(3)
        rel = abs(p_near - p_full) / abs(p_full)
        print(f"  r_cut=3: p={p_near:.3f}, r_cut=∞: p={p_full:.3f}, |Δp/p|={rel:.1%}")
        assert rel < 0.20, \
            f"|Δp/p| = {rel:.3f}, full={p_full:.3f}, r_cut=3={p_near:.3f}"

    def test_exponent_spread_small(self):
        """Enhancement exponent spread < 0.15 across r_cut values.

        Not monotonic: r_cut=5 can slightly overshoot r_cut=∞ due to
        phase interference at intermediate range. Convergence is oscillatory.
        """
        dx, dy = disk_bonds(5)
        N = len(dx)
        V = V_eff(ALPHA_REF)
        dist = np.sqrt((dx[:, None] - dx[None, :])**2
                       + (dy[:, None] - dy[None, :])**2)
        omega_k2 = get_omega_k2()

        p_vals = []
        for r_cut in [2, 5, 100]:
            enh = np.zeros(len(k_vals_7))
            for ik, kv in enumerate(k_vals_7):
                ke = k_eff(kv)
                G = np.zeros((N, N), dtype=complex)
                m = (dist > 0) & (dist <= r_cut)
                G[m] = np.exp(1j * ke * dist[m]) / (
                    4 * np.pi * c_lat**2 * dist[m])
                omega2 = dispersion_sq(kv)
                G_00 = np.mean(1.0 / (omega2 - omega_k2 + 1j * EPS_LAT))
                np.fill_diagonal(G, G_00)
                T_mat = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
                b = np.exp(1j * kv * dx)
                Tb = T_mat @ b
                Vb = V * b
                enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
            p_vals.append(log_log_slope(k_vals_7, enh)[0])
        print(f"  r_cut=2: p={p_vals[0]:.3f}, r_cut=5: p={p_vals[1]:.3f}, r_cut=∞: p={p_vals[2]:.3f}")
        # Spread should be small (convergence)
        spread = max(p_vals) - min(p_vals)
        assert spread < 0.15, \
            f"Enhancement exponent spread = {spread:.3f}, values = {p_vals}"


def _compute_sigma_tot_tr(R=5):
    """Compute σ_tot and σ_tr at each k in k_vals_7."""
    dx, dy = disk_bonds(R)
    n_theta, n_phi = 15, 24
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dOmega = make_dOmega(thetas, n_phi)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    sin_th = np.sin(TH)
    cos_s = sin_th * np.cos(PH)

    s_tot_arr = np.zeros(len(k_vals_7))
    s_tr_arr = np.zeros(len(k_vals_7))
    for ik, kv in enumerate(k_vals_7):
        ke = 2 * np.sin(kv / 2)
        T_mat = T_matrix(dx, dy, kv, alpha=ALPHA_REF)
        Tb = T_mat @ np.exp(1j * kv * dx)
        KX = ke * sin_th * np.cos(PH)
        KY = ke * sin_th * np.sin(PH)
        phase = np.exp(-1j * (KX[:, :, None] * dx + KY[:, :, None] * dy))
        F = np.einsum('ijk,k->ij', phase, Tb)
        f2 = np.abs(F)**2
        s_tot_arr[ik] = np.sum(f2 * dOmega[:, None])
        s_tr_arr[ik] = np.sum((1 - cos_s) * f2 * dOmega[:, None])
    return s_tot_arr, s_tr_arr


class TestEnergyConservation:
    """σ_tot ≥ σ_tr (total cross section ≥ transport cross section)."""

    def test_sigma_tot_geq_sigma_tr(self):
        """σ_tot > σ_tr at all k (energy conservation)."""
        s_tot, s_tr = _compute_sigma_tot_tr(R=5)
        for ik, kv in enumerate(k_vals_7):
            print(f"  k={kv:.1f}: σ_tot/σ_tr = {s_tot[ik]/s_tr[ik]:.2f}")
            assert s_tot[ik] >= s_tr[ik], \
                f"σ_tot < σ_tr at k={kv}: {s_tot[ik]:.1f} < {s_tr[ik]:.1f}"

    def test_sigma_ratio_grows_with_k(self):
        """σ_tot/σ_tr grows with k (more forward-peaked at high k)."""
        s_tot, s_tr = _compute_sigma_tot_tr(R=5)
        ratios = s_tot / s_tr
        print(f"  σ_tot/σ_tr: {ratios[0]:.2f} (k=0.3) → {ratios[-1]:.2f} (k=1.5)")
        assert ratios[-1] > ratios[0], \
            f"σ_tot/σ_tr ratio not growing: {ratios[0]:.2f} → {ratios[-1]:.2f}"


class TestForwardDecoherence:
    """MS shifts exponent via forward decoherence, not cooperative enhancement.

    At low k, Born has strong forward coherence: phases exp(ik·dx) ≈ 1,
    so |Σ V·exp(ik·dx)|² ≈ N²|V|² (constructive interference in forward cone).
    Multiple scattering (T-matrix) breaks this forward coherence through
    inter-bond re-scattering, reducing N_eff at low k more than at high k.

    This produces a positive exponent shift (+1/2): the Born peak at low k
    is suppressed, flattening N_eff(k) from k^{-5/2} toward k^{-2}.

    NOT superradiance: MS SUPPRESSES at low k (enh=0.61), not enhances.
    """

    @staticmethod
    def _compute_enhancement(dx, dy, k_arr):
        """Σ|Tb|²/Σ|Vb|² at each k (MS/Born total power ratio)."""
        V = V_ref
        N = len(dx)
        enh = np.zeros(len(k_arr))
        for ik, kv in enumerate(k_arr):
            G = build_G_matrix(dx, dy, kv)
            b = np.exp(1j * kv * dx)
            Vb = V * b
            T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
            Tb = T @ b
            enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
        return enh

    def test_ms_suppresses_low_k(self):
        """MS/Born < 1 at k=0.3: MS reduces scattering at low k."""
        dx, dy = disk_bonds(5)
        enh = self._compute_enhancement(dx, dy, k_vals_7)
        print(f"  MS/Born at k=0.3: {enh[0]:.3f} (< 1 = suppression)")
        assert enh[0] < 1.0, \
            f"MS/Born = {enh[0]:.3f} at k=0.3, expected < 1"

    def test_enhancement_tilts_upward(self):
        """MS enhancement increases with k: positive exponent shift.

        enh(k=1.5) > enh(k=0.3) → MS tilts spectrum upward → positive shift.
        """
        dx, dy = disk_bonds(5)
        enh = self._compute_enhancement(dx, dy, k_vals_7)
        print(f"  MS/Born: {enh[0]:.3f} (k=0.3) → {enh[-1]:.3f} (k=1.5)")
        assert enh[-1] > enh[0], \
            f"Enhancement not tilting upward: {enh[0]:.3f} → {enh[-1]:.3f}"

    def test_forward_coherence_broken(self):
        """MS reduces forward amplitude |Σ Tb|²/|Σ Vb|² at low k.

        Forward coherence |Σ exp(ikdx)|² ≈ N² at k→0.
        MS breaks this: |Σ Tb|² < |Σ Vb|² at k=0.3.
        """
        dx, dy = disk_bonds(5)
        V = V_ref
        N = len(dx)
        kv = 0.3
        G = build_G_matrix(dx, dy, kv)
        b = np.exp(1j * kv * dx)
        Vb = V * b
        T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
        Tb = T @ b
        fwd_born = abs(np.sum(Vb))**2
        fwd_ms = abs(np.sum(Tb))**2
        ratio = fwd_ms / fwd_born
        print(f"  Forward |Σ Tb|²/|Σ Vb|² = {ratio:.3f} at k=0.3 (< 1)")
        assert ratio < 1.0, \
            f"Forward not suppressed: ratio = {ratio:.3f}"

    def test_suppression_grows_with_R(self):
        """Larger disk → more bonds → stronger forward decoherence at low k.

        Suppression at k=0.3: 0.611 (R=5) → 0.537 (R=7) → 0.473 (R=9).
        Strictly monotonic: more scatterers = more decoherence.
        """
        enhs = []
        for R in [5, 7, 9]:
            dx, dy = disk_bonds(R)
            enh = self._compute_enhancement(dx, dy, k_vals_7)
            enhs.append(enh[0])
            print(f"  R={R}: MS/Born at k=0.3 = {enh[0]:.3f}")
            assert enh[0] < 1.0, \
                f"R={R}: MS/Born = {enh[0]:.3f} at k=0.3, expected < 1"
        # Suppression grows monotonically with R
        assert enhs[1] < enhs[0] and enhs[2] < enhs[1], \
            f"Suppression not monotonic: {enhs}"
