"""§10 Forward decoherence phase 2: scaling, consistency, eigenvalue structure.

Extends test_9 with additional verifications from Phase 2 review.
test_9 (42 tests) covers the core mechanism (F2-F13).
test_10 covers scaling laws, cross-method consistency, and structural tests.

Results:
  TestCrossoverScaling (B1, 2 tests):
    Crossover k (enh=1) ~ R^0.59. k*=0.69 (R=3) → 1.56 (R=12).
    Power law p=0.59 in [0.4, 0.8]. Predicts k*→BZ edge at R≥25.
  TestEigenvalueDominance (B4, 2 tests):
    |λ_max|/|λ_second| = 2.52 at k=0.3 (single dominant mode).
    Ratio → 1.01 at k=1.5 (no dominant mode). Justifies mean-field at low k.
  TestShiftMethodConsistency (A3+A4, 1 test):
    test_9 (with F²) and ms.py (without F²) give shifts +0.474 and +0.480
    (1.2% diff). F² invariant. Born exponent near -5/2 (links to test_2).
  TestPathExcessConcentration (B3, 2 tests):
    1/r weighting concentrates cooperation at short range.
    Weighted median 2.0 vs unweighted 4.0. frac(pe<3) = 62% by weight.
  TestIntegrandCVMonotonic (A2, 1 test):
    MS integrand CV decreases monotonically with α (44% at 0.15 → 27% at 0.40).
    No minimum at α=0.25 — stronger coupling always flattens more.
  TestKappaConsistency (A1, 2 tests):
    MS/FDTD gap (~15×) is per-bond normalization, not collective.
    N_eff_MS/N_eff_FDTD = 1.05 (mean). Mechanism validated.
  TestRScaling (B6, 2 tests):
    Born σ_tr ~ R^{3/2} at fixed k. MS reduces R-scaling (p_ms < p_born)
    because forward decoherence strengthens with R.

Analytic (seconds). Matrix computations + FDTD data comparison.
"""
import numpy as np
from helpers.config import K1, K2, c_lat, k_vals_7, ALPHA_REF, V_ref
from helpers.geometry import disk_bonds
from helpers.lattice import k_eff
from helpers.born import V_eff, sigma_bond_born
from helpers.ms import (build_G_matrix, T_matrix, eigenvalues_VG,
                         sigma_tr_born_ms, make_dOmega)
from helpers.stats import cv, log_log_slope
from data.sigma_bond import sigma_bond as sigma_bond_fdtd
from data.sigma_ring import sigma_ring


def _enhancement_at_k(dx, dy, k_arr, alpha=ALPHA_REF):
    """Compute total power enhancement Σ|Tb|²/Σ|Vb|² at each k."""
    V = V_eff(alpha)
    enh = np.zeros(len(k_arr))
    for ik, kv in enumerate(k_arr):
        b = np.exp(1j * kv * dx)
        T = T_matrix(dx, dy, kv, alpha)
        Tb = T @ b
        Vb = V * b
        enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
    return enh


def _find_crossover(enh, k_arr):
    """Find k where enh crosses 1 (linear interpolation). None if no crossing."""
    for i in range(len(enh) - 1):
        if enh[i] < 1 and enh[i + 1] >= 1:
            return k_arr[i] + (k_arr[i + 1] - k_arr[i]) * \
                (1 - enh[i]) / (enh[i + 1] - enh[i])
    return None


# ── B1: Crossover k scaling with R ──────────────────────────────

class TestCrossoverScaling:
    """B1: Crossover k ~ R^p power law.

    The crossover k (where total power enh = 1) grows with R because
    larger disks extend the suppression region to higher k.
    Fit: k* ~ R^0.59. Connects to cooperation range scaling.

    Ref: internal/0_tracker_v5.md, Phase 2 B1.
    """

    def test_crossover_power_law(self):
        """Crossover k follows power law k* ~ R^p with p in [0.4, 0.8].

        Extended k range [0.3, 2.5] to capture crossover at large R.
        """
        k_fine = np.linspace(0.3, 2.5, 50)
        crossovers = {}
        for R in [3, 5, 7, 9, 12]:
            dx, dy = disk_bonds(R)
            enh = _enhancement_at_k(dx, dy, k_fine)
            ck = _find_crossover(enh, k_fine)
            if ck is not None:
                crossovers[R] = ck

        R_vals = np.array(sorted(crossovers.keys()), dtype=float)
        k_stars = np.array([crossovers[int(r)] for r in R_vals])

        p, _ = log_log_slope(R_vals, k_stars)
        print(f"  k* ~ R^{p:.3f}")
        for R in sorted(crossovers):
            print(f"    R={R}: k*={crossovers[R]:.3f}")

        assert len(crossovers) >= 4, \
            f"Only {len(crossovers)} crossovers found, need ≥ 4"
        assert 0.4 < p < 0.8, \
            f"Crossover exponent p={p:.3f} outside [0.4, 0.8]"

    def test_crossover_monotonic(self):
        """Crossover k increases monotonically with R."""
        k_fine = np.linspace(0.3, 2.5, 50)
        prev_ck = 0
        for R in [3, 5, 7, 9]:
            dx, dy = disk_bonds(R)
            enh = _enhancement_at_k(dx, dy, k_fine)
            ck = _find_crossover(enh, k_fine)
            assert ck is not None, f"No crossover at R={R}"
            print(f"  R={R}: k*={ck:.3f}")
            assert ck > prev_ck, \
                f"Crossover not monotonic: R={R} gives k*={ck:.3f} ≤ {prev_ck:.3f}"
            prev_ck = ck


# ── B4: Eigenvalue dominance ─────────────────────────────────────

class TestEigenvalueDominance:
    """B4: Single dominant eigenvalue at low k, none at high k.

    At k=0.3: |λ_max|/|λ_second| ≈ 2.5 — one dominant mode concentrates
    cooperation. This justifies a mean-field / single-mode approximation.
    At k=1.5: ratio ≈ 1.0 — all modes contribute equally (no dominant mode).

    Ref: internal/0_tracker_v5.md, Phase 2 B4.
    """

    def test_dominant_mode_at_low_k(self):
        """|λ_max|/|λ_second| > 2 at k=0.3 (single dominant mode).

        The dominant eigenvalue (|λ|=0.60) is well separated from the
        rest (|λ₂|=|λ₃|≈0.24, degenerate from disk symmetry).
        """
        dx, dy = disk_bonds(5)
        eigs = eigenvalues_VG(dx, dy, 0.3, ALPHA_REF)
        mags = np.sort(np.abs(eigs))[::-1]
        ratio = mags[0] / mags[1]
        print(f"  k=0.3: |λ₁|={mags[0]:.4f}, |λ₂|={mags[1]:.4f}, "
              f"ratio={ratio:.2f}")
        assert ratio > 2.0, \
            f"|λ₁|/|λ₂| = {ratio:.2f} at k=0.3, expected > 2.0"

    def test_no_dominant_mode_at_high_k(self):
        """|λ_max|/|λ_second| < 1.5 at k=1.5 (no dominant mode).

        Phases randomize → eigenvalues spread uniformly → no single
        mode concentrates the cooperation.
        """
        dx, dy = disk_bonds(5)
        eigs = eigenvalues_VG(dx, dy, 1.5, ALPHA_REF)
        mags = np.sort(np.abs(eigs))[::-1]
        ratio = mags[0] / mags[1]
        print(f"  k=1.5: |λ₁|={mags[0]:.4f}, |λ₂|={mags[1]:.4f}, "
              f"ratio={ratio:.2f}")
        assert ratio < 1.5, \
            f"|λ₁|/|λ₂| = {ratio:.2f} at k=1.5, expected < 1.5"


# ── A3: Shift method consistency ─────────────────────────────────

class TestShiftMethodConsistency:
    """A3: Transport shift invariant under F² polarization factor.

    test_9 computes N_eff_transport with F²=4cos²(qz/2) (z-bond
    polarization factor). ms.py sigma_tr_ms omits F². The two give
    very different absolute N_eff (~28× ratio) but nearly identical
    shifts (+0.474 vs +0.480, 1.2% difference).

    F² adds ~k^{-2.3} to both Born and MS equally, so it cancels in
    the shift. This verifies internal consistency between test_9 and
    ms.py computations.

    Ref: internal/0_tracker_v5.md, Phase 2 A3.
    """

    def test_shift_agrees_within_5pct(self):
        """Transport shift from ms.py (no F²) matches test_9 style (with F²)."""
        dx, dy = disk_bonds(5)
        k = np.array(k_vals_7)
        alpha = ALPHA_REF
        V = V_eff(alpha)

        # Method 1: sigma_tr_born_ms (no F²)
        sb1, sm1 = sigma_tr_born_ms(dx, dy, k, alpha, n_theta=50, n_phi=100)
        p_born_1 = log_log_slope(k, sb1)[0]
        p_ms_1 = log_log_slope(k, sm1)[0]
        shift_1 = p_ms_1 - p_born_1

        # Method 2: angular integration with F² (test_9 style)
        n_theta, n_phi = 50, 100
        TH_grid = np.linspace(0, np.pi, n_theta)
        PH_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(TH_grid, PH_grid, indexing='ij')
        sin_th, cos_th = np.sin(TH), np.cos(TH)
        transport = 1 - sin_th * np.cos(PH)

        dO = make_dOmega(TH_grid, n_phi)

        sigma_born_2 = np.zeros(len(k))
        sigma_ms_2 = np.zeros(len(k))
        for ik, kv in enumerate(k):
            ke = k_eff(kv)
            KX = ke * sin_th * np.cos(PH)
            KY = ke * sin_th * np.sin(PH)
            KZ = ke * cos_th
            F2 = 4 * np.cos(KZ / 2)**2

            T = T_matrix(dx, dy, kv, alpha)
            b = np.exp(1j * kv * dx)
            phase = np.exp(-1j * (KX[:, :, None] * dx + KY[:, :, None] * dy))
            f_ms = np.einsum('ijk,k->ij', phase, T @ b)
            f_born = np.einsum('ijk,k->ij', phase, V * b)

            sigma_ms_2[ik] = np.sum(
                F2 * transport * np.abs(f_ms)**2 * dO[:, None]
            ) / (4 * np.pi * ke**2)
            sigma_born_2[ik] = np.sum(
                F2 * transport * np.abs(f_born)**2 * dO[:, None]
            ) / (4 * np.pi * ke**2)

        p_born_2 = log_log_slope(k, sigma_born_2)[0]
        p_ms_2 = log_log_slope(k, sigma_ms_2)[0]
        shift_2 = p_ms_2 - p_born_2

        print(f"  No F²: shift = {shift_1:+.3f} "
              f"(Born={p_born_1:.2f}, MS={p_ms_1:.2f})")
        print(f"  With F²: shift = {shift_2:+.3f} "
              f"(Born={p_born_2:.2f}, MS={p_ms_2:.2f})")
        print(f"  Difference: {abs(shift_1 - shift_2):.3f} "
              f"({100 * abs(shift_1 - shift_2) / abs(shift_1):.1f}%)")

        assert abs(shift_1 - shift_2) / abs(shift_1) < 0.05, \
            f"Shifts disagree: {shift_1:+.3f} vs {shift_2:+.3f} " \
            f"(>{5}% difference)"

        # A4: Born exponent near -5/2 (links test_10 to test_2)
        assert -2.8 < p_born_1 < -2.2, \
            f"Born exponent {p_born_1:.2f} outside [-2.8, -2.2] " \
            f"(expected near -5/2)"


# ── B3: Path excess concentration ────────────────────────────────

class TestPathExcessConcentration:
    """B3: 1/r weighting concentrates cooperation at short range.

    Raw path excess has median ≈ 4.0 (not short). But the cross_12
    correlator weights each triplet by 1/(r_ij · r_il · r_lj), which
    strongly emphasizes short-range contributions. This 1/r weighting
    is why r_cut=3 converges (test_9 F8).

    Ref: internal/0_tracker_v5.md, Phase 2 B3.
    """

    def test_weighted_path_excess_shorter_than_unweighted(self):
        """1/r-weighted path excess median < unweighted median.

        The Green function 1/r weighting shifts the effective path
        excess distribution toward shorter paths.
        """
        dx, dy = disk_bonds(5)
        N = len(dx)
        dist = np.sqrt((dx[:, None] - dx[None, :])**2
                       + (dy[:, None] - dy[None, :])**2)
        pe_3d = (dist[:, None, :] + dist[None, :, :]
                 - dist[:, :, None])
        # w[i,j,l] = 1 / (dist[i,j] * dist[i,l] * dist[j,l])
        d_ij = dist[:, :, None]   # (N,N,1)
        d_il = dist[:, None, :]   # (N,1,N)
        d_jl = dist[None, :, :]   # (1,N,N)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = 1.0 / (d_ij * d_il * d_jl)
        ii = np.arange(N)
        mask = ((ii[:, None, None] != ii[None, :, None])
                & (ii[:, None, None] != ii[None, None, :])
                & (ii[None, :, None] != ii[None, None, :]))
        w[~mask] = 0
        w[np.isinf(w)] = 0

        pe_flat = pe_3d[mask]
        w_flat = w[mask]

        med_unweighted = np.median(pe_flat)
        # Weighted median: sort by pe, find where cumulative weight = 50%
        order = np.argsort(pe_flat)
        pe_sorted = pe_flat[order]
        w_sorted = w_flat[order]
        cum_w = np.cumsum(w_sorted)
        cum_w /= cum_w[-1]
        med_weighted = pe_sorted[np.searchsorted(cum_w, 0.5)]

        print(f"  Unweighted median: {med_unweighted:.2f}")
        print(f"  1/r-weighted median: {med_weighted:.2f}")
        assert med_weighted < med_unweighted, \
            f"Weighted median {med_weighted:.2f} not < " \
            f"unweighted {med_unweighted:.2f}"

    def test_short_range_dominates_cross12(self):
        """Triplets with pe < 3.0 contribute > 50% of |cross_12| weight.

        Even though only 40% of triplets have pe < 3.0, they contribute
        >50% of the 1/r³-weighted sum. This is why r_cut=3 converges.
        """
        dx, dy = disk_bonds(5)
        N = len(dx)
        dist = np.sqrt((dx[:, None] - dx[None, :])**2
                       + (dy[:, None] - dy[None, :])**2)
        pe_3d = (dist[:, None, :] + dist[None, :, :]
                 - dist[:, :, None])
        d_ij = dist[:, :, None]
        d_il = dist[:, None, :]
        d_jl = dist[None, :, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            w = 1.0 / (d_ij * d_il * d_jl)
        ii = np.arange(N)
        mask = ((ii[:, None, None] != ii[None, :, None])
                & (ii[:, None, None] != ii[None, None, :])
                & (ii[None, :, None] != ii[None, None, :]))
        w[~mask] = 0
        w[np.isinf(w)] = 0

        pe_flat = pe_3d[mask]
        w_flat = w[mask]

        short = pe_flat < 3.0
        frac_count = np.mean(short)
        frac_weight = np.sum(w_flat[short]) / np.sum(w_flat)

        print(f"  frac(pe<3) by count: {100*frac_count:.1f}%")
        print(f"  frac(pe<3) by 1/r³ weight: {100*frac_weight:.1f}%")
        assert frac_weight > frac_count, \
            f"1/r³ weight ({frac_weight:.3f}) not > count ({frac_count:.3f})"
        assert frac_weight > 0.50, \
            f"Short-range weight = {frac_weight:.3f}, expected > 0.50"


# ── A2: Integrand CV monotonic in alpha ──────────────────────────

class TestIntegrandCVMonotonic:
    """A2: MS integrand CV decreases monotonically with α.

    No minimum at α=0.25 despite V(k) CV=0 there. Stronger coupling
    always gives larger shift → flatter integrand. The V(k) constancy
    at α=0.25 does not dominate the total integrand CV.

    Ref: internal/0_tracker_v5.md, Phase 2 A2.
    """

    def test_cv_decreases_with_alpha(self):
        """CV(sin²k · σ_ms) monotonically decreasing over 5 alpha values."""
        dx, dy = disk_bonds(5)
        k = np.array(k_vals_7)
        sin2k = np.sin(k)**2

        alpha_list = [0.20, 0.25, 0.30, 0.35, 0.40]
        cv_list = []
        for alpha in alpha_list:
            _, sm = sigma_tr_born_ms(dx, dy, k, alpha, n_theta=50, n_phi=100)
            c = cv(sin2k * sm)
            cv_list.append(c)
            print(f"  α={alpha:.2f}: CV={c:.1f}%")

        assert all(cv_list[i] > cv_list[i + 1]
                   for i in range(len(cv_list) - 1)), \
            f"CV not monotonically decreasing: {cv_list}"


# ── A1: κ MS vs FDTD — N_eff mechanism validation ────────────

class TestKappaConsistency:
    """A1: MS/FDTD normalization gap is per-bond, not collective.

    MS σ_tr is ~15× FDTD at α=0.30 (varies 8–20× with k). This gap
    comes entirely from per-bond σ normalization (scalar MS vs vector
    FDTD). The collective N_eff is correctly predicted by MS:

    N_eff_MS(k) / N_eff_FDTD(k) ≈ 1.0 (mean 1.05, CV=14%).

    This validates the forward decoherence mechanism: MS correctly
    predicts how N bonds interact, even though single-bond σ has an
    overall normalization offset.

    Ref: internal/0_tracker_v5.md, Phase 2 A1.
    """

    def test_neff_ratio_near_one(self):
        """N_eff_MS / N_eff_FDTD ≈ 1 at all k.

        N_eff(k) = σ_ring(k) / σ_bond(k). If the MS/FDTD gap is
        purely per-bond, the ratio of N_eff values = 1.

        FDTD data: sigma_bond at α=0.30, sigma_ring[5] at α=0.30.
        Both on k_vals_7 grid (verified by assertion).
        """
        from data.sigma_bond import k_vals as k_bond_grid
        from data.sigma_ring import k_vals as k_ring_grid

        dx_1, dy_1 = np.array([0.0]), np.array([0.0])
        dx_5, dy_5 = disk_bonds(5)
        k = np.array(k_vals_7)

        # Verify k-grids are aligned
        assert np.allclose(k, k_bond_grid), \
            "k_vals_7 ≠ sigma_bond k-grid"
        assert np.allclose(k, k_ring_grid), \
            "k_vals_7 ≠ sigma_ring k-grid"

        _, sm_1 = sigma_tr_born_ms(dx_1, dy_1, k, ALPHA_REF,
                                    n_theta=50, n_phi=100)
        _, sm_5 = sigma_tr_born_ms(dx_5, dy_5, k, ALPHA_REF,
                                    n_theta=50, n_phi=100)
        fdtd_1 = np.array(sigma_bond_fdtd)   # α=0.30, single z-bond
        fdtd_5 = np.array(sigma_ring[5])       # α=0.30, R=5

        ratio_bond = sm_1 / fdtd_1
        ratio_ring = sm_5 / fdtd_5
        neff_ratio = ratio_ring / ratio_bond

        mean_r = np.mean(neff_ratio)
        cv_r = 100 * np.std(neff_ratio) / mean_r
        print(f"  N_eff_MS/N_eff_FDTD at each k:")
        for i, kv in enumerate(k):
            print(f"    k={kv:.1f}: {neff_ratio[i]:.3f}")
        print(f"  Mean: {mean_r:.3f}, CV: {cv_r:.1f}%")

        assert 0.8 < mean_r < 1.2, \
            f"Mean N_eff ratio = {mean_r:.3f}, expected in [0.8, 1.2]"
        assert cv_r < 20, \
            f"N_eff ratio CV = {cv_r:.1f}%, expected < 20%"

    def test_normalization_gap_is_per_bond(self):
        """MS/FDTD ratio for ring ≈ ratio for single bond (same gap).

        If the gap were collective (from N_eff), the ring ratio would
        differ from the single-bond ratio.

        FDTD data at α=0.30 = ALPHA_REF on k_vals_7 grid.
        """
        dx_1, dy_1 = np.array([0.0]), np.array([0.0])
        dx_5, dy_5 = disk_bonds(5)
        k = np.array(k_vals_7)

        _, sm_1 = sigma_tr_born_ms(dx_1, dy_1, k, ALPHA_REF,
                                    n_theta=50, n_phi=100)
        _, sm_5 = sigma_tr_born_ms(dx_5, dy_5, k, ALPHA_REF,
                                    n_theta=50, n_phi=100)
        fdtd_1 = np.array(sigma_bond_fdtd)   # α=0.30, single z-bond
        fdtd_5 = np.array(sigma_ring[5])       # α=0.30, R=5

        ratio_bond = sm_1 / fdtd_1
        ratio_ring = sm_5 / fdtd_5

        mean_bond = np.mean(ratio_bond)
        mean_ring = np.mean(ratio_ring)
        print(f"  Single-bond MS/FDTD: {mean_bond:.1f}")
        print(f"  Ring R=5 MS/FDTD: {mean_ring:.1f}")
        print(f"  Ratio: {mean_ring/mean_bond:.3f}")

        assert 0.8 < mean_ring / mean_bond < 1.2, \
            f"Ring/bond gap ratio = {mean_ring/mean_bond:.3f}, " \
            f"expected in [0.8, 1.2]"


# ── B6: σ_tr R-scaling ───────────────────────────────────────

class TestRScaling:
    """B6: σ_tr ~ R^{3/2} from stationary phase, MS reduces R-scaling.

    Born σ_tr_born ~ R^{3/2} at fixed k (from coherent cone area R²
    minus transport weight ~R^{-1/2}). Verified: p_born ≈ 1.5.

    MS reduces R-scaling (p_ms < p_born) because forward decoherence
    (suppression at low k) increases with R — larger disks have more
    destructive interference. FDTD gives p ≈ 1.63 (higher than both).

    Ref: internal/0_tracker_v5.md, Phase 2 B6.
    """

    def test_born_r_exponent_near_3_over_2(self):
        """Born σ_tr ~ R^p with p in [1.3, 1.7] at fixed k (near 3/2)."""
        k_fixed = [0.5, 0.9]
        for kv in k_fixed:
            sb_list = []
            R_vals = [3, 5, 7, 9]
            for R in R_vals:
                dx, dy = disk_bonds(R)
                sb, _ = sigma_tr_born_ms(dx, dy, np.array([kv]), ALPHA_REF,
                                          n_theta=50, n_phi=100)
                sb_list.append(sb[0])

            p = log_log_slope(np.array(R_vals, dtype=float),
                              np.array(sb_list))[0]
            print(f"  k={kv}: p_born={p:.3f}")
            assert 1.3 < p < 1.7, \
                f"Born R-exponent {p:.3f} at k={kv} outside [1.3, 1.7]"

    def test_ms_reduces_r_scaling(self):
        """MS R-exponent < Born R-exponent (forward decoherence grows with R).

        Larger disks have more bonds → more destructive interference →
        greater suppression at low k → slower growth of σ_ms with R.
        """
        k_fixed = [0.5, 0.9]
        for kv in k_fixed:
            sb_list, sm_list = [], []
            R_vals = [3, 5, 7, 9]
            for R in R_vals:
                dx, dy = disk_bonds(R)
                sb, sm = sigma_tr_born_ms(dx, dy, np.array([kv]), ALPHA_REF,
                                           n_theta=50, n_phi=100)
                sb_list.append(sb[0])
                sm_list.append(sm[0])

            R_arr = np.array(R_vals, dtype=float)
            p_born = log_log_slope(R_arr, np.array(sb_list))[0]
            p_ms = log_log_slope(R_arr, np.array(sm_list))[0]
            print(f"  k={kv}: p_born={p_born:.3f}, p_ms={p_ms:.3f} "
                  f"(Δ={p_ms-p_born:+.3f})")
            assert p_ms < p_born, \
                f"MS R-exponent {p_ms:.3f} not < Born {p_born:.3f} " \
                f"at k={kv}"
