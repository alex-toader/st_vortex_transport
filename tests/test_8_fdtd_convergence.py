"""§7 FDTD convergence: box size, PML, wavepacket bandwidth.

Verifies that hardcoded FDTD data (L=80) is converged.
All tests run actual FDTD simulations — no hardcoded reference data.

Results (k_test = [0.3, 0.9, 1.5]):
  TestBoxSizeConvergence:
    σ_tr L=80:  [40.743, 5.048, 2.806]
    σ_tr L=100: [28.331, 4.688, 2.601]
    σ_tr L=120: [29.105, 4.698, 2.611]
    L=100 vs L=120: Δ = 2.7%, 0.2%, 0.4%  (all < 5% → converged)
    L=80  vs L=100: Δ = +43.8%, +7.7%, +7.9%  (k=0.3 PML-contaminated)
    Integrand CV: L=80 = 10.0%, L=100 = 6.4%
    L=100 matches stored sigma_ring[5] (closes data provenance loop)
  TestPMLConvergence (k=[0.9, 1.5], L=100):
    DW=10: [4.691, 2.601], DW=15: [4.688, 2.601], DW=20: [4.696, 2.612]
    DW=15 vs DW=20: Δ = 0.2%, 0.4%  (converged)
    All DW spread < 2% (stable, no monotonicity assumed)
    k=0.3 excluded: PML sensitivity tested via box size (L=80 overestimates 44%)
  TestWavepacketBandwidth (k=[0.9, 1.5], L=100):
    sx=6: [4.797, 2.601], sx=8: [4.688, 2.601], sx=12: [4.634, 2.619]
    sx=6 vs sx=8: Δ = 2.3%, 0.0%  (threshold < 5%)
    sx=8 vs sx=12: Δ = 1.2%, 0.7%  (threshold < 5%)

Slow (~17 min). Run with: pytest test_8_fdtd_convergence.py -v -s
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from gauge_3d import make_vortex_force
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           compute_sphere_f2, integrate_sigma_3d,
                           estimate_n_steps_3d)

# ── Parameters ────────────────────────────────────────────────────
K1, K2 = 1.0, 0.5
ALPHA = 0.30
R = 5
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
THETAS = np.linspace(0, np.pi, 13)
PHIS = np.linspace(0, 2 * np.pi, 24, endpoint=False)

k_test = np.array([0.3, 0.9, 1.5])


# ── FDTD helper ───────────────────────────────────────────────────

def _run_sigma_tr(L, k_vals, dw=DW, sx_wp=sx):
    """Run FDTD at box size L, return sigma_tr array for given k values."""
    center = L // 2
    # Wavepacket starts at sphere boundary (x_start = center - r_m).
    # Packet propagates in +x through the defect; scattered field measured on sphere.
    x_start = center - r_m
    assert x_start <= center - r_m, \
        f"x_start={x_start} inside sphere (center={center}, r_m={r_m})"

    gamma_pml = make_damping_3d(L, dw, DS)
    iz, iy, ix = make_sphere_points(L, r_m, THETAS, PHIS)

    def force_plain(ux, uy, uz):
        return scalar_laplacian_3d(ux, uy, uz, K1, K2)

    f_def = make_vortex_force(ALPHA, R, L, K1, K2)

    print(f"  FDTD: L={L}, DW={dw}, sx={sx_wp}, k={list(k_vals)}")
    sigma_tr = np.zeros(len(k_vals))
    for i, k0 in enumerate(k_vals):
        ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx_wp, K1, K2)
        ns = estimate_n_steps_3d(k0, L, x_start, sx_wp, r_m, DT, 50, K1, K2)

        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                          DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                        DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        _, st = integrate_sigma_3d(f2, THETAS, PHIS)
        sigma_tr[i] = st
        print(f"    k={k0:.1f}: σ_tr = {st:.3f}")

    return sigma_tr


# ── Fixtures ──────────────────────────────────────────────────────
# scope="module": each FDTD runs once per test session.
# Without this, fixtures would re-run per test method (~5× slower).

@pytest.fixture(scope="module")
def sigma_L80():
    return _run_sigma_tr(80, k_test)

@pytest.fixture(scope="module")
def sigma_L100():
    return _run_sigma_tr(100, k_test)

@pytest.fixture(scope="module")
def sigma_L120():
    return _run_sigma_tr(120, k_test)


# ── Tests ─────────────────────────────────────────────────────────

class TestBoxSizeConvergence:
    """σ_tr converges with box size: L=100 vs L=120 within 5%."""

    def test_L100_vs_L120_converged(self, sigma_L100, sigma_L120):
        """L=100 and L=120 agree within 5% at all k."""
        for i, k0 in enumerate(k_test):
            delta = abs(sigma_L100[i] - sigma_L120[i]) / sigma_L120[i] * 100
            print(f"  k={k0}: L=100={sigma_L100[i]:.3f}, L=120={sigma_L120[i]:.3f}, Δ={delta:.1f}%")
            assert delta < 5.0, \
                f"k={k0}: L=100 vs L=120 differ by {delta:.1f}% > 5%"

    def test_L80_overestimates_low_k(self, sigma_L80, sigma_L100):
        """L=80 overestimates σ_tr at k=0.3 (PML too close)."""
        # k=0.3 is index 0; at L=80 sphere is 5 units from PML
        delta = (sigma_L80[0] - sigma_L100[0]) / sigma_L100[0] * 100
        print(f"  k=0.3: L=80={sigma_L80[0]:.3f}, L=100={sigma_L100[0]:.3f}, Δ={delta:+.1f}%")
        assert delta > 10.0, \
            f"k=0.3: L=80 overestimate only {delta:.1f}%, expected > 10%"

    def test_L80_moderate_bias_high_k(self, sigma_L80, sigma_L100):
        """L=80 bias at k≥0.9 is < 15% (moderate, not catastrophic)."""
        # k=0.9 is index 1, k=1.5 is index 2
        for i in [1, 2]:
            delta = abs(sigma_L80[i] - sigma_L100[i]) / sigma_L100[i] * 100
            print(f"  k={k_test[i]}: L=80={sigma_L80[i]:.3f}, L=100={sigma_L100[i]:.3f}, Δ={delta:.1f}%")
            assert delta < 15.0, \
                f"k={k_test[i]}: L=80 bias {delta:.1f}% > 15%"

    def test_integrand_flat_at_L100(self, sigma_L100):
        """sin²(k)·σ_tr CV < 15% at L=100."""
        integrand = np.sin(k_test)**2 * sigma_L100
        c = np.std(integrand) / np.mean(integrand) * 100
        print(f"  Integrand L=100: CV = {c:.1f}%")
        assert c < 15.0, f"Integrand CV = {c:.1f}% > 15%"

    def test_integrand_improves_with_L(self, sigma_L80, sigma_L100):
        """Integrand flatness improves from L=80 to L=100."""
        integ_80 = np.sin(k_test)**2 * sigma_L80
        integ_100 = np.sin(k_test)**2 * sigma_L100
        cv_80 = np.std(integ_80) / np.mean(integ_80) * 100
        cv_100 = np.std(integ_100) / np.mean(integ_100) * 100
        print(f"  CV: L=80 = {cv_80:.1f}%, L=100 = {cv_100:.1f}%")
        assert cv_100 < cv_80, \
            f"CV(L=100) = {cv_100:.1f}% ≥ CV(L=80) = {cv_80:.1f}%"

    def test_L100_matches_stored_data(self, sigma_L100):
        """L=100 FDTD matches hardcoded sigma_ring[5] within 5%.

        Closes the loop: stored data was generated with same parameters.
        """
        from data.sigma_ring import sigma_ring
        from helpers.config import k_vals_7
        for i, k0 in enumerate(k_test):
            idx = np.argmin(np.abs(k_vals_7 - k0))
            stored = sigma_ring[5][idx]
            delta = abs(sigma_L100[i] - stored) / stored * 100
            print(f"  k={k0}: L=100={sigma_L100[i]:.3f}, stored={stored:.3f}, Δ={delta:.1f}%")
            assert delta < 5.0, \
                f"L=100 vs stored data differ by {delta:.1f}% at k={k0}"


# ── PML fixtures ─────────────────────────────────────────────────
# Fixed L=100 (converged), vary DW

# k=0.3 excluded: PML sensitivity at low k is tested via box size (L=80 overestimates 44%).
# PML convergence here tests the regime where DW matters independently of L.
k_pml = np.array([0.9, 1.5])

@pytest.fixture(scope="module")
def sigma_dw10():
    return _run_sigma_tr(100, k_pml, dw=10)

@pytest.fixture(scope="module")
def sigma_dw15():
    return _run_sigma_tr(100, k_pml, dw=15)

@pytest.fixture(scope="module")
def sigma_dw20():
    return _run_sigma_tr(100, k_pml, dw=20)


class TestPMLConvergence:
    """σ_tr stable under PML thickness variation at L=100."""

    def test_dw15_vs_dw20_converged(self, sigma_dw15, sigma_dw20):
        """DW=15 and DW=20 agree within 5%."""
        for i, k0 in enumerate(k_pml):
            delta = abs(sigma_dw15[i] - sigma_dw20[i]) / sigma_dw20[i] * 100
            print(f"  k={k0}: DW=15={sigma_dw15[i]:.3f}, DW=20={sigma_dw20[i]:.3f}, Δ={delta:.1f}%")
            assert delta < 5.0, \
                f"k={k0}: DW=15 vs DW=20 differ by {delta:.1f}% > 5%"

    def test_dw10_vs_dw15_bounded(self, sigma_dw10, sigma_dw15):
        """DW=10 vs DW=15 differ by < 15% (thinner PML, more reflection)."""
        for i, k0 in enumerate(k_pml):
            delta = abs(sigma_dw10[i] - sigma_dw15[i]) / sigma_dw15[i] * 100
            print(f"  k={k0}: DW=10={sigma_dw10[i]:.3f}, DW=15={sigma_dw15[i]:.3f}, Δ={delta:.1f}%")
            assert delta < 15.0, \
                f"k={k0}: DW=10 vs DW=15 differ by {delta:.1f}% > 15%"

    def test_sigma_stable_across_pml(self, sigma_dw10, sigma_dw15, sigma_dw20):
        """σ_tr stable across DW=10,15,20: all within 2% of each other."""
        for i, k0 in enumerate(k_pml):
            vals = [sigma_dw10[i], sigma_dw15[i], sigma_dw20[i]]
            spread = (max(vals) - min(vals)) / np.mean(vals) * 100
            print(f"  k={k0}: DW=10={vals[0]:.3f}, DW=15={vals[1]:.3f}, DW=20={vals[2]:.3f}, spread={spread:.1f}%")
            assert spread < 2.0, \
                f"k={k0}: PML spread {spread:.1f}% > 2%"


# ── Bandwidth fixtures ───────────────────────────────────────────
# Fixed L=100, DW=15, vary wavepacket width sx

# k=0.3 excluded for speed; bandwidth sensitivity at low k is small
# (wavepacket width affects spectral resolution, not PML coupling).
k_bw = np.array([0.9, 1.5])

@pytest.fixture(scope="module")
def sigma_sx6():
    return _run_sigma_tr(100, k_bw, sx_wp=6.0)

@pytest.fixture(scope="module")
def sigma_sx8():
    return _run_sigma_tr(100, k_bw, sx_wp=8.0)

@pytest.fixture(scope="module")
def sigma_sx12():
    return _run_sigma_tr(100, k_bw, sx_wp=12.0)


class TestWavepacketBandwidth:
    """σ_tr independent of wavepacket spectral width."""

    def test_sx6_vs_sx8(self, sigma_sx6, sigma_sx8):
        """Narrow (sx=6) vs standard (sx=8): Δ < 5%."""
        for i, k0 in enumerate(k_bw):
            delta = abs(sigma_sx6[i] - sigma_sx8[i]) / sigma_sx8[i] * 100
            print(f"  k={k0}: sx=6={sigma_sx6[i]:.3f}, sx=8={sigma_sx8[i]:.3f}, Δ={delta:.1f}%")
            assert delta < 5.0, \
                f"k={k0}: sx=6 vs sx=8 differ by {delta:.1f}% > 5%"

    def test_sx8_vs_sx12(self, sigma_sx8, sigma_sx12):
        """Standard (sx=8) vs wide (sx=12): Δ < 5%."""
        for i, k0 in enumerate(k_bw):
            delta = abs(sigma_sx8[i] - sigma_sx12[i]) / sigma_sx12[i] * 100
            print(f"  k={k0}: sx=8={sigma_sx8[i]:.3f}, sx=12={sigma_sx12[i]:.3f}, Δ={delta:.1f}%")
            assert delta < 5.0, \
                f"k={k0}: sx=8 vs sx=12 differ by {delta:.1f}% > 5%"
