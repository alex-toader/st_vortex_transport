"""
Route 6: 2D Aharonov-Bohm verification of scattering pipeline.

Question: Does 2D FDTD scattering on a Peierls vortex match the analytic
AB cross-section σ_tr = 2·sin²(πα)/k ?

At α=1/2, R(π) = -I — each displacement component independently sees
a scalar phase flip. The vector problem reduces exactly to scalar AB.

RESULT: PARTIAL.

  α=0.3 (existing test regime):
    r_m=35: slope = -0.94, R² = 0.98  — matches AB slope of -1.0

  α=0.5 (exact AB mapping):
    r_m=35: slope = -0.74, R² = 0.97  — significantly worse

  Alpha ratio σ_tr(0.5)/σ_tr(0.3):
    Measured: 1.02–1.43 (increases with k)
    AB prediction: sin²(π/2)/sin²(0.3π) = 1.528
    → Prefactor scaling with α does NOT match AB at α=0.5

Absolute prefactor check:
  AB analytic: σ_tr(α=0.3, k=0.5) = 2·sin²(0.3π)/0.5 = 2.618
  Measured:    σ_tr(α=0.3, k=0.5, r_m=35) = 31.49
  Ratio: ~12×. The normalization of compute_ring_f2 is uncalibrated
  relative to the AB convention. Only slope (k-dependence) is validated,
  NOT the absolute cross-section value.

Interpretation:
  - At α=0.3 (weak scattering), AB works well (slope ≈ -1)
  - At α=0.5 (maximal scattering, R(π)=-I), slope degrades to -0.74.
    Candidate causes: NNN bonds ungauged (33% of coupling), finite core,
    multiple scattering. These are plausible but NOT separated.
  - Route 6 validates the PIPELINE (slope correct at α=0.3) but cannot
    verify the PREFACTOR at the physically relevant α=1/2.

Cause separation (not yet done):
  Running with K2=0 (NN-only, no NNN) at α=0.5 would isolate the NNN
  contribution. If slope recovers to ~-1.0 → NNN ungauged is the main
  cause. If not → finite core or multiple scattering dominates.

Near-field contamination:
  r_m=25 gives R²=0.81 (α=0.3) vs R²=0.98 at r_m=35. Scattered field
  not fully separated from incident at small r_m. All conclusions use
  r_m=35 only.

Known issue in existing code:
  make_wave_packet uses c_eff = √(K1+K2) = 1.225 (central force) but
  the force law is scalar Laplacian with c = √(K1+2K2) = 1.414.
  Mismatch: 15.4%. Does not affect slopes (relative measurement) but
  contributes to the uncalibrated absolute prefactor.

Uses existing infrastructure from src/3_bombardment/:
  - gauge_coupling.py (Peierls monopole)
  - elastic_lattice_2d.py (FDTD + PML)
  - measurement_ring.py (angular σ_tr integration)

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/3_route6_ab_verification.py

--- RAW OUTPUT ---

=== alpha=0.3 ===
  r_m=25: slope=-0.87, R2=0.81
    sigma_tr = [36.90 38.58 41.25 23.27 15.62 10.42]
  r_m=35: slope=-0.94, R2=0.98
    sigma_tr = [66.45 52.01 43.76 31.49 21.05 15.75]

=== alpha=0.5 ===
  r_m=25: slope=-0.68, R2=0.76
    sigma_tr = [39.00 41.83 46.22 27.40 20.71 14.80]
  r_m=35: slope=-0.74, R2=0.97
    sigma_tr = [67.93 56.58 50.00 37.95 26.58 22.46]

=== Alpha ratio (0.5 / 0.3) at r_m=35 ===
  k=0.20: ratio=1.022 (AB predicts 1.528)
  k=0.30: ratio=1.088 (AB predicts 1.528)
  k=0.40: ratio=1.143 (AB predicts 1.528)
  k=0.50: ratio=1.205 (AB predicts 1.528)
  k=0.70: ratio=1.263 (AB predicts 1.528)
  k=1.00: ratio=1.426 (AB predicts 1.528)

Slope summary:
  alpha=0.3: slope=-0.94 (AB: -1.0) — GOOD
  alpha=0.5: slope=-0.74 (AB: -1.0) — DEGRADED

Mar 2026
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', '3_bombardment'))
from physics.gauge_coupling import elastic_force_peierls_dirac
from physics.elastic_lattice_2d import (make_damping, make_wave_packet,
                                         run_fdtd, estimate_n_steps)
from physics.measurement_ring import (make_ring_points_multi,
                                       compute_ring_f2, integrate_sigma)


# ─────────────────────────────────────────────────────────
# Parameters (match existing test_zone_a_scattering.py)
# ─────────────────────────────────────────────────────────

L = 200
K1, K2, M = 1.0, 0.5, 1.0
DT = 0.4
DW = 25
DS = 0.15
SX = 80.0
X_START = DW + 5
N_REC = 100
ANGLES = np.radians(np.arange(0, 181, 5))  # 37 points, hemisphere
KS = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
R_MS = [25, 35]


def run_scan(alpha):
    """Run FDTD scattering scan at given alpha, return σ_tr per (r_m, k)."""
    gamma = make_damping(L, DW, DS)
    all_iy, all_ix, slices = make_ring_points_multi(L, R_MS, ANGLES)

    f_ref = lambda ux, uy: elastic_force_peierls_dirac(ux, uy, 0.0, K1, K2)
    f_def = lambda ux, uy: elastic_force_peierls_dirac(ux, uy, alpha, K1, K2)

    results = {r: [] for r in R_MS}
    for k0 in KS:
        ns = estimate_n_steps(k0, L, X_START, SX, max(R_MS), DT, N_REC, K1, K2)
        ux0, vx0 = make_wave_packet(L, k0, X_START, SX, K1, K2)

        ref = run_fdtd(f_ref, ux0, vx0, gamma, DT, ns,
                       all_iy, all_ix, rec_n=N_REC, m=M)
        out = run_fdtd(f_def, ux0, vx0, gamma, DT, ns,
                       all_iy, all_ix, rec_n=N_REC, m=M)

        for r_m in R_MS:
            s, e = slices[r_m]
            f2 = compute_ring_f2(out['ux'][:, s:e], out['uy'][:, s:e],
                                 ref['ux'][:, s:e], ref['uy'][:, s:e], r_m)
            _, sig_tr = integrate_sigma(f2, ANGLES)
            results[r_m].append(sig_tr)

    return results


def fit_power_law(ks, sigma_tr):
    """Fit σ_tr = A·k^n, return (slope, R², A)."""
    log_k = np.log(np.array(ks))
    log_st = np.log(np.array(sigma_tr))
    slope, log_A = np.polyfit(log_k, log_st, 1)
    fitted = slope * log_k + log_A
    ss_res = np.sum((log_st - fitted)**2)
    ss_tot = np.sum((log_st - np.mean(log_st))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, r2, np.exp(log_A)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = {}

    for alpha in [0.3, 0.5]:
        print(f"=== alpha={alpha} ===")
        results = run_scan(alpha)
        all_results[alpha] = results

        for r_m in R_MS:
            st = np.array(results[r_m])
            slope, r2, A = fit_power_law(KS, st)
            st_str = " ".join(f"{v:.2f}" for v in st)
            print(f"  r_m={r_m}: slope={slope:.2f}, R²={r2:.2f}")
            print(f"    sigma_tr = [{st_str}]")
        print()

    # Alpha ratio at r_m=35
    AB_ratio = np.sin(np.pi * 0.5)**2 / np.sin(np.pi * 0.3)**2
    print(f"=== Alpha ratio (0.5 / 0.3) at r_m=35 ===")
    st03 = np.array(all_results[0.3][35])
    st05 = np.array(all_results[0.5][35])
    for i, k in enumerate(KS):
        r = st05[i] / st03[i]
        print(f"  k={k:.2f}: ratio={r:.3f} (AB predicts {AB_ratio:.3f})")

    # Summary
    print()
    print("Slope summary:")
    for alpha in [0.3, 0.5]:
        slope, r2, _ = fit_power_law(KS, all_results[alpha][35])
        quality = "GOOD" if abs(slope + 1.0) < 0.15 else "DEGRADED"
        print(f"  alpha={alpha}: slope={slope:.2f} (AB: -1.0) — {quality}")
