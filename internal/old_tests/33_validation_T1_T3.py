"""
Route 33: Validation tests T1 + T3.

T1: α ↔ 1−α symmetry — σ_tr(α=0.7) must equal σ_tr(α=0.3).
  Both α run from same references in same script.
  Symmetry is exact: s_phi → −s_phi, under uy → −uy leaves |u_scat|²
  invariant at every sphere point. Threshold: 1e-8 (floating-point exact).
  Tests both NN and NNN. Secondary cross-check vs file 28 data.

T3: Angular grid convergence — 13θ×24φ (312 pts) vs 25θ×48φ (1200 pts).
  At k=0.5 and k=2.0, α=0.3, R=5. Both NN and NNN.
  If < 2% difference → current grid sufficient.
  FDTD is fully deterministic — separate refs at different sphere grids
  produce identical physics, just sampled at different points.

Results (R=5, L=80, 371s total):

  T1 — α ↔ 1−α symmetry:

    PRIMARY (same refs, exact symmetry):
      k     NN(0.3)   NN(0.7)  rel_diff   NNN(0.3)  NNN(0.7)  rel_diff
     0.30    40.743    40.743   1.2e-15     54.255    54.255   0.0e+00
     0.50    14.158    14.158   7.5e-16     25.436    25.436   1.4e-16
     0.70     7.694     7.694   3.5e-16     18.081    18.081   3.9e-16
     0.90     5.048     5.048   3.5e-16     14.234    14.234   1.3e-16
     1.10     3.776     3.776   2.4e-16     12.243    12.243   0.0e+00
     1.30     3.138     3.138   5.7e-16     10.786    10.786   1.7e-16
     1.50     2.806     2.806   3.2e-16      9.978     9.978   0.0e+00

    NN:  max |rel_diff| = 1.2e-15. PASS.
    NNN: max |rel_diff| = 3.9e-16. PASS.
    Symmetry verified to floating-point precision.

    SECONDARY (vs file 28): NN max diff = 0.016%, NNN max diff = 0.004%.
    Reproducibility confirmed.

  T3 — Angular grid convergence:

      k    NN(13×24)  NN(25×48)  diff%   NNN(13×24)  NNN(25×48)  diff%
     0.5    14.158     14.296    0.98%     25.436      25.690    1.00%
     2.0     3.011      3.011    0.01%      9.728       9.570    1.63%

    NN:  max diff = 0.98%. PASS.
    NNN: max diff = 1.63%. PASS.
    Current 13×24 grid sufficient (< 2% at all k, both gaugings).
    Largest deviation: NNN at k=2.0 (1.63%) — high-k NNN pattern has more
    angular structure. Still well within tolerance.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/33_validation_T1_T3.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../src/1_foam'))
from parallel_fdtd import compute_references, compute_scattering
from gauge_3d import make_vortex_force
from elastic_3d import make_damping_3d, run_fdtd_3d
from scattering_3d import (make_sphere_points, compute_sphere_f2,
                           integrate_sigma_3d)

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 5
N_WORKERS = 4

# Standard angular grid
thetas_std = np.linspace(0, np.pi, 13)
phis_std = np.linspace(0, 2 * np.pi, 24, endpoint=False)

# Fine angular grid
thetas_fine = np.linspace(0, np.pi, 25)
phis_fine = np.linspace(0, 2 * np.pi, 48, endpoint=False)

k_vals_t1 = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
k_vals_t3 = np.array([0.5, 2.0])

# File 28 data for secondary cross-check only
sigma_nn_03_f28 = np.array([40.743, 14.158, 7.694, 5.048, 3.776, 3.138, 2.806])
sigma_nnn_03_f28 = np.array([54.255, 25.436, 18.081, 14.234, 12.243, 10.786, 9.978])


def run_nnn_serial(alpha, k_vals, refs, gamma, iz, iy, ix, thetas, phis):
    """Run NNN scattering serially for one alpha."""
    f_nnn = make_vortex_force(alpha, R_LOOP, L, K1, K2, gauge_nnn=True)
    sigma = np.zeros(len(k_vals))
    for j, k0 in enumerate(k_vals):
        ref, ux0, vx0, ns = refs[k0]
        d = run_fdtd_3d(f_nnn, ux0.copy(), vx0.copy(), gamma, DT, ns,
                        rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        _, st = integrate_sigma_3d(f2, thetas, phis)
        sigma[j] = st
    return sigma


if __name__ == '__main__':
    t0 = time.time()

    # ═══════════════════════════════════════════════════════
    # T1: α ↔ 1−α symmetry
    # ═══════════════════════════════════════════════════════
    print("=" * 70)
    print("T1: α ↔ 1−α symmetry — σ_tr(0.7) vs σ_tr(0.3)")
    print("=" * 70)
    print(f"  R={R_LOOP}, L={L}, r_m={r_m}, N_WORKERS={N_WORKERS}")
    print(f"  k = {list(k_vals_t1)}")
    print()

    # Shared references (standard grid)
    print("Computing T1 references (7 k-pts)...")
    t1 = time.time()
    refs_t1 = compute_references(k_vals_t1, L, DW, DS, DT, sx, r_m,
                                 thetas_std, phis_std, K1, K2,
                                 n_workers=N_WORKERS)
    dt_refs = time.time() - t1
    print(f"  Done ({dt_refs:.0f}s)")

    # NN at α=0.3 (parallel)
    print("NN scattering at α=0.3...")
    t1 = time.time()
    sigma_nn_03 = compute_scattering(k_vals_t1, refs_t1, 0.3, R_LOOP,
                                     L, DW, DS, DT, r_m, thetas_std, phis_std,
                                     K1, K2, n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")

    # NN at α=0.7 (parallel)
    print("NN scattering at α=0.7...")
    t1 = time.time()
    sigma_nn_07 = compute_scattering(k_vals_t1, refs_t1, 0.7, R_LOOP,
                                     L, DW, DS, DT, r_m, thetas_std, phis_std,
                                     K1, K2, n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")

    # NNN serial setup
    gamma = make_damping_3d(L, DW, DS)
    iz_std, iy_std, ix_std = make_sphere_points(L, r_m, thetas_std, phis_std)

    # NNN at α=0.3 (serial)
    print("NNN scattering at α=0.3 (serial)...")
    t1 = time.time()
    sigma_nnn_03 = run_nnn_serial(0.3, k_vals_t1, refs_t1, gamma,
                                  iz_std, iy_std, ix_std, thetas_std, phis_std)
    print(f"  Done ({time.time()-t1:.0f}s)")

    # NNN at α=0.7 (serial)
    print("NNN scattering at α=0.7 (serial)...")
    t1 = time.time()
    sigma_nnn_07 = run_nnn_serial(0.7, k_vals_t1, refs_t1, gamma,
                                  iz_std, iy_std, ix_std, thetas_std, phis_std)
    print(f"  Done ({time.time()-t1:.0f}s)")

    # Sanity: σ > 0
    assert np.all(sigma_nn_03 > 0), "σ_NN(0.3) has zero entries!"
    assert np.all(sigma_nn_07 > 0), "σ_NN(0.7) has zero entries!"
    assert np.all(sigma_nnn_03 > 0), "σ_NNN(0.3) has zero entries!"
    assert np.all(sigma_nnn_07 > 0), "σ_NNN(0.7) has zero entries!"

    # Primary test: α=0.3 vs α=0.7 (exact symmetry, same refs)
    print()
    print("PRIMARY: α=0.3 vs α=0.7 (same refs, exact symmetry)")
    print(f"  {'k':>5s}  {'NN(0.3)':>9s}  {'NN(0.7)':>9s}  {'rel_diff':>10s}  "
          f"{'NNN(0.3)':>10s}  {'NNN(0.7)':>10s}  {'rel_diff':>10s}")
    print(f"  {'-'*70}")
    for j, k0 in enumerate(k_vals_t1):
        nn_rd = abs(sigma_nn_03[j] - sigma_nn_07[j]) / sigma_nn_03[j]
        nnn_rd = abs(sigma_nnn_03[j] - sigma_nnn_07[j]) / sigma_nnn_03[j]
        print(f"  {k0:5.2f}  {sigma_nn_03[j]:9.3f}  {sigma_nn_07[j]:9.3f}  "
              f"{nn_rd:10.2e}  {sigma_nnn_03[j]:10.3f}  "
              f"{sigma_nnn_07[j]:10.3f}  {nnn_rd:10.2e}")

    nn_max_rd = np.max(np.abs(sigma_nn_03 - sigma_nn_07) / sigma_nn_03)
    nnn_max_rd = np.max(np.abs(sigma_nnn_03 - sigma_nnn_07) / sigma_nnn_03)
    print()
    print(f"  NN:  max |rel_diff| = {nn_max_rd:.2e}  "
          f"({'PASS' if nn_max_rd < 1e-8 else 'FAIL'})")
    print(f"  NNN: max |rel_diff| = {nnn_max_rd:.2e}  "
          f"({'PASS' if nnn_max_rd < 1e-8 else 'FAIL'})")

    # Secondary: compare this run's α=0.3 with file 28 (reproducibility check)
    print()
    print("SECONDARY: α=0.3 this run vs file 28 (reproducibility)")
    nn_f28_diffs = np.abs(sigma_nn_03 - sigma_nn_03_f28) / sigma_nn_03_f28 * 100
    nnn_f28_diffs = np.abs(sigma_nnn_03 - sigma_nnn_03_f28) / sigma_nnn_03_f28 * 100
    print(f"  NN  vs f28: max diff = {np.max(nn_f28_diffs):.4f}%")
    print(f"  NNN vs f28: max diff = {np.max(nnn_f28_diffs):.4f}%")

    # ═══════════════════════════════════════════════════════
    # T3: Angular grid convergence
    # ═══════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("T3: Angular grid convergence — 13×24 vs 25×48")
    print("=" * 70)
    print(f"  k = {list(k_vals_t3)}, α=0.3, R={R_LOOP}")
    print(f"  Standard: {len(thetas_std)}θ × {len(phis_std)}φ "
          f"= {len(thetas_std) * len(phis_std)} pts")
    print(f"  Fine:     {len(thetas_fine)}θ × {len(phis_fine)}φ "
          f"= {len(thetas_fine) * len(phis_fine)} pts")
    print(f"  Note: FDTD is deterministic — separate refs at different grids")
    print(f"  produce identical fields, only measurement sampling differs.")
    print()

    # Standard grid
    print("Standard grid refs + NN scattering...")
    t1 = time.time()
    refs_std = compute_references(k_vals_t3, L, DW, DS, DT, sx, r_m,
                                  thetas_std, phis_std, K1, K2,
                                  n_workers=N_WORKERS)
    sigma_std_nn = compute_scattering(k_vals_t3, refs_std, 0.3, R_LOOP,
                                      L, DW, DS, DT, r_m, thetas_std, phis_std,
                                      K1, K2, n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")

    print("Standard grid NNN scattering (serial)...")
    t1 = time.time()
    sigma_std_nnn = run_nnn_serial(0.3, k_vals_t3, refs_std, gamma,
                                   iz_std, iy_std, ix_std, thetas_std, phis_std)
    print(f"  Done ({time.time()-t1:.0f}s)")

    # Fine grid
    print("Fine grid refs + NN scattering...")
    t1 = time.time()
    refs_fine_t3 = compute_references(k_vals_t3, L, DW, DS, DT, sx, r_m,
                                      thetas_fine, phis_fine, K1, K2,
                                      n_workers=N_WORKERS)
    sigma_fine_nn = compute_scattering(k_vals_t3, refs_fine_t3, 0.3, R_LOOP,
                                       L, DW, DS, DT, r_m, thetas_fine, phis_fine,
                                       K1, K2, n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")

    iz_fine, iy_fine, ix_fine = make_sphere_points(L, r_m, thetas_fine, phis_fine)
    print("Fine grid NNN scattering (serial)...")
    t1 = time.time()
    sigma_fine_nnn = run_nnn_serial(0.3, k_vals_t3, refs_fine_t3, gamma,
                                    iz_fine, iy_fine, ix_fine,
                                    thetas_fine, phis_fine)
    print(f"  Done ({time.time()-t1:.0f}s)")

    # Comparison
    print()
    print(f"  {'k':>5s}  {'NN(13×24)':>10s}  {'NN(25×48)':>10s}  {'diff%':>7s}  "
          f"{'NNN(13×24)':>11s}  {'NNN(25×48)':>11s}  {'diff%':>7s}")
    print(f"  {'-'*70}")
    for j, k0 in enumerate(k_vals_t3):
        nn_diff = abs(sigma_fine_nn[j] - sigma_std_nn[j]) / sigma_std_nn[j] * 100
        nnn_diff = abs(sigma_fine_nnn[j] - sigma_std_nnn[j]) / sigma_std_nnn[j] * 100
        print(f"  {k0:5.1f}  {sigma_std_nn[j]:10.4f}  {sigma_fine_nn[j]:10.4f}  "
              f"{nn_diff:6.2f}%  {sigma_std_nnn[j]:11.4f}  "
              f"{sigma_fine_nnn[j]:11.4f}  {nnn_diff:6.2f}%")

    nn_max_diff = max(abs(sigma_fine_nn[j] - sigma_std_nn[j]) / sigma_std_nn[j] * 100
                      for j in range(len(k_vals_t3)))
    nnn_max_diff = max(abs(sigma_fine_nnn[j] - sigma_std_nnn[j]) / sigma_std_nnn[j] * 100
                       for j in range(len(k_vals_t3)))
    print()
    print(f"  NN:  max diff = {nn_max_diff:.2f}%  "
          f"({'PASS' if nn_max_diff < 2 else 'FAIL'})")
    print(f"  NNN: max diff = {nnn_max_diff:.2f}%  "
          f"({'PASS' if nnn_max_diff < 2 else 'FAIL'})")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
