"""
Test D: Gauge invariance — Dirac surface position independence.

Physical observables (σ_tr) must not depend on where we place the
Dirac surface (cz). Any dependence = gauge violation. Tests whether
NNN gauging reduces the violation.

Moves Dirac disk to cz = 38, 40 (default L//2), 42 (±2 sites, small
displacement relative to R=5). Measures σ_tr at k=0.5, 1.0, 1.5, 2.0
for both NN-only and NN+NNN gauging. k=1.5 and k=2.0 test whether
gauge violation increases at high k (E4 diagnostic).

Results (α=0.3, R=5, L=80, 364s):

  Spread (max-min)/mean at each k:
    k     NN      NNN     reduction
   0.5   0.77%   0.64%     17%
   1.0   1.68%   1.07%     37%
   1.5   2.34%   1.41%     40%
   2.0   2.58%   2.13%     17%

  Gauge violation INCREASES with k: NN goes from 0.77% to 2.58%.
  NNN consistently better (17-40% reduction at all k).
  Both remain small (< 3%) — NN-only not catastrophically wrong.
  But at k>1.5, gauge violation is no longer negligible (~2.5%).

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/25_test_gauge_invariance.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gauge_3d import make_vortex_force
from elastic_3d import make_damping_3d, run_fdtd_3d
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           compute_sphere_f2, integrate_sigma_3d,
                           estimate_n_steps_3d)

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 5
ALPHA = 0.3

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

k_vals = [0.5, 1.0, 1.5, 2.0]
cz_vals = [38, 40, 42]

if __name__ == '__main__':
    t0 = time.time()

    print("Test D: Gauge invariance (Dirac surface position)")
    print(f"  α={ALPHA}, R={R_LOOP}, L={L}")
    print(f"  cz values: {cz_vals}")
    print(f"  k values: {k_vals}")
    print()

    # Shared setup
    gamma = make_damping_3d(L, DW, DS)
    iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
    x_start = DW + 5

    # Compute references (no gauging, cz-independent)
    print("Computing references...")
    from elastic_3d import scalar_laplacian_3d

    def force_plain(ux, uy, uz):
        return scalar_laplacian_3d(ux, uy, uz, K1, K2)

    refs = {}
    for k0 in k_vals:
        ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
        ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma,
                          DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        refs[k0] = (ref, ux0, vx0, ns)
    print(f"  Done ({time.time()-t0:.0f}s)")
    print()

    # Scan cz × k × gauging
    results = {}  # (cz, k, gauging) -> sigma_tr
    for cz in cz_vals:
        for gauge_nnn in [False, True]:
            label = "NNN" if gauge_nnn else "NN"
            f_def = make_vortex_force(ALPHA, R_LOOP, L, K1, K2,
                                      cz=cz, gauge_nnn=gauge_nnn)
            for k0 in k_vals:
                ref, ux0, vx0, ns = refs[k0]
                t1 = time.time()
                d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma,
                                DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix,
                                rec_n=ns)
                f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                       ref['ux'], ref['uy'], ref['uz'], r_m)
                _, st = integrate_sigma_3d(f2, thetas, phis)
                results[(cz, k0, label)] = st
                dt = time.time() - t1
                print(f"  cz={cz}, k={k0}, {label}: σ_tr={st:.3f} ({dt:.0f}s)")

    # Results tables
    print()
    for k0 in k_vals:
        print("=" * 50)
        print(f"σ_tr at k={k0} (α={ALPHA})")
        print("=" * 50)
        print(f"  {'cz':>4s}  {'σ_NN':>8s}  {'σ_NNN':>8s}")
        print(f"  {'-'*25}")
        nn_vals = []
        nnn_vals = []
        for cz in cz_vals:
            snn = results[(cz, k0, "NN")]
            snnn = results[(cz, k0, "NNN")]
            nn_vals.append(snn)
            nnn_vals.append(snnn)
            print(f"  {cz:4d}  {snn:8.3f}  {snnn:8.3f}")

        nn_spread = (max(nn_vals) - min(nn_vals)) / np.mean(nn_vals) * 100
        nnn_spread = (max(nnn_vals) - min(nnn_vals)) / np.mean(nnn_vals) * 100
        nn_abs = (max(nn_vals) - min(nn_vals)) / 2
        nnn_abs = (max(nnn_vals) - min(nnn_vals)) / 2
        improvement = (nn_spread - nnn_spread) / nn_spread * 100 if nn_spread > 0 else 0
        print(f"\n  Spread: NN {nn_spread:.2f}% → NNN {nnn_spread:.2f}% "
              f"({improvement:+.0f}% reduction)")
        print(f"  Absolute: NN ±{nn_abs:.3f}, NNN ±{nnn_abs:.3f}")
        print()

    t_total = time.time() - t0
    print(f"Total time: {t_total:.0f}s ({t_total/60:.1f} min)")
