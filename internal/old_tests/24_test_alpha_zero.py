"""
Test C+A: α=0 artifact check + n_steps verification.

Test C: σ_tr must be exactly zero at α=0 for ALL k values.
If σ_tr ≠ 0 at high k with α=0, the pipeline has a numeric artifact.
If σ_tr = 0 (exact), high-k scattering is real physics (defect-dependent).

Test A: n_steps must scale with 1/vg to ensure recording is complete.
At high k, vg = c·cos(k/2) → 0, so n_steps must increase.

Results (R=5, L=80, α=0, 5 k-pts, fork(4), 88s):

  Test C — σ_tr(α=0):
    k     σ_tr         n_steps   vg
   0.70   0.000000       183    1.627
   1.50   0.000000       208    1.267
   1.90   0.000000       237    1.008
   2.10   0.000000       260    0.862
   2.70   0.000000       470    0.379

  Verdict: σ_tr = 0.000000 at ALL k. No artifact. High-k rise is real.

  Test A — n_steps scaling:
  Correlation n_steps vs 1/vg: 1.0000. Perfect scaling.

  Recording completeness (E_tail/E_peak):
    k=0.70: 0.0000 OK
    k=1.50: 0.0000 OK
    k=1.90: 0.0003 OK
    k=2.10: 0.0019 OK
    k=2.70: 0.0480 WARNING — recording may be slightly truncated
  All k ≤ 2.1 fully complete. k=2.70 marginal (4.8% residual energy).

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/24_test_alpha_zero.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from parallel_fdtd import compute_references, compute_scattering
from scattering_3d import group_velocity_3d, estimate_n_steps_3d

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 5
N_WORKERS = 4

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

# k values spanning low to high — includes the "suspicious" region k > 1.5
k_vals = np.array([0.7, 1.5, 1.9, 2.1, 2.7])

if __name__ == '__main__':
    t0 = time.time()

    print("Test C+A: α=0 artifact check + n_steps verification")
    print(f"  L={L}, R={R_LOOP}, k-pts: {len(k_vals)}")
    print(f"  k values: {k_vals}")
    print()

    # Test A: report n_steps and vg for each k
    print("── Test A: n_steps scaling ──")
    x_start = DW + 5
    ns_arr = np.array([estimate_n_steps_3d(k, L, x_start, sx, r_m, DT, 50, K1, K2)
                       for k in k_vals])
    vg_arr = np.array([group_velocity_3d(k, K1, K2) for k in k_vals])
    print(f"  {'k':>5s}  {'vg':>6s}  {'n_steps':>7s}  {'1/vg':>6s}")
    print(f"  {'-'*30}")
    for j, k0 in enumerate(k_vals):
        print(f"  {k0:5.2f}  {vg_arr[j]:6.3f}  {ns_arr[j]:7d}  {1/vg_arr[j]:6.3f}")
    corr = np.corrcoef(1 / vg_arr, ns_arr)[0, 1]
    print(f"\n  Correlation n_steps vs 1/vg: {corr:.4f} (expect ≈ 1.0)")
    print()

    # Test C: compute σ_tr at α=0
    print("── Test C: σ_tr at α=0 ──")
    print(f"Computing references...")
    t1 = time.time()
    refs = compute_references(k_vals, L, DW, DS, DT, sx, r_m,
                              thetas, phis, K1, K2, n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")

    print(f"Computing scattering at α=0...")
    t1 = time.time()
    sigma_tr = compute_scattering(k_vals, refs, 0.0, R_LOOP, L, DW, DS, DT,
                                  r_m, thetas, phis, K1, K2,
                                  n_workers=N_WORKERS)
    print(f"  Done ({time.time()-t1:.0f}s)")
    print()

    # Results table
    print("=" * 60)
    print("σ_tr at α=0 (must be exactly zero)")
    print("=" * 60)
    print(f"  {'k':>5s}  {'σ_tr':>12s}  {'n_steps':>7s}  {'vg':>6s}")
    print(f"  {'-'*35}")
    all_zero = True
    for i, k0 in enumerate(k_vals):
        vg = group_velocity_3d(k0, K1, K2)
        ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
        st = sigma_tr[i]
        print(f"  {k0:5.2f}  {st:12.6f}  {ns:7d}  {vg:6.3f}")
        if abs(st) > 1e-10:
            all_zero = False

    print()
    if all_zero:
        print("** PASS: σ_tr = 0 at all k. No numeric artifact. **")
        print("   High-k σ_tr rise (at α≠0) is real physics, not pipeline bug.")
    else:
        print("** FAIL: σ_tr ≠ 0 at α=0! Pipeline has artifact! **")

    # Recording completeness diagnostic:
    # Check that sphere energy peaked and decayed before end of recording.
    # If energy at last timestep > 1% of peak → recording may be truncated.
    print()
    print("── Recording completeness check ──")
    for k0 in k_vals:
        ref, ux0, vx0, ns = refs[k0]
        # ref['ux'] has shape (ns, N_pts) — energy on sphere vs time
        energy_t = np.mean(ref['ux']**2 + ref['uy']**2 + ref['uz']**2, axis=1)
        peak = energy_t.max()
        tail = energy_t[-1]
        ratio = tail / peak if peak > 0 else 0
        status = "OK" if ratio < 0.01 else "WARNING: truncated?"
        print(f"  k={k0:.2f}: E_tail/E_peak = {ratio:.4f}  ({status})")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
