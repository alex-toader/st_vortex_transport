"""
Route 23: Tail verification — does σ_tr at k > 1.5 match power law prediction?

POWER LAW TAIL IS INVALID. σ_tr has minimum at k≈1.7, then rises.

Step 1 (k=1.7, 1.9, 2.1) — power law comparison:

    k     σ_tr    predicted   deviation
   1.70   2.713     2.360      +15%
   1.90   2.816     2.075      +36%
   2.10   3.341     1.849      +81%

Step 2 (k=2.4, 2.7, 3.0) — extended to BZ edge:

    k     σ_tr    sin²(k)   sin²·σ_tr
   1.50   2.81    0.9975     2.796
   1.70   2.713   0.9832     2.668
   1.90   2.816   0.8957     2.522
   2.10   3.341   0.7452     2.489
   2.40   5.120   0.4563     2.336
   2.70   8.839   0.1827     1.615
   3.00  22.447   0.0199     0.447

σ_tr rises dramatically at k > 1.7 (minimum). Opposite of power law.
Possible causes: BZ artifact (λ≈3a at k=2.1) or physical
(wave resolves Dirac disk micro-structure at short λ).

Integrand sin²(k)·σ_tr decreases slowly at k < 2.4, then drops fast:
  sin²(k) suppression at k > 2 kills the integrand regardless of σ_tr.

κ integration (α=0.3, R=5):
  κ_data (k=0.3-1.5,  7 pts) = 0.944     (original)
  κ_data (k=0.3-2.1, 10 pts) = 1.341     (Step 1)
  κ_data (k=0.3-3.0, 13 pts) = 1.753     (Step 2)
  Remaining tail (k=3.0→π) negligible: sin²(3.0) = 0.02.
  Old κ_total = 1.382 (power law tail) was WRONG — underestimated.
  True κ(α=0.3) ≈ 1.75 (measured across nearly full BZ).

α=0.3, R=5, L=80. Uses parallel_fdtd for speed.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/21_tail_verification.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from parallel_fdtd import compute_references, compute_scattering

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 5
ALPHA = 0.3
N_POL = 2
N_WORKERS = 2

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

# Existing grid (for power law fit) + new high-k points
k_existing = np.array([0.9, 1.1, 1.3, 1.5])
k_new = np.array([1.7, 1.9, 2.1, 2.4, 2.7, 3.0])
k_all = np.concatenate([k_existing, k_new])

# Known σ_tr from file 16 (α=0.3, R=5, L=80)
sigma_known = {0.9: 5.05, 1.1: 3.78, 1.3: 3.14, 1.5: 2.81}

if __name__ == '__main__':
    t0 = time.time()

    print("Route 23: Tail verification")
    print(f"  α={ALPHA}, R={R_LOOP}, L={L}")
    print(f"  Existing k: {k_existing} (power law fit region)")
    print(f"  New k: {k_new} (verification points)")
    print()

    # Power law fit from existing data: σ_tr = A * k^β
    log_k = np.log(k_existing)
    log_s = np.log([sigma_known[k] for k in k_existing])
    beta, logA = np.polyfit(log_k, log_s, 1)
    A = np.exp(logA)
    print(f"Power law fit (k=0.9-1.5): σ_tr = {A:.3f} * k^{beta:.3f}")
    print(f"  Predictions for new k:")
    for k in k_new:
        pred = A * k**beta
        print(f"    k={k:.1f}: σ_tr_pred = {pred:.3f}")
    print()

    # Compute references and scattering for new k points only
    print(f"Computing {len(k_new)} new references + scattering...")
    refs = compute_references(k_new, L, DW, DS, DT, sx, r_m,
                              thetas, phis, K1, K2, n_workers=N_WORKERS)
    sigma_new = compute_scattering(k_new, refs, ALPHA, R_LOOP, L, DW, DS, DT,
                                   r_m, thetas, phis, K1, K2,
                                   n_workers=N_WORKERS)
    t_compute = time.time() - t0
    print(f"  Done ({t_compute:.0f}s)")
    print()

    # Compare measured vs predicted
    print("=" * 60)
    print("Tail verification: measured vs power law prediction")
    print("=" * 60)
    print(f"{'k':>5s}  {'σ_meas':>8s}  {'σ_pred':>8s}  {'ratio':>6s}  {'diff%':>6s}")
    print("-" * 40)

    # First show existing points (should match exactly)
    for k in k_existing:
        pred = A * k**beta
        meas = sigma_known[k]
        ratio = meas / pred
        diff = (meas - pred) / pred * 100
        print(f"{k:5.2f}  {meas:8.3f}  {pred:8.3f}  {ratio:6.3f}  {diff:+5.1f}%  (fit)")

    # Then new points
    max_diff = 0
    for i, k in enumerate(k_new):
        pred = A * k**beta
        meas = sigma_new[i]
        ratio = meas / pred
        diff = (meas - pred) / pred * 100
        max_diff = max(max_diff, abs(diff))
        print(f"{k:5.2f}  {meas:8.3f}  {pred:8.3f}  {ratio:6.3f}  {diff:+5.1f}%  ** NEW **")

    # Updated power law fit including new points
    print()
    k_all_data = np.concatenate([k_existing, k_new])
    s_all_data = np.concatenate([[sigma_known[k] for k in k_existing], sigma_new])
    beta2, logA2 = np.polyfit(np.log(k_all_data), np.log(s_all_data), 1)
    A2 = np.exp(logA2)
    print(f"Updated fit (k=0.9-2.1): σ_tr = {A2:.3f} * k^{beta2:.3f}")
    print(f"  Original: β = {beta:.3f}")
    print(f"  Updated:  β = {beta2:.3f}")
    print(f"  Δβ = {beta2 - beta:+.3f}")

    # Recompute tail with updated fit
    prefactor = N_POL * R_LOOP / (4 * np.pi**2)

    # Original tail (from k=1.5 to π)
    k_tail = np.linspace(1.5, np.pi, 200)[1:]
    tail_orig = prefactor * np.trapz(np.sin(k_tail)**2 * A * k_tail**beta, k_tail)

    # Updated tail (from k=2.1 to π, less extrapolation)
    k_tail2 = np.linspace(2.1, np.pi, 200)[1:]
    tail_new = prefactor * np.trapz(np.sin(k_tail2)**2 * A2 * k_tail2**beta2, k_tail2)

    # κ_data on extended grid (k=0.3-2.1)
    # Need full grid for this — use known + new
    k_full = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1])
    s_full_known = {0.3: 40.74, 0.5: 14.16, 0.7: 7.69, 0.9: 5.05,
                    1.1: 3.78, 1.3: 3.14, 1.5: 2.81}
    s_full = []
    for k in k_full:
        if k in s_full_known:
            s_full.append(s_full_known[k])
        else:
            idx = np.argmin(np.abs(k_new - k))
            s_full.append(sigma_new[idx])
    s_full = np.array(s_full)
    integrand_full = np.sin(k_full)**2 * s_full
    kd_extended = prefactor * np.trapz(integrand_full, k_full)

    # Original κ_data (k=0.3-1.5, 7 pts)
    k_orig = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    s_orig = np.array([s_full_known[k] for k in k_orig])
    kd_orig = prefactor * np.trapz(np.sin(k_orig)**2 * s_orig, k_orig)

    print()
    print("=" * 60)
    print("κ comparison")
    print("=" * 60)
    print(f"  Original (k=0.3-1.5, 7 pts):  κ_data = {kd_orig:.3f}")
    print(f"  Extended (k=0.3-2.1, 10 pts): κ_data = {kd_extended:.3f}")
    print(f"  Gain from 3 new k-pts: {kd_extended - kd_orig:.3f} "
          f"({(kd_extended - kd_orig)/kd_orig*100:+.1f}%)")
    print()
    print(f"  Original tail (β={beta:.3f}, k=1.5→π):  κ_tail = {tail_orig:.3f}")
    print(f"  Updated tail  (β={beta2:.3f}, k=2.1→π): κ_tail = {tail_new:.3f}")
    print()
    ktot_orig = kd_orig + tail_orig
    ktot_new = kd_extended + tail_new
    tail_pct_orig = tail_orig / ktot_orig * 100
    tail_pct_new = tail_new / ktot_new * 100
    print(f"  Original κ_total = {ktot_orig:.3f} (tail {tail_pct_orig:.0f}%)")
    print(f"  Updated  κ_total = {ktot_new:.3f} (tail {tail_pct_new:.0f}%)")
    print(f"  Difference: {(ktot_new - ktot_orig)/ktot_orig*100:+.1f}%")

    # Verdict
    print()
    if max_diff < 10:
        print(f"** TAIL VALIDATED: max deviation {max_diff:.1f}% < 10% **")
        print(f"   Power law extrapolation is reliable.")
    elif max_diff < 20:
        print(f"** TAIL MARGINAL: max deviation {max_diff:.1f}% **")
        print(f"   Power law extrapolation is approximate.")
    else:
        print(f"** TAIL FAILS: max deviation {max_diff:.1f}% > 20% **")
        print(f"   Power law extrapolation is unreliable. Use extended grid.")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
