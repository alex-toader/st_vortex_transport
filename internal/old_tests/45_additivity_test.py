"""
Route 45: Additivity test — is scattering incoherent?

If scattering is incoherent (each bond independent):
  σ(A ∪ B) = σ(A) + σ(B)   for disjoint bond sets A, B.

Test: split disk into 4 quarters (Q1-Q4), measure σ individually and
in combinations. Compare sums with direct measurements.

Quarters (angular ranges on [0, 2π)):
  Q1 (0, π/2):       front-top, faces incoming wave (+x)
  Q2 (π/2, π):       back-top
  Q3 (π, 3π/2):      back-bottom
  Q4 (3π/2, 2π):     front-bottom

Runs (8 FDTD per k):
  Individual: Q1, Q2, Q3, Q4
  Continuous halves: half-front (0,π), half-back (π,2π)
  Discontinuous pairs: Q1∪Q3 (opposite), Q2∪Q4 (opposite)

Known: full ring (81 bonds) from file 42.

Additivity tests:
  T1: σ(Q1) + σ(Q2)  vs  σ(half-front)    — adjacent quarters
  T2: σ(Q3) + σ(Q4)  vs  σ(half-back)     — adjacent quarters
  T3: σ(Q1) + σ(Q3)  vs  σ(Q1∪Q3 sim.)   — opposite quarters (non-adjacent)
  T4: σ(Q2) + σ(Q4)  vs  σ(Q2∪Q4 sim.)   — opposite quarters (non-adjacent)
  T5: Σ σ(Qi)        vs  σ(full ring)      — all four
  T6: σ(half-f) + σ(half-b) vs σ(full)    — halves
  T7: σ(Q1∪Q3) + σ(Q2∪Q4) vs σ(full)     — cross-pairs

If all ratios ≈ 1.0: INCOHERENT.
If adjacent fail but non-adjacent pass: LOCAL coherence.
If all fail: GLOBAL coherence.

Results:

SCATTERING IS NOT INCOHERENT. Massive interference effects (up to 41%).

Bond counts: Q1=21, Q2=20, Q3=20, Q4=20, half-f=41, half-b=40,
  Q1+Q3=41, Q2+Q4=40, full=81. Q1+Q2+Q3+Q4=81 ✓.

σ_tr values (k = 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5):
  Q1 (21):       8.20  4.43  2.67  1.78  1.35  1.12  1.02
  Q2 (20):       8.41  4.50  2.71  2.01  1.67  1.40  1.31
  Q3 (20):       9.12  5.52  3.48  2.35  1.82  1.54  1.38
  Q4 (20):       7.44  3.81  2.27  1.68  1.41  1.19  1.10
  half-front (41): 15.89  5.29  3.00  2.20  1.77  1.47  1.38
  half-back (40):  15.83  5.76  3.43  2.42  1.91  1.61  1.45
  Q1+Q3 sim (41): 16.84  7.29  5.53  3.86  3.00  2.58  2.31
  Q2+Q4 sim (40): 19.13  7.71  5.17  4.07  3.19  2.67  2.45

Additivity excess % (ratio = σ_combined / Σσ_parts - 1):

                    k=0.3  k=0.5  k=0.7  k=0.9  k=1.1  k=1.3  k=1.5  mean
  T1 Q1+Q2→hf       -4%   -41%   -44%   -42%   -41%   -42%   -41%  -36%
  T2 Q3+Q4→hb       -4%   -38%   -40%   -40%   -41%   -41%   -42%  -35%
  T3 Q1+Q3→sim      -3%   -27%   -10%    -7%    -5%    -3%    -4%   -8%
  T4 Q2+Q4→sim     +21%    -7%    +4%   +11%    +4%    +3%    +2%   +5%
  T5 ΣQi→full      +23%   -22%   -31%   -35%   -40%   -40%   -42%  -27%
  T6 hf+hb→full    +28%   +28%   +20%    +9%    +3%    +2%    -1%  +13%
  T7 pairs→full    +13%    -6%   -28%   -36%   -39%   -40%   -41%  -25%

Key findings:
  1. Adjacent quarters (T1,T2): DESTRUCTIVE interference, -36% mean.
     Scattered waves from Q1 and Q2 partially cancel at k ≥ 0.5.
  2. Opposite quarters (T3,T4): approximately ADDITIVE, mean -1.6%.
     Bonds separated by ~2R do not interfere significantly.
  3. Halves → full (T6): CONSTRUCTIVE at low k (+28% at k=0.3),
     decaying to additive at k ≥ 1.1.
  4. Four quarters → full (T5): NET DESTRUCTIVE (-27%).
     Σσ(Qi) = 33.18 but σ(full) = 40.74 at k=0.3 (+23%);
     Σσ(Qi) = 4.82 but σ(full) = 2.81 at k=1.5 (-42%).
  5. Coherence ANISOTROPIC: T3 mean -8% vs T4 mean +5%.
  6. Sign of coherence FLIPS with k: constructive at k=0.3,
     destructive at k ≥ 0.5 for most combinations.

Conclusion: the flat integrand sin²(k)·σ ≈ const for the full ring
is NOT due to incoherent scattering. It arises from the specific
coherence pattern of the complete ring geometry, where destructive
and constructive interference terms produce a net k-dependence that
happens to match the incoherent Born shape Z_avg(k)/sin²(k).
The "incoherent" label from files 42-43 is incorrect.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/45_additivity_test.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../src/1_foam'))
from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           compute_sphere_f2, integrate_sigma_3d,
                           estimate_n_steps_3d)
from gauge_3d import precompute_disk_bonds

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 5
ALPHA = 0.30

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# Full ring from file 42 (same parameters)
sigma_full = np.array([40.7432, 14.1580, 7.6935, 5.0477, 3.7764, 3.1376, 2.8065])
N_FULL = 81


def make_arc_force(alpha, R_loop, L, angle_ranges, K1=1.0, K2=0.5):
    """Peierls vortex on disk bonds within any of the given angle ranges.

    angle_ranges: list of (angle_min, angle_max) tuples.
    Each selects bonds with angle in [angle_min, angle_max).
    """
    iy_disk, ix_disk = precompute_disk_bonds(L, R_loop)
    cx, cy = L // 2, L // 2
    cz = L // 2

    angles = np.arctan2((iy_disk - cy).astype(float),
                        (ix_disk - cx).astype(float))
    angles = angles % (2 * np.pi)

    mask = np.zeros(len(angles), dtype=bool)
    for a_min, a_max in angle_ranges:
        if a_min < a_max:
            mask |= (angles >= a_min) & (angles < a_max)
        else:
            mask |= (angles >= a_min) | (angles < a_max)

    iy_arc = iy_disk[mask]
    ix_arc = ix_disk[mask]
    n_bonds = len(iy_arc)

    phi = 2.0 * np.pi * (float(alpha) % 1.0)
    cm1 = np.cos(phi) - 1.0
    s_phi = np.sin(phi)
    if abs(s_phi) < 1e-12:
        s_phi = 0.0
    if abs(cm1) < 1e-12:
        cm1 = 0.0

    iz_lo = cz - 1
    iz_hi = cz

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if cm1 == 0.0 and s_phi == 0.0:
            return fx, fy, fz

        ux_hi = ux[iz_hi, iy_arc, ix_arc]
        uy_hi = uy[iz_hi, iy_arc, ix_arc]
        ux_lo = ux[iz_lo, iy_arc, ix_arc]
        uy_lo = uy[iz_lo, iy_arc, ix_arc]

        fx[iz_lo, iy_arc, ix_arc] += K1 * (cm1 * ux_hi - s_phi * uy_hi)
        fy[iz_lo, iy_arc, ix_arc] += K1 * (s_phi * ux_hi + cm1 * uy_hi)

        fx[iz_hi, iy_arc, ix_arc] += K1 * (cm1 * ux_lo + s_phi * uy_lo)
        fy[iz_hi, iy_arc, ix_arc] += K1 * (-s_phi * ux_lo + cm1 * uy_lo)

        return fx, fy, fz

    force_fn.n_bonds = n_bonds
    return force_fn


# Define all 8 runs
PI = np.pi
RUNS = [
    ("Q1",         [(0, PI/2)]),
    ("Q2",         [(PI/2, PI)]),
    ("Q3",         [(PI, 3*PI/2)]),
    ("Q4",         [(3*PI/2, 2*PI)]),
    ("half-front", [(0, PI)]),
    ("half-back",  [(PI, 2*PI)]),
    ("Q1+Q3",      [(0, PI/2), (PI, 3*PI/2)]),
    ("Q2+Q4",      [(PI/2, PI), (3*PI/2, 2*PI)]),
]


if __name__ == '__main__':
    t0 = time.time()

    print("Route 45: Additivity test — incoherent or coherent?")
    print(f"  R={R_LOOP}, α={ALPHA}, L={L}")
    print(f"  k = {list(k_vals)}")
    print()

    # Bond counts and verification
    print("  Bond counts:")
    n_total_check = 0
    for label, ranges in RUNS:
        f = make_arc_force(ALPHA, R_LOOP, L, ranges, K1, K2)
        print(f"    {label:>12s}: {f.n_bonds:3d} bonds")
        if label in ("Q1", "Q2", "Q3", "Q4"):
            n_total_check += f.n_bonds
    print(f"    {'full (f42)':>12s}: {N_FULL:3d} bonds")
    print(f"    Q1+Q2+Q3+Q4 count: {n_total_check}"
          f" {'✓' if n_total_check == N_FULL else '✗ MISMATCH'}")

    # Completeness: verify angular sweep covers all bonds
    f_all = make_arc_force(ALPHA, R_LOOP, L, [(0, 2*PI)], K1, K2)
    print(f"    angular sweep (0,2π): {f_all.n_bonds} bonds"
          f" {'✓' if f_all.n_bonds == N_FULL else '✗ GAPS'}")
    print()

    gamma = make_damping_3d(L, DW, DS)
    iz_s, iy_s, ix_s = make_sphere_points(L, r_m, thetas, phis)

    def force_plain(ux, uy, uz):
        return scalar_laplacian_3d(ux, uy, uz, K1, K2)

    print("Computing references...")
    t1 = time.time()
    refs = {}
    for k0 in k_vals:
        ux0, vx0 = make_wave_packet_3d(L, k0, DW + 5, sx, K1, K2)
        ns = estimate_n_steps_3d(k0, L, DW + 5, sx, r_m, DT, 50, K1, K2)
        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma, DT, ns,
                          rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
        refs[k0] = (ref, ux0, vx0, ns)
    print(f"  Done ({time.time()-t1:.0f}s)")

    # Run all 8 configurations
    results = {}

    for label, ranges in RUNS:
        f_test = make_arc_force(ALPHA, R_LOOP, L, ranges, K1, K2)
        n = f_test.n_bonds
        print(f"\n{label:>12s}  ({n} bonds)")

        t1 = time.time()
        sigma_tr = np.zeros(len(k_vals))
        for j, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_test, ux0.copy(), vx0.copy(), gamma, DT, ns,
                            rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
            f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                   ref['ux'], ref['uy'], ref['uz'], r_m)
            _, st = integrate_sigma_3d(f2, thetas, phis)
            sigma_tr[j] = max(st, 0.0)
        dt = time.time() - t1

        results[label] = {'n_bonds': n, 'sigma_tr': sigma_tr.copy()}
        print(f"  σ_tr: " + "  ".join(f"{s:.4f}" for s in sigma_tr))
        print(f"  ({dt:.0f}s)")

    # ═══════════════════════════════════════════════════════════════════
    # ADDITIVITY TESTS
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("ADDITIVITY TESTS: ratio = σ(combined) / [σ(A) + σ(B)]")
    print("  ratio = 1.00 → incoherent.  ratio > 1 → constructive coherence.")
    print()
    print("  T1/T2: adjacent quarters — tests LOCAL coherence (nearby bonds)")
    print("  T3/T4: opposite quarters — tests GLOBAL coherence (distance ~2R)")
    print("         ratio ≠ 1 at T3/T4 → coherence length > R_loop")
    print("  T5-T7: full ring decomposition — overall consistency check")
    print()
    print("  NOTE: transport weighting (1-cosθ_s) affects σ per bond but NOT")
    print("  additivity. σ(A∪B) = σ(A)+σ(B) holds iff incoherent, regardless")
    print("  of angular weighting. Gauge invariance error ~2.6% (file 28).")
    print("=" * 80)

    tests = [
        ("T1: Q1+Q2 vs half-front",
         "half-front", [("Q1", "Q2")]),
        ("T2: Q3+Q4 vs half-back",
         "half-back", [("Q3", "Q4")]),
        ("T3: Q1+Q3 (sum) vs Q1+Q3 (sim.)",
         "Q1+Q3", [("Q1", "Q3")]),
        ("T4: Q2+Q4 (sum) vs Q2+Q4 (sim.)",
         "Q2+Q4", [("Q2", "Q4")]),
        ("T5: Σ Qi vs full",
         None, [("Q1", "Q2", "Q3", "Q4")]),
        ("T6: half-f + half-b vs full",
         None, [("half-front", "half-back")]),
        ("T7: (Q1+Q3) + (Q2+Q4) vs full",
         None, [("Q1+Q3", "Q2+Q4")]),
    ]

    for test_label, combined_key, sum_parts in tests:
        print(f"\n{test_label}")

        # σ_combined: either a measured run or sigma_full
        if combined_key is not None:
            sigma_comb = results[combined_key]['sigma_tr']
            n_comb = results[combined_key]['n_bonds']
        else:
            sigma_comb = sigma_full
            n_comb = N_FULL

        # σ_sum: sum of parts
        parts = sum_parts[0]  # tuple of labels
        sigma_sum = np.zeros(len(k_vals))
        n_sum = 0
        for p in parts:
            sigma_sum += results[p]['sigma_tr']
            n_sum += results[p]['n_bonds']

        print(f"  N: combined={n_comb}, sum={n_sum}"
              f" {'✓' if n_comb == n_sum else '✗ MISMATCH'}")

        print(f"  {'k':>5s}  {'σ_comb':>8s}  {'Σ σ_parts':>10s}"
              f"  {'ratio':>7s}  {'excess':>8s}")
        ratios = []
        for j, k0 in enumerate(k_vals):
            r = sigma_comb[j] / sigma_sum[j] if sigma_sum[j] > 0 else 0
            exc = (r - 1) * 100
            ratios.append(r)
            print(f"  {k0:5.1f}  {sigma_comb[j]:8.4f}  {sigma_sum[j]:10.4f}"
                  f"  {r:7.4f}  {exc:+7.1f}%")
        mean_r = np.mean(ratios)
        # Threshold 5%: gauge invariance ~2.6% (file 28), combined error ~3-4%
        if abs(mean_r - 1) < 0.05:
            verdict = "INCOHERENT (within 5%)"
        elif mean_r > 1.05:
            verdict = "COHERENT (excess > 5%)"
        else:
            verdict = "ANTI-COHERENT (deficit > 5%)"
        print(f"  Mean ratio = {mean_r:.4f}  ({verdict})")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("SUMMARY: excess % at each k")
    print("=" * 80)

    test_configs = [
        ("T1 Q1+Q2→hf",  "half-front", ["Q1", "Q2"]),
        ("T2 Q3+Q4→hb",  "half-back",  ["Q3", "Q4"]),
        ("T3 Q1+Q3→sim", "Q1+Q3",      ["Q1", "Q3"]),
        ("T4 Q2+Q4→sim", "Q2+Q4",      ["Q2", "Q4"]),
        ("T5 ΣQi→full",  None,         ["Q1", "Q2", "Q3", "Q4"]),
        ("T6 hf+hb→full", None,        ["half-front", "half-back"]),
        ("T7 pairs→full", None,        ["Q1+Q3", "Q2+Q4"]),
    ]

    print(f"  {'test':>16s}", end="")
    for k0 in k_vals:
        print(f"  {f'k={k0:.1f}':>7s}", end="")
    print(f"  {'mean':>7s}")

    for tlabel, ckey, parts in test_configs:
        sigma_c = results[ckey]['sigma_tr'] if ckey else sigma_full
        sigma_s = sum(results[p]['sigma_tr'] for p in parts)
        excess = (sigma_c / sigma_s - 1) * 100
        print(f"  {tlabel:>16s}", end="")
        for e in excess:
            print(f"  {e:+6.1f}%", end="")
        print(f"  {np.mean(excess):+6.1f}%")

    # T3 vs T4 comparison: is coherence isotropic?
    sigma_c3 = results["Q1+Q3"]['sigma_tr']
    sigma_s3 = results["Q1"]['sigma_tr'] + results["Q3"]['sigma_tr']
    excess_T3 = (sigma_c3 / sigma_s3 - 1) * 100

    sigma_c4 = results["Q2+Q4"]['sigma_tr']
    sigma_s4 = results["Q2"]['sigma_tr'] + results["Q4"]['sigma_tr']
    excess_T4 = (sigma_c4 / sigma_s4 - 1) * 100

    print(f"\n  T3 vs T4 (coherence isotropy):")
    print(f"    T3 mean excess (Q1+Q3, front+back): {np.mean(excess_T3):+.1f}%")
    print(f"    T4 mean excess (Q2+Q4, lateral):    {np.mean(excess_T4):+.1f}%")
    if abs(np.mean(excess_T3) - np.mean(excess_T4)) < 3:
        print(f"    → Coherence ISOTROPIC (T3 ≈ T4)")
    else:
        print(f"    → Coherence ANISOTROPIC (T3 ≠ T4, depends on wave direction)")

    # T1/T2 vs T3/T4: local vs global
    sigma_c1 = results["half-front"]['sigma_tr']
    sigma_s1 = results["Q1"]['sigma_tr'] + results["Q2"]['sigma_tr']
    excess_T1 = (sigma_c1 / sigma_s1 - 1) * 100

    sigma_c2 = results["half-back"]['sigma_tr']
    sigma_s2 = results["Q3"]['sigma_tr'] + results["Q4"]['sigma_tr']
    excess_T2 = (sigma_c2 / sigma_s2 - 1) * 100

    mean_adj = (np.mean(excess_T1) + np.mean(excess_T2)) / 2
    mean_opp = (np.mean(excess_T3) + np.mean(excess_T4)) / 2
    print(f"\n  Adjacent (T1,T2) mean excess: {mean_adj:+.1f}%")
    print(f"  Opposite (T3,T4) mean excess: {mean_opp:+.1f}%")
    if abs(mean_adj) < 5 and abs(mean_opp) < 5:
        print(f"  → INCOHERENT at all scales")
    elif abs(mean_adj) > 5 and abs(mean_opp) < 5:
        print(f"  → LOCAL coherence only (adjacent bonds interfere, distant don't)")
    elif abs(mean_adj) > 5 and abs(mean_opp) > 5:
        print(f"  → GLOBAL coherence (all bonds interfere regardless of distance)")

    print()
    print("Interpretation key:")
    print("  All ≈ 0%: INCOHERENT (each bond independent)")
    print("  Adjacent > 0%, opposite ≈ 0%: LOCAL coherence (nearby bonds interfere)")
    print("  All > 0%: GLOBAL coherence (all bonds interfere)")
    print("  k-dependent: PARTIAL coherence (coherent at low k, incoherent at high k)")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
