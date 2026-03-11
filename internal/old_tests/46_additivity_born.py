"""
Route 46: Additivity test at α=0.10 (weak coupling).

File 45 showed massive interference (±40%) at α=0.30. If this is a
non-perturbative (T-matrix) effect, it should vanish at weak coupling
where Born approximation holds.

α=0.10: coupling (cos(2πα)-1)² = 0.037 (vs 1.71 at α=0.30, ratio 46×).
Expected σ_full ≈ 2.4 (from C(α) ~ α^{2.56}), σ_quarter ≈ 0.6.
S/N: relative gauge error 2.6% is independent of α. FDTD is deterministic
so absolute noise floor is negligible (< 1e-10).

NOTE: Born validity at α=0.10 not fully guaranteed — file 39 shows
non-Born enhancement 7-36× even at weak coupling. But coupling is 46×
weaker than α=0.30, so if additivity holds here and fails at α=0.30,
the transition is clearly coupling-driven.

Same 8 runs as file 45: Q1-Q4, half-front, half-back, Q1∪Q3, Q2∪Q4.
Same T1-T7 additivity tests.

If all ratios ≈ 1.0: Born is additive → interference at α=0.30 is T-matrix.
If ratios ≠ 1.0 even here: interference is inherent to coherent geometry.

Results (831s):

  σ_tr at α=0.10, R=5, L=80:
    full ring (81):  2.4706  0.6725  0.3715  0.2721  0.2345  0.2279  0.2311
    Q1        (21):  0.3326  0.1899  0.1253  0.0962  0.0859  0.0845  0.0890
    Q2        (20):  0.3390  0.1945  0.1301  0.1102  0.1075  0.1088  0.1187
    Q3        (20):  0.3669  0.2377  0.1684  0.1338  0.1223  0.1228  0.1301
    Q4        (20):  0.2990  0.1614  0.1038  0.0859  0.0839  0.0847  0.0917
    half-front(41):  0.7751  0.2425  0.1441  0.1157  0.1070  0.1047  0.1110
    half-back (40):  0.7560  0.2523  0.1581  0.1243  0.1133  0.1122  0.1170
    Q1+Q3     (41):  0.7486  0.3189  0.2618  0.2076  0.1869  0.1904  0.2016
    Q2+Q4     (40):  0.8320  0.3231  0.2333  0.2054  0.1928  0.1938  0.2085
    k values:        0.3     0.5     0.7     0.9     1.1     1.3     1.5

  sin²(k)·σ_tr CV = 15.3%  (vs 7.5% at α=0.30)

  Excess % (ratio = σ_combined / Σσ_parts - 1):
                test   k=0.3  k=0.5  k=0.7  k=0.9  k=1.1  k=1.3  k=1.5   mean  α=0.30
       T1 Q1+Q2→hf    +15.4  -36.9  -43.6  -43.9  -44.7  -45.8  -46.6  -35.1  -36.4
       T2 Q3+Q4→hb    +13.5  -36.8  -41.9  -43.4  -45.1  -45.9  -47.3  -35.3  -35.1
      T3 Q1+Q3→sim     +7.0  -25.4  -10.9   -9.8  -10.2   -8.2   -8.0   -9.3   -8.4
      T4 Q2+Q4→sim    +30.4   -9.2   -0.3   +4.7   +0.8   +0.2   -0.9   +3.7   +5.2
       T5 ΣQi→full    +84.7  -14.2  -29.6  -36.2  -41.3  -43.1  -46.2  -18.0  -26.8
     T6 hf+hb→full    +61.4  +35.9  +22.9  +13.4   +6.5   +5.1   +1.3  +20.9  +12.6
     T7 pairs→full    +56.3   +4.8  -25.0  -34.1  -38.2  -40.7  -43.7  -17.2  -25.3

  Pattern (α=0.10 vs α=0.30):
    Adjacent (T1,T2) mean: -35.2%  (α=0.30: -35.8%)  ← IDENTICAL
    Opposite (T3,T4) mean:  -2.8%  (α=0.30:  -1.6%)  ← IDENTICAL
    T6 halves→full mean:   +20.9%  (α=0.30: +12.6%)  ← LARGER at weak coupling

  CONCLUSION: STRONGLY NON-ADDITIVE (mean |excess| = 27.8%).
  Interference is GEOMETRIC, not coupling-driven.
  T-matrix hypothesis REFUTED: same pattern at 47× weaker coupling.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/46_additivity_born.py
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
from gauge_3d import precompute_disk_bonds, make_vortex_force

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 5
ALPHA = 0.10  # weak coupling

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

PI = np.pi


def make_arc_force(alpha, R_loop, L, angle_ranges, K1=1.0, K2=0.5):
    """Peierls vortex on disk bonds within any of the given angle ranges."""
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

    coupling = (np.cos(2 * np.pi * ALPHA) - 1)**2
    coupling_030 = (np.cos(2 * np.pi * 0.30) - 1)**2
    print(f"Route 46: Additivity at α={ALPHA} (weak coupling)")
    print(f"  Coupling (cos(2πα)-1)² = {coupling:.4f}"
          f"  (vs {coupling_030:.4f} at α=0.30, ratio {coupling_030/coupling:.0f}×)")
    print(f"  R={R_LOOP}, L={L}")
    print(f"  k = {list(k_vals)}")
    print()

    # Bond counts
    print("  Bond counts:")
    n_total = 0
    for label, ranges in RUNS:
        f = make_arc_force(ALPHA, R_LOOP, L, ranges, K1, K2)
        print(f"    {label:>12s}: {f.n_bonds:3d} bonds")
        if label in ("Q1", "Q2", "Q3", "Q4"):
            n_total += f.n_bonds
    print(f"    Q1+Q2+Q3+Q4 = {n_total} ✓")
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

    # Full ring (run fresh — different α than file 42)
    print("\nfull ring  (81 bonds)")
    f_full = make_vortex_force(ALPHA, R_LOOP, L, K1, K2)
    t1 = time.time()
    sigma_full = np.zeros(len(k_vals))
    for j, k0 in enumerate(k_vals):
        ref, ux0, vx0, ns = refs[k0]
        d = run_fdtd_3d(f_full, ux0.copy(), vx0.copy(), gamma, DT, ns,
                        rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        _, st = integrate_sigma_3d(f2, thetas, phis)
        sigma_full[j] = max(st, 0.0)
    print(f"  σ_tr: " + "  ".join(f"{s:.4f}" for s in sigma_full))

    integrand_full = np.sin(k_vals)**2 * sigma_full
    cv_full = np.std(integrand_full) / np.mean(integrand_full) * 100
    print(f"  sin²(k)·σ_tr CV = {cv_full:.1f}%  (vs 7.5% at α=0.30)")
    print(f"  ({time.time()-t1:.0f}s)")

    # Run 8 configurations
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
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print(f"ADDITIVITY SUMMARY at α={ALPHA}")
    print(f"  α=0.30 values from file 45 header (not re-run).")
    print("=" * 80)

    test_configs = [
        ("T1 Q1+Q2→hf",   "half-front", ["Q1", "Q2"]),
        ("T2 Q3+Q4→hb",   "half-back",  ["Q3", "Q4"]),
        ("T3 Q1+Q3→sim",  "Q1+Q3",      ["Q1", "Q3"]),
        ("T4 Q2+Q4→sim",  "Q2+Q4",      ["Q2", "Q4"]),
        ("T5 ΣQi→full",   None,         ["Q1", "Q2", "Q3", "Q4"]),
        ("T6 hf+hb→full", None,         ["half-front", "half-back"]),
        ("T7 pairs→full", None,         ["Q1+Q3", "Q2+Q4"]),
    ]

    # File 45 results (from header, not re-run)
    excess_f45 = {
        "T1 Q1+Q2→hf":   [-4.3, -40.8, -44.2, -41.9, -41.3, -41.6, -40.9],
        "T2 Q3+Q4→hb":   [-4.4, -38.2, -40.3, -39.9, -40.7, -41.0, -41.5],
        "T3 Q1+Q3→sim":  [-2.8, -26.7, -10.1, -6.6, -5.4, -3.4, -4.1],
        "T4 Q2+Q4→sim":  [20.7, -7.3, 3.8, 10.6, 3.6, 3.0, 1.7],
        "T5 ΣQi→full":   [22.8, -22.5, -30.9, -35.4, -39.5, -40.3, -41.8],
        "T6 hf+hb→full": [28.4, 28.1, 19.5, 9.1, 2.5, 1.7, -0.9],
        "T7 pairs→full": [13.3, -5.6, -28.1, -36.4, -38.9, -40.2, -41.1],
    }

    print(f"\n  {'test':>16s}", end="")
    for k0 in k_vals:
        print(f"  {f'k={k0:.1f}':>7s}", end="")
    print(f"  {'mean':>7s}  {'α=0.30':>7s}")

    all_excess_by_test = {}
    for tlabel, ckey, parts in test_configs:
        sigma_c = results[ckey]['sigma_tr'] if ckey else sigma_full
        sigma_s = sum(results[p]['sigma_tr'] for p in parts)
        excess = (sigma_c / sigma_s - 1) * 100
        mean_exc = np.mean(excess)
        mean_f45 = np.mean(excess_f45[tlabel])
        all_excess_by_test[tlabel] = excess
        print(f"  {tlabel:>16s}", end="")
        for e in excess:
            print(f"  {e:+6.1f}%", end="")
        print(f"  {mean_exc:+6.1f}%  {mean_f45:+6.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # PATTERN ANALYSIS (more robust than binary threshold)
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)

    excess_T1 = all_excess_by_test["T1 Q1+Q2→hf"]
    excess_T2 = all_excess_by_test["T2 Q3+Q4→hb"]
    excess_T3 = all_excess_by_test["T3 Q1+Q3→sim"]
    excess_T4 = all_excess_by_test["T4 Q2+Q4→sim"]

    mean_adj = (np.mean(excess_T1) + np.mean(excess_T2)) / 2
    mean_opp = (np.mean(excess_T3) + np.mean(excess_T4)) / 2

    print(f"\n  Adjacent (T1,T2) mean excess: {mean_adj:+.1f}%"
          f"  (α=0.30: {(np.mean(excess_f45['T1 Q1+Q2→hf'])+np.mean(excess_f45['T2 Q3+Q4→hb']))/2:+.1f}%)")
    print(f"  Opposite (T3,T4) mean excess: {mean_opp:+.1f}%"
          f"  (α=0.30: {(np.mean(excess_f45['T3 Q1+Q3→sim'])+np.mean(excess_f45['T4 Q2+Q4→sim']))/2:+.1f}%)")

    # T6 is the cleanest test (two large halves, symmetric)
    excess_T6 = all_excess_by_test["T6 hf+hb→full"]
    print(f"  T6 (halves→full) mean excess: {np.mean(excess_T6):+.1f}%"
          f"  (α=0.30: {np.mean(excess_f45['T6 hf+hb→full']):+.1f}%)")

    print(f"\n  sin²(k)·σ_tr CV: {cv_full:.1f}%  (α=0.30: 7.5%)")

    # CV per quarter
    print(f"\n  CV per configuration:")
    for label in ["Q1", "Q2", "Q3", "Q4", "half-front", "half-back",
                   "Q1+Q3", "Q2+Q4"]:
        s = results[label]['sigma_tr']
        integrand = np.sin(k_vals)**2 * s
        if np.mean(integrand) > 1e-10:
            cv = np.std(integrand) / np.mean(integrand) * 100
            print(f"    {label:>12s}: CV = {cv:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # CONCLUSION
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    all_excess = []
    for tlabel in all_excess_by_test:
        all_excess.extend(all_excess_by_test[tlabel])
    max_abs = max(abs(e) for e in all_excess)
    mean_abs = np.mean([abs(e) for e in all_excess])

    print(f"\n  Max |excess| = {max_abs:.1f}%, mean |excess| = {mean_abs:.1f}%")
    print(f"  Gauge invariance error: ~2.6% (file 28)")

    if mean_abs < 5:
        print(f"\n  ADDITIVE at α={ALPHA}: mean |excess| < 5%.")
        print(f"  Interference at α=0.30 (mean |excess| ~25%) is coupling-driven.")
        print(f"  → Consistent with T-matrix mechanism.")
    elif mean_abs < 15:
        print(f"\n  WEAKLY NON-ADDITIVE at α={ALPHA}: mean |excess| = {mean_abs:.1f}%.")
        print(f"  Some coherent interference present but much weaker than α=0.30.")
        print(f"  → Interference grows with coupling: Born → partial → T-matrix.")
    else:
        print(f"\n  STRONGLY NON-ADDITIVE at α={ALPHA}: mean |excess| = {mean_abs:.1f}%.")
        print(f"  Interference is GEOMETRIC, not just coupling-driven.")
        print(f"  → Coherent form factor dominates even in Born regime.")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
