"""
Route 48: Cross-term spectral shape — decomposing the flat integrand.

File 45 measured σ_tr for 8 sub-ring configurations at α=0.30 (Q1-Q4 quarters,
two contiguous halves, two opposite-quarter pairs). File 46 repeated at α=0.10.

Decomposition: σ_full = σ_diag + σ_cross, where
  σ_diag(k) = Σ σ(Qi, k)           [incoherent sum of 4 quarters]
  σ_cross(k) = σ_full(k) - σ_diag(k) [interference/cross-term]

Question: is sin²(k)·σ_cross flat (→ interference uniform across k)
or shaped (→ interference is what flattens/deforms the integrand)?

Sub-decomposition from intermediate groupings:
  Adjacent cross:  σ(half-front) - σ(Q1) - σ(Q2)  [Q1,Q2 share boundary]
  Distant cross:   σ(full) - σ(half-front) - σ(half-back)  [halves ~πR apart]

Data (zero FDTD — all from files 18, 45, 46):

Results (instant, pure arithmetic):

  Decomposition sin²·full = sin²·diag + sin²·cross:
    CV(sin²·full):  7.4% (α=0.30),  15.3% (α=0.10)
    CV(sin²·diag): 15.2% (α=0.30),  37.4% (α=0.10)
    CV(sin²·cross): 66.5% (α=0.30), 111.1% (α=0.10)
    Neither component is flat — their SUM is flat at α=0.30.

  Hierarchical flattening (α=0.30):
    Quarter (15.5%) → Half (4.8%) → Full (7.4%)
    Adjacent destructive interference: ×0.32 (Q→half, strong flattening)
    Distant constructive interference: ×1.72 (half→full, DEGRADES)
    Half-rings FLATTER than full ring (4.8% vs 7.4%).

  Hierarchy reverses at α=0.10:
    Quarter (37.3%) → Half (24.3%) → Full (15.3%)
    Both levels flatten. Halves NOT flatter than full.

  Mean cross/full: -0.448 (α=0.30) vs -0.431 (α=0.10).
  Difference -0.018 → GEOMETRIC (α-independent).
  α-dependence is in the DIAGONAL (per-quarter CV: 15.5% vs 37.3%).

  1/√N test (α=0.30): quarter ratio=0.98, full ratio=0.94 (works).
  Half ratio=0.42 — adjacent interference flattens BEYOND 1/√N.
  α=0.10: cannot test (CV_1 unknown, intra-quarter coherence invalidates
  inference from CV(quarter) × √N).

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/48_cross_term_shape.py
"""

import numpy as np

k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
sin2 = np.sin(k_vals)**2


def cv(x):
    """CV with N denominator (biased), consistent with all w_21_kappa files."""
    return np.std(x) / np.abs(np.mean(x)) * 100


# ═══════════════════════════════════════════════════════════════════════════
# Data from files 18, 45, 46 (FDTD, not recomputed here)
# ═══════════════════════════════════════════════════════════════════════════

# α=0.30, R=5 (file 45 header, full ring from file 18)
data_030 = {
    'Q1':   np.array([8.20,  4.43,  2.67,  1.78,  1.35,  1.12,  1.02]),  # 21 bonds
    'Q2':   np.array([8.41,  4.50,  2.71,  2.01,  1.67,  1.40,  1.31]),  # 20 bonds
    'Q3':   np.array([9.12,  5.52,  3.48,  2.35,  1.82,  1.54,  1.38]),  # 20 bonds
    'Q4':   np.array([7.44,  3.81,  2.27,  1.68,  1.41,  1.19,  1.10]),  # 20 bonds
    'hf':   np.array([15.89, 5.29,  3.00,  2.20,  1.77,  1.47,  1.38]),  # 41 bonds (Q1∪Q2 contiguous)
    'hb':   np.array([15.83, 5.76,  3.43,  2.42,  1.91,  1.61,  1.45]),  # 40 bonds (Q3∪Q4 contiguous)
    'Q13':  np.array([16.84, 7.29,  5.53,  3.86,  3.00,  2.58,  2.31]),  # 41 bonds (Q1∪Q3 opposite)
    'Q24':  np.array([19.13, 7.71,  5.17,  4.07,  3.19,  2.67,  2.45]),  # 40 bonds (Q2∪Q4 opposite)
    'full': np.array([40.74, 14.16, 7.69,  5.05,  3.78,  3.14,  2.81]),  # 81 bonds (file 18)
}

# α=0.10, R=5 (file 46 header)
data_010 = {
    'Q1':   np.array([0.3326, 0.1899, 0.1253, 0.0962, 0.0859, 0.0845, 0.0890]),
    'Q2':   np.array([0.3390, 0.1945, 0.1301, 0.1102, 0.1075, 0.1088, 0.1187]),
    'Q3':   np.array([0.3669, 0.2377, 0.1684, 0.1338, 0.1223, 0.1228, 0.1301]),
    'Q4':   np.array([0.2990, 0.1614, 0.1038, 0.0859, 0.0839, 0.0847, 0.0917]),
    'hf':   np.array([0.7751, 0.2425, 0.1441, 0.1157, 0.1070, 0.1047, 0.1110]),
    'hb':   np.array([0.7560, 0.2523, 0.1581, 0.1243, 0.1133, 0.1122, 0.1170]),
    'Q13':  np.array([0.7486, 0.3189, 0.2618, 0.2076, 0.1869, 0.1904, 0.2016]),
    'Q24':  np.array([0.8320, 0.3231, 0.2333, 0.2054, 0.1928, 0.1938, 0.2085]),
    'full': np.array([2.4706, 0.6725, 0.3715, 0.2721, 0.2345, 0.2279, 0.2311]),
}


def analyze(data, label):
    """Full cross-term decomposition for one α value."""

    Q1, Q2, Q3, Q4 = data['Q1'], data['Q2'], data['Q3'], data['Q4']
    hf, hb = data['hf'], data['hb']
    full = data['full']

    # Diagonal = incoherent sum of 4 quarters
    diag = Q1 + Q2 + Q3 + Q4

    # Total cross-term
    cross_total = full - diag

    # Sub-cross-terms from intermediate groupings
    cross_adj_front = hf - Q1 - Q2      # adjacent Q1+Q2 → half-front
    cross_adj_back = hb - Q3 - Q4       # adjacent Q3+Q4 → half-back
    cross_distant = full - hf - hb       # halves → full

    # ── Sanity checks ──
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Check: diag + cross = full
    err = np.max(np.abs(diag + cross_total - full))
    print(f"\n  Sanity: max|diag + cross - full| = {err:.2e}")

    # Check: cross_adj_front + cross_adj_back + cross_distant
    #       + (Q1+Q2+Q3+Q4) = full?
    # Actually: hf = Q1+Q2+cross_adj_front, hb = Q3+Q4+cross_adj_back
    # full = hf + hb + cross_distant
    # = Q1+Q2+Q3+Q4 + cross_adj_front + cross_adj_back + cross_distant
    # = diag + cross_adj_front + cross_adj_back + cross_distant
    # So cross_total = cross_adj_front + cross_adj_back + cross_distant
    cross_sum = cross_adj_front + cross_adj_back + cross_distant
    err2 = np.max(np.abs(cross_total - cross_sum))
    print(f"  Sanity: max|cross_total - (adj_f + adj_b + dist)| = {err2:.2e}")

    # ── Raw cross-terms ──
    print(f"\n  Raw cross-terms:")
    print(f"  {'k':>5s}  {'diag':>8s}  {'cross_tot':>9s}  {'adj_front':>9s}"
          f"  {'adj_back':>9s}  {'distant':>9s}  {'full':>8s}")
    print(f"  {'-'*60}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {diag[i]:8.2f}  {cross_total[i]:9.3f}"
              f"  {cross_adj_front[i]:9.3f}  {cross_adj_back[i]:9.3f}"
              f"  {cross_distant[i]:9.3f}  {full[i]:8.2f}")

    # ── sin²(k) × quantities (integrand decomposition) ──
    s2_full = sin2 * full
    s2_diag = sin2 * diag
    s2_cross = sin2 * cross_total
    s2_adj_f = sin2 * cross_adj_front
    s2_adj_b = sin2 * cross_adj_back
    s2_dist = sin2 * cross_distant

    print(f"\n  sin²(k) × cross-terms:")
    print(f"  {'k':>5s}  {'s2*diag':>8s}  {'s2*cross':>9s}  {'s2*adj_f':>9s}"
          f"  {'s2*adj_b':>9s}  {'s2*dist':>9s}  {'s2*full':>8s}")
    print(f"  {'-'*60}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {s2_diag[i]:8.3f}  {s2_cross[i]:9.3f}"
              f"  {s2_adj_f[i]:9.3f}  {s2_adj_b[i]:9.3f}"
              f"  {s2_dist[i]:9.3f}  {s2_full[i]:8.3f}")

    # ── CV table ──
    print(f"\n  CV of sin²(k) × quantity:")
    print(f"    sin²·full:       {cv(s2_full):6.1f}%")
    print(f"    sin²·diag:       {cv(s2_diag):6.1f}%")
    print(f"    sin²·cross_tot:  {cv(s2_cross):6.1f}%")
    print(f"    sin²·adj_front:  {cv(s2_adj_f):6.1f}%")
    print(f"    sin²·adj_back:   {cv(s2_adj_b):6.1f}%")
    print(f"    sin²·distant:    {cv(s2_dist):6.1f}%")

    # ── Cross/full fraction at each k ──
    print(f"\n  cross/full fraction at each k:")
    print(f"  {'k':>5s}  {'cross/full':>10s}  {'diag/full':>10s}"
          f"  {'adj/full':>9s}  {'dist/full':>10s}")
    print(f"  {'-'*50}")
    for i in range(len(k_vals)):
        adj_i = (cross_adj_front[i] + cross_adj_back[i]) / full[i]
        print(f"  {k_vals[i]:5.2f}  {cross_total[i]/full[i]:10.3f}"
              f"  {diag[i]/full[i]:10.3f}  {adj_i:9.3f}"
              f"  {cross_distant[i]/full[i]:10.3f}")

    # ── Hierarchical flattening ──
    print(f"\n  Hierarchical flattening (CV at each level):")

    # Per-quarter
    cvs_q = [cv(sin2 * q) for q in [Q1, Q2, Q3, Q4]]
    print(f"    Quarters:       mean CV = {np.mean(cvs_q):.1f}%"
          f"  (range {min(cvs_q):.1f}-{max(cvs_q):.1f}%)")

    # Half-rings (FDTD, includes adj interference)
    cv_hf = cv(sin2 * hf)
    cv_hb = cv(sin2 * hb)
    print(f"    Half-front:     CV = {cv_hf:.1f}%")
    print(f"    Half-back:      CV = {cv_hb:.1f}%")

    # Full ring
    cv_full = cv(s2_full)
    print(f"    Full ring:      CV = {cv_full:.1f}%")

    # Incoherent sums (no interference)
    cv_q12_inc = cv(sin2 * (Q1 + Q2))
    cv_q34_inc = cv(sin2 * (Q3 + Q4))
    cv_hh_inc = cv(sin2 * (hf + hb))

    print(f"\n  Interference effect on CV:")
    print(f"    Level 1 — adjacent Q → half (destructive interference):")
    print(f"      Q1+Q2 incoh: {cv_q12_inc:.1f}%  →  half-front FDTD: {cv_hf:.1f}%"
          f"  (ratio {cv_hf/cv_q12_inc:.2f})")
    print(f"      Q3+Q4 incoh: {cv_q34_inc:.1f}%  →  half-back  FDTD: {cv_hb:.1f}%"
          f"  (ratio {cv_hb/cv_q34_inc:.2f})")

    print(f"    Level 2 — distant halves → full:")
    print(f"      hf+hb  incoh: {cv_hh_inc:.1f}%  →  full ring  FDTD: {cv_full:.1f}%"
          f"  (ratio {cv_full/cv_hh_inc:.2f})")

    half_mean = (cv_hf + cv_hb) / 2
    half_flatter = half_mean < cv_full
    print(f"\n    Half-rings flatter than full? "
          f"{'YES' if half_flatter else 'NO'}"
          f"  (half avg: {half_mean:.1f}%, full: {cv_full:.1f}%)")

    # ── Adjacent cross-term shape: adj/half ratio at each k ──
    adj_over_hf = cross_adj_front / hf
    adj_over_hb = cross_adj_back / hb

    print(f"\n  Adjacent cross/half ratio at each k:")
    print(f"  {'k':>5s}  {'adj_f/hf':>9s}  {'adj_b/hb':>9s}")
    print(f"  {'-'*25}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {adj_over_hf[i]:9.3f}  {adj_over_hb[i]:9.3f}")

    return {
        'diag': diag, 'cross_total': cross_total,
        'cross_adj_f': cross_adj_front, 'cross_adj_b': cross_adj_back,
        'cross_dist': cross_distant,
        'cv_full': cv_full, 'cv_diag': cv(s2_diag),
        'cv_cross': cv(s2_cross),
        'cv_quarters': np.mean(cvs_q),
        'cv_hf': cv_hf, 'cv_hb': cv_hb,
        'flattening_adj': ((cv_hf / cv_q12_inc) + (cv_hb / cv_q34_inc)) / 2,
        'flattening_dist': cv_full / cv_hh_inc,
    }


if __name__ == '__main__':
    print("Route 48: Cross-term spectral shape analysis")
    print(f"  k = {list(k_vals)}")
    print(f"  Data from files 18, 45, 46 (zero compute)")

    r030 = analyze(data_030, "α = 0.30 (flat integrand, file 45)")
    r010 = analyze(data_010, "α = 0.10 (not flat, file 46)")

    # ═══════════════════════════════════════════════════════════════════════
    # Comparison table
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  COMPARISON: α=0.30 vs α=0.10")
    print("=" * 70)

    print(f"\n  {'Quantity':>25s}  {'α=0.30':>8s}  {'α=0.10':>8s}")
    print(f"  {'-'*45}")
    print(f"  {'CV(sin²·full)':>25s}  {r030['cv_full']:7.1f}%  {r010['cv_full']:7.1f}%")
    print(f"  {'CV(sin²·diag)':>25s}  {r030['cv_diag']:7.1f}%  {r010['cv_diag']:7.1f}%")
    print(f"  {'CV(sin²·cross)':>25s}  {r030['cv_cross']:7.1f}%  {r010['cv_cross']:7.1f}%")
    print(f"  {'CV(quarter mean)':>25s}  {r030['cv_quarters']:7.1f}%  {r010['cv_quarters']:7.1f}%")
    print(f"  {'CV(half-front)':>25s}  {r030['cv_hf']:7.1f}%  {r010['cv_hf']:7.1f}%")
    print(f"  {'CV(half-back)':>25s}  {r030['cv_hb']:7.1f}%  {r010['cv_hb']:7.1f}%")
    fa030 = r030['flattening_adj']
    fa010 = r010['flattening_adj']
    fd030 = r030['flattening_dist']
    fd010 = r010['flattening_dist']
    print(f"  {'adj flattening ratio':>25s}  {fa030:6.2f}x {'F' if fa030<1 else 'D':1s}  {fa010:6.2f}x {'F' if fa010<1 else 'D':1s}")
    print(f"  {'dist flattening ratio':>25s}  {fd030:6.2f}x {'F' if fd030<1 else 'D':1s}  {fd010:6.2f}x {'F' if fd010<1 else 'D':1s}")
    print(f"  {'':>25s}  {'(F=flattens, D=degrades)':>16s}")

    # ═══════════════════════════════════════════════════════════════════════
    # Cross-term / full fraction comparison
    # ═══════════════════════════════════════════════════════════════════════
    cf_030 = r030['cross_total'] / data_030['full']
    cf_010 = r010['cross_total'] / data_010['full']

    print(f"\n  cross/full ratio at each k:")
    print(f"  {'k':>5s}  {'α=0.30':>8s}  {'α=0.10':>8s}")
    print(f"  {'-'*25}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {cf_030[i]:8.3f}  {cf_010[i]:8.3f}")
    mean_cf_030 = np.mean(cf_030)
    mean_cf_010 = np.mean(cf_010)
    print(f"  {'mean':>5s}  {mean_cf_030:8.3f}  {mean_cf_010:8.3f}")

    diff_cf = mean_cf_030 - mean_cf_010
    print(f"\n  Mean cross/full: {mean_cf_030:.3f} (α=0.30) vs {mean_cf_010:.3f} (α=0.10)")
    print(f"  Difference: {diff_cf:+.3f} — "
          f"{'GEOMETRIC (α-independent)' if abs(diff_cf) < 0.05 else 'α-DEPENDENT'}")

    # ═══════════════════════════════════════════════════════════════════════
    # Statistical averaging test: CV ~ 1/sqrt(N)?
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  STATISTICAL AVERAGING: CV vs 1/sqrt(N)")
    print("  If bonds scatter independently: CV(N) = CV(1) / sqrt(N)")
    print("=" * 70)

    # At α=0.30, single bond CV=71% from tracker (F4, file 35)
    cv1_030 = 71.0

    print(f"\n  α=0.30 (CV_1 = 71% from file 35, measured single-bond data):")
    for name, N, cv_meas in [('quarter (~20)', 20, r030['cv_quarters']),
                              ('half (~40)', 40, (r030['cv_hf'] + r030['cv_hb']) / 2),
                              ('full (81)', 81, r030['cv_full'])]:
        cv_pred = cv1_030 / np.sqrt(N)
        print(f"    {name:>16s}: measured={cv_meas:5.1f}%  predicted={cv_pred:5.1f}%"
              f"  ratio={cv_meas/cv_pred:.2f}")

    # α=0.10: NO measured single-bond CV. Cannot infer from quarter data
    # because file 46 shows ±40% interference WITHIN quarters (bonds are
    # NOT independent inside a quarter). CV_1 × 1/√N is invalid here.
    print(f"\n  α=0.10 (NO measured single-bond CV — inference invalid):")
    print(f"    Cannot use CV(quarter) × √N to infer CV(single bond).")
    print(f"    File 46 shows ±40% interference within quarters →")
    print(f"    bonds are NOT independent, 1/√N model does not apply.")
    print(f"    Need single-bond FDTD at α=0.10 (cf. file 35 at α=0.30).")

    # Still show quarter→full scaling (empirical, not 1/√N)
    print(f"\n    Empirical scaling (no model assumed):")
    cv_q10 = r010['cv_quarters']
    cv_h10 = (r010['cv_hf'] + r010['cv_hb']) / 2
    cv_f10 = r010['cv_full']
    print(f"      quarter: {cv_q10:.1f}%  →  half: {cv_h10:.1f}%  →  full: {cv_f10:.1f}%")
    print(f"      Q→half ratio: {cv_h10/cv_q10:.2f}   half→full ratio: {cv_f10/cv_h10:.2f}")

    print(f"\n  NOTE: 1/√N works at α=0.30 (quarter and full match prediction).")
    print(f"  Half at α=0.30 is BELOW prediction (ratio 0.42) — adjacent")
    print(f"  destructive interference provides extra flattening beyond 1/√N.")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY (all values from computed results, not hardcoded)
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    half_030 = (r030['cv_hf'] + r030['cv_hb']) / 2
    half_010 = (r010['cv_hf'] + r010['cv_hb']) / 2

    print()
    print(f"  1. sin²(k)·σ_cross is NOT flat:"
          f" CV={r030['cv_cross']:.1f}% (α=0.30), {r010['cv_cross']:.1f}% (α=0.10).")
    print(f"     The cross-term is highly k-shaped: positive at k=0.3,")
    print(f"     increasingly negative at k≥0.5.")

    print()
    print(f"  2. sin²(k)·σ_diag is NOT flat either:"
          f" CV={r030['cv_diag']:.1f}% (α=0.30), {r010['cv_diag']:.1f}% (α=0.10).")
    print(f"     But diag + cross = full IS flat ({r030['cv_full']:.1f}%) at α=0.30.")
    print(f"     The shaped cross-term compensates the diag excess at high k.")

    print()
    print(f"  3. Two-level flattening hierarchy (α=0.30):")
    print(f"     Quarter ({r030['cv_quarters']:.1f}%)"
          f" → Half ({half_030:.1f}%)"
          f" → Full ({r030['cv_full']:.1f}%)")
    print(f"     Adjacent destructive interference:"
          f" ×{r030['flattening_adj']:.2f} (Q→half)")
    print(f"     Distant constructive interference:"
          f" ×{r030['flattening_dist']:.2f} (half→full,"
          f" {'DEGRADES' if r030['flattening_dist'] > 1 else 'flattens'})")

    print()
    if half_030 < r030['cv_full'] and r030['cv_hf'] < r030['cv_full']:
        print(f"  4. PARADOX: half-rings FLATTER than full ring at α=0.30"
              f" ({half_030:.1f}% vs {r030['cv_full']:.1f}%).")
        print(f"     Cause: constructive distant interference at k=0.3 (T6: +28%)")
        print(f"     lifts integrand at low k, DEGRADING flatness of full ring.")
        print(f"     Optimal flatness at half-ring level, NOT full ring.")
    else:
        print(f"  4. Half-rings vs full at α=0.30:"
              f" half avg={half_030:.1f}%, full={r030['cv_full']:.1f}%.")

    print()
    if half_010 > r010['cv_full']:
        print(f"  5. Hierarchy REVERSES at α=0.10:")
        print(f"     Quarter ({r010['cv_quarters']:.1f}%)"
              f" → Half ({half_010:.1f}%)"
              f" → Full ({r010['cv_full']:.1f}%)")
        print(f"     Both adjacent (×{r010['flattening_adj']:.2f})"
              f" and distant (×{r010['flattening_dist']:.2f}) help flatten.")
        print(f"     Halves are NOT flatter than full ring here.")
    else:
        print(f"  5. α=0.10: Quarter ({r010['cv_quarters']:.1f}%)"
              f" → Half ({half_010:.1f}%)"
              f" → Full ({r010['cv_full']:.1f}%)")

    print()
    print(f"  6. α-dependence is primarily in the DIAGONAL:")
    print(f"     Per-quarter CV: {r030['cv_quarters']:.1f}% (α=0.30)"
          f" vs {r010['cv_quarters']:.1f}% (α=0.10).")
    print(f"     Non-Born enhancement at strong coupling makes individual")
    print(f"     bonds flatter. Geometric interference then finishes the job.")

    print()
    print(f"  7. Mean cross/full fraction:"
          f" {mean_cf_030:.3f} (α=0.30) vs {mean_cf_010:.3f} (α=0.10).")
    geom_label = "GEOMETRIC" if abs(diff_cf) < 0.05 else "α-DEPENDENT"
    print(f"     Difference {diff_cf:+.3f} — {geom_label}.")
    print(f"     Interference removes ~{abs(mean_cf_030)*100:.0f}% of the"
          f" incoherent sum at both couplings.")
    print(f"     Consistent with file 46 finding (identical ±35% pattern).")
