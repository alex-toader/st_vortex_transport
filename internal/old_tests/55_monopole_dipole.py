"""
Route 55 (I18): Born mechanism for σ_bond. Per-bond Born + collective N_eff → flat ring.

Central question: WHY is sin²(k)·σ_ring ≈ const (CV=7.4%)?

Answer (2 layers — per-bond DERIVED, ring-level EMPIRIC):
  Per-bond (COMPLETE): σ_bond ≈ C₀ · V(α) / v_g²  (Born, first principles)
    FDTD/Born shape match CV=4.6% per bond. Fully derived.
  Ring (SEMI-EMPIRIC): σ_ring = σ_bond · N_eff(k), N_eff ∝ 1/sin²(k/2).
    cos²(k/2) cancellation is algebraic (DERIVED).
    sin²(k/2) · N_eff ≈ const (CV=5.4%) is CIRCULAR: follows tautologically
    from σ_ring ∝ 1/sin²(k) (empiric, file 18) + σ_bond ∝ 1/cos²(k/2) (Born).
    WHY N_eff ∝ 1/sin²(k/2) from the ring geometry is OPEN.

Mechanism ingredients:
  1. Monopole/dipole source decomposition (z-bond, x-incident):
     - x-channel: MONOPOLE (same forces at both sites), Z_mono = 8π(1+sinc(k)) ≈ const
     - y-channel: DIPOLE (opposite forces), Z_dipo = 8π(1-sinc(k)) → grows 22×
  2. Born normalization: σ_bond ∝ V / v_g², where v_g = √(K1+4K2)·cos(k/2)
     The old 1/sin²(k) had a spurious 1/(4sin²(k/2)) factor varying 20.8×.
  3. α-threshold: |cm1|>|s_phi| at α>0.25 → monopole dominates → V ≈ const → flat

NOTE: Ring uses only z-bonds (Peierls in xy-plane). For x-bonds the form factor
  is 4cos²((k+q_x)/2) ≠ 4cos²(q_z/2) — direction-dependent.
  This does not affect the ring mechanism.

NOTE: Per-bond is Born (CV=4.6%). Ring is non-Born at the α-scaling level
  (σ_ring ∝ α^2.56, file 26) because N_eff encodes collective interference.

Verification (zero-compute, uses FDTD data from files 18, 51, 53):
  Part A: Z_mono vs Z_dipo angular integrals
  Part B: Correct Born formula: σ = C₀ × V / v_g² (not 1/sin²(k))
  Part C: FDTD vs Born for σ_xx, σ_xy, σ_tot (per z-bond)
  Part D: Flat integrand chain: sin²(k)·σ_ring decomposition
  Part E: α-threshold and CV(V) scan
  Part F: Source structure derivation

Results (0s):

  Part A: Z_mono CV=5.9%, Z_dipo grows 22.4×.
  Part B: 1/v_g² varies 1.83×. 1/sin²(k) varies 11.4×. Spurious factor 20.8×.
  Part C: FDTD/Born ratio CV: σ_xx 4.8%, σ_xy 11.0%, σ_tot 4.6%. Born SHAPE matches.
  Part D: sin²(k)·σ_ring CV=7.4%. sin²(k/2)·N_eff CV=5.4%. V CV=2.7%.
  Part E: |cm1|=|s_phi| at α=0.25 exactly.
          CV(V): α=0.10→28.2%, α=0.20→4.6%, α=0.25→0.0%, α=0.30→2.7%, α=0.50→5.9%.

  Per-bond: DERIVED (Born, first principles). FDTD confirms at ~5%.
  Ring-level: N_eff ∝ 1/sin²(k/2) EMPIRIC (not derived from ring geometry).
  Mechanism ~60% explained: per-bond complete, ring collective open.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/55_monopole_dipole.py
"""

import numpy as np

K1, K2 = 1.0, 0.5
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# FDTD data from file 53 (α=0.30, single z-bond, polarization-resolved)
sigma_xx_fdtd = np.array([0.0595, 0.0535, 0.0553, 0.0593, 0.0651, 0.0713, 0.0804])
sigma_xy_fdtd = np.array([0.0003, 0.0008, 0.0017, 0.0032, 0.0056, 0.0094, 0.0158])
sigma_tot_fdtd = sigma_xx_fdtd + sigma_xy_fdtd

# Sanity vs file 51 (σ_bond total)
sigma_51 = np.array([0.0598, 0.0543, 0.0571, 0.0624, 0.0707, 0.0807, 0.0962])
assert np.max(np.abs(sigma_tot_fdtd - sigma_51) / sigma_51) < 0.02

# FDTD ring data from file 18 (α=0.30, R=5, 81 bonds)
sigma_ring = np.array([40.74, 14.16, 7.69, 5.05, 3.78, 3.14, 2.81])


def cv(x):
    return np.std(x) / np.abs(np.mean(x)) * 100


if __name__ == '__main__':
    print("Route 55: Born mechanism for σ_bond + flat ring integrand")
    print()

    alpha = 0.30
    cm1 = np.cos(2 * np.pi * alpha) - 1
    s_phi = np.sin(2 * np.pi * alpha)
    v_g = np.sqrt(K1 + 4 * K2) * np.cos(k_vals / 2)

    print(f"  α = {alpha}, cm1 = {cm1:.4f}, s_phi = {s_phi:.4f}")
    print(f"  |cm1|/|s_phi| = {abs(cm1) / abs(s_phi):.3f}"
          f" ({'cm1 dominates' if abs(cm1) > abs(s_phi) else 's_phi dominates'})")

    # ═══════════════════════════════════════════════════════════════
    # Part A: Z_mono vs Z_dipo (angular integrals of form factors)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part A: Monopole vs dipole angular integrals")
    print(f"{'=' * 60}")

    Z_mono = 8 * np.pi * (1 + np.sin(k_vals) / k_vals)
    Z_dipo = 8 * np.pi * (1 - np.sin(k_vals) / k_vals)

    print(f"\n  Z-bond form factors (x-incident, z-bond):")
    print(f"    Monopole (x-channel): |1+exp(-iq_z)|² = 4cos²(q_z/2)")
    print(f"    Dipole (y-channel):   |1-exp(-iq_z)|² = 4sin²(q_z/2)")
    print(f"    Z = ∫ |F|² · (1 - cosθ_s) · sinθ dθ dφ")

    print(f"\n  {'k':>5s}  {'Z_mono':>8s}  {'Z_dipo':>8s}  {'ratio':>8s}")
    print(f"  {'-' * 35}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {Z_mono[i]:8.2f}  {Z_dipo[i]:8.4f}"
              f"  {Z_mono[i] / Z_dipo[i]:8.1f}")

    print(f"\n  Z_mono: {Z_mono[0]:.2f} → {Z_mono[-1]:.2f}"
          f" ({Z_mono[0] / Z_mono[-1]:.2f}×), CV={cv(Z_mono):.1f}%")
    print(f"  Z_dipo: {Z_dipo[0]:.4f} → {Z_dipo[-1]:.2f}"
          f" ({Z_dipo[-1] / Z_dipo[0]:.1f}×)")

    # ═══════════════════════════════════════════════════════════════
    # Part B: Correct Born normalization (1/v_g², not 1/sin²(k))
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part B: Born normalization — 1/v_g², not 1/sin²(k)")
    print(f"{'=' * 60}")

    print(f"\n  Lattice Green function far-field: G ~ exp(ik·r) / (r · v_g)")
    print(f"  σ_bond ∝ V / v_g²")
    print(f"  v_g = √(K1+4K2) · cos(k/2)")
    print(f"\n  Old (WRONG): σ ∝ V / sin²(k) = V / [4sin²(k/2)cos²(k/2)]")
    print(f"  Correct:      σ ∝ V / v_g² ∝ V / cos²(k/2)")
    print(f"\n  The spurious factor 1/(4sin²(k/2)) varies"
          f" {1 / (4 * np.sin(k_vals[0] / 2)**2):.1f} → "
          f"{1 / (4 * np.sin(k_vals[-1] / 2)**2):.2f}"
          f" ({4 * np.sin(k_vals[-1] / 2)**2 / (4 * np.sin(k_vals[0] / 2)**2):.1f}×)")

    print(f"\n  {'k':>5s}  {'v_g':>6s}  {'1/v_g²':>8s}  {'1/sin²k':>8s}  {'ratio':>8s}")
    print(f"  {'-' * 40}")
    for i in range(len(k_vals)):
        ivg2 = 1 / v_g[i]**2
        isin2 = 1 / np.sin(k_vals[i])**2
        print(f"  {k_vals[i]:5.2f}  {v_g[i]:6.4f}  {ivg2:8.4f}  {isin2:8.4f}"
              f"  {isin2 / ivg2:8.2f}")

    print(f"\n  1/v_g² varies {v_g[0]**2 / v_g[-1]**2:.2f}×."
          f" 1/sin²(k) varies {np.sin(k_vals[-1])**2 / np.sin(k_vals[0])**2:.1f}×.")

    # ═══════════════════════════════════════════════════════════════
    # Part C: FDTD vs Born per bond
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part C: FDTD vs Born per z-bond (v_g normalization)")
    print(f"{'=' * 60}")

    # Born shapes (normalized at k=0.3):
    born_xx = cm1**2 * Z_mono / v_g**2
    born_xy = s_phi**2 * Z_dipo / v_g**2
    born_tot = born_xx + born_xy

    # --- σ_xx ---
    born_xx_n = born_xx / born_xx[0]
    fdtd_xx_n = sigma_xx_fdtd / sigma_xx_fdtd[0]
    R_xx = fdtd_xx_n / born_xx_n

    print(f"\n  σ_xx (same-pol, monopole channel):")
    print(f"  {'k':>5s}  {'FDTD_n':>8s}  {'Born_n':>8s}  {'FDTD/Born':>10s}")
    print(f"  {'-' * 35}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {fdtd_xx_n[i]:8.4f}  {born_xx_n[i]:8.4f}"
              f"  {R_xx[i]:10.4f}")
    print(f"  CV(FDTD/Born) = {cv(R_xx):.1f}%")

    # --- σ_xy ---
    # Normalize at k=0.9 (Z_dipo ≈ 0 at k=0.3 causes instability)
    born_xy_m = born_xy / born_xy[3]
    fdtd_xy_m = sigma_xy_fdtd / sigma_xy_fdtd[3]
    R_xy = fdtd_xy_m / born_xy_m

    print(f"\n  σ_xy (cross-pol, dipole channel, normalized at k=0.9):")
    print(f"  {'k':>5s}  {'FDTD_n':>8s}  {'Born_n':>8s}  {'FDTD/Born':>10s}")
    print(f"  {'-' * 35}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {fdtd_xy_m[i]:8.4f}  {born_xy_m[i]:8.4f}"
              f"  {R_xy[i]:10.4f}")
    print(f"  CV(FDTD/Born) = {cv(R_xy):.1f}%")

    # --- σ_tot ---
    born_tot_n = born_tot / born_tot[0]
    fdtd_tot_n = sigma_tot_fdtd / sigma_tot_fdtd[0]
    R_tot = fdtd_tot_n / born_tot_n

    print(f"\n  σ_tot (per z-bond):")
    print(f"  {'k':>5s}  {'FDTD_n':>8s}  {'Born_n':>8s}  {'FDTD/Born':>10s}")
    print(f"  {'-' * 35}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {fdtd_tot_n[i]:8.4f}  {born_tot_n[i]:8.4f}"
              f"  {R_tot[i]:10.4f}")
    print(f"  CV(FDTD/Born) = {cv(R_tot):.1f}%")

    print(f"\n  Per-bond Born with v_g normalization matches FDTD at ~5% level.")
    print(f"  NOTE: σ_ring ∝ α^2.56 (file 26) is non-Born at ring level.")
    print(f"  This comes from N_eff (collective interference), not per-bond.")

    # ═══════════════════════════════════════════════════════════════
    # Part D: Flat integrand derivation
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part D: Flat integrand chain")
    print(f"{'=' * 60}")

    N_eff = sigma_ring / sigma_tot_fdtd

    print(f"\n  sin²(k) · σ_ring = sin²(k) · σ_bond · N_eff")
    print(f"  ≈ [4sin²(k/2)·cos²(k/2)] · [C₀ · V / cos²(k/2)] · N_eff")
    print(f"  = 4sin²(k/2) · C₀ · V · N_eff")
    print(f"  where V = cm1²·Z_mono + s²·Z_dipo (Born vertex)")
    print(f"  cos²(k/2) from sin²(k) CANCELS 1/cos²(k/2) from v_g².")
    print(f"  (≈ because per-bond Born is not exact — 4.6% residual)")

    integrand = np.sin(k_vals)**2 * sigma_ring
    print(f"\n  sin²(k)·σ_ring: CV = {cv(integrand):.1f}%")

    # Born vertex V
    V = cm1**2 * Z_mono + s_phi**2 * Z_dipo
    print(f"  V = cm1²·Z_mono + s_phi²·Z_dipo: CV = {cv(V):.1f}%")

    # N_eff
    print(f"\n  N_eff = σ_ring / σ_bond (collective enhancement):")
    print(f"  {'k':>5s}  {'σ_bond':>8s}  {'σ_ring':>8s}  {'N_eff':>8s}")
    print(f"  {'-' * 35}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {sigma_tot_fdtd[i]:8.4f}"
              f"  {sigma_ring[i]:8.2f}  {N_eff[i]:8.1f}")
    print(f"  N_eff: {N_eff[0]:.0f} → {N_eff[-1]:.0f} ({N_eff[0] / N_eff[-1]:.1f}×)")
    print(f"  81 bonds but N_eff(k=0.3) = {N_eff[0]:.0f} >> 81 (constructive)")
    print(f"  N_eff(k=1.5) = {N_eff[-1]:.0f} < 81 (partial destructive)")

    # Key balance: sin²(k/2) * N_eff ≈ const
    balance = np.sin(k_vals / 2)**2 * N_eff
    balance_n = balance / balance[0]

    print(f"\n  Key balance: sin²(k/2) · N_eff ≈ const")
    print(f"  {'k':>5s}  {'sin²(k/2)':>10s}  {'N_eff':>8s}  {'product':>8s}  {'norm':>8s}")
    print(f"  {'-' * 45}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {np.sin(k_vals[i] / 2)**2:10.4f}"
              f"  {N_eff[i]:8.1f}  {balance[i]:8.2f}  {balance_n[i]:8.4f}")
    print(f"  CV = {cv(balance_n):.1f}%")
    print(f"\n  N_eff ∝ 1/sin²(k/2) — lattice analog of continuum 1/k².")

    # Full chain verification
    remaining = 4 * np.sin(k_vals / 2)**2 * V * N_eff
    remaining_n = remaining / remaining[0]
    print(f"\n  Full chain: 4sin²(k/2) · V · N_eff (normalized):")
    print(f"  {'k':>5s}  {'4sin²k/2':>9s}  {'V':>8s}  {'N_eff':>8s}  {'product_n':>10s}  {'integ_n':>8s}")
    print(f"  {'-' * 55}")
    integ_n = integrand / integrand[0]
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {4 * np.sin(k_vals[i] / 2)**2:9.4f}"
              f"  {V[i]:8.2f}  {N_eff[i]:8.1f}  {remaining_n[i]:10.4f}  {integ_n[i]:8.4f}")
    print(f"  Chain CV = {cv(remaining_n):.1f}%, integrand CV = {cv(integ_n):.1f}%")
    print(f"  Residual (integ/chain) CV = {cv(integrand / remaining):.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # Part E: α-threshold and CV(V) scan
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part E: α-threshold and CV(V) scan")
    print(f"{'=' * 60}")

    print(f"\n  V(k) = cm1²·Z_mono(k) + s_phi²·Z_dipo(k)")
    print(f"  V ≈ const when monopole dominates (α > 0.25)")

    print(f"\n  {'α':>6s}  {'cm1²':>8s}  {'s_phi²':>8s}  {'cm1²/s²':>8s}"
          f"  {'CV(V)%':>8s}  {'note':>12s}")
    print(f"  {'-' * 60}")
    for a in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        c = np.cos(2 * np.pi * a) - 1
        s = np.sin(2 * np.pi * a)
        Vscan = c**2 * Z_mono + s**2 * Z_dipo
        ratio_cs = c**2 / s**2 if s**2 > 1e-10 else float('inf')
        note = ''
        if abs(ratio_cs - 1) < 0.1:
            note = 'THRESHOLD'
        elif ratio_cs > 1:
            note = 'mono dom'
        else:
            note = 'dipo dom'
        print(f"  {a:6.2f}  {c**2:8.4f}  {s**2:8.4f}  {ratio_cs:8.2f}"
              f"  {cv(Vscan):8.1f}  {note:>12s}")

    print(f"\n  |cm1| = |s_phi| at α = 0.25 exactly.")
    print(f"  α > 0.25: monopole dominates → V ≈ cm1²·Z_mono ≈ const (CV < 6%)")
    print(f"  α < 0.20: dipole significant → V varies with k (CV > 5%)")
    print(f"  α = 0.25: exact minimum CV = 0.0% (Z_mono and Z_dipo cancel)")

    # ═══════════════════════════════════════════════════════════════
    # Part F: Source structure
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part F: Source structure (z-bond, from gauge_3d.py)")
    print(f"{'=' * 60}")

    print(f"""
  Peierls perturbation on z-bond at (x0,y0,z0)-(x0,y0,z0+1):
    delta_Fx(z0)   = K1 * (cm1*ux(z0+1) - s_phi*uy(z0+1))
    delta_Fx(z0+1) = K1 * (cm1*ux(z0)   + s_phi*uy(z0))
    delta_Fy(z0)   = K1 * (s_phi*ux(z0+1) + cm1*uy(z0+1))
    delta_Fy(z0+1) = K1 * (-s_phi*ux(z0) + cm1*uy(z0))

  For x-incident plane wave: ux = A*exp(ikx), uy = 0
  u(z0+1) = u(z0) = A*exp(ikx0)  [no z-dependence for x-incident]

    delta_Fx(z0)   = K1*cm1*A*exp(ikx0)    ← same sign   → MONOPOLE
    delta_Fx(z0+1) = K1*cm1*A*exp(ikx0)
    delta_Fy(z0)   = +K1*s_phi*A*exp(ikx0) ← opposite    → DIPOLE
    delta_Fy(z0+1) = -K1*s_phi*A*exp(ikx0)

  Assumption: u(z+1) = u(z) for x-incident wave (valid at sx=8: Δk/k<0.1).

  Monopole form factor: |1+exp(-iq_z)|² = 4cos²(q_z/2)
    → Z_mono = 8π(1+sin(k)/k) ≈ const (CV={cv(Z_mono):.1f}%)
  Dipole form factor:  |1-exp(-iq_z)|² = 4sin²(q_z/2)
    → Z_dipo = 8π(1-sin(k)/k) → grows {Z_dipo[-1] / Z_dipo[0]:.0f}×

  NOTE on bond direction: For z-bond with x-incident wave, both sites see
  the same incident phase → monopole form factor is 4cos²(q_z/2), which
  depends only on scattered q_z. For an x-bond along the propagation
  direction, the two sites see DIFFERENT incident phases (exp(ikx0) vs
  exp(ik(x0+1))), giving form factor |exp(ik)+exp(-iq_x)|² = 4cos²((k+q_x)/2)
  which depends on BOTH k and q_x. The ring (all z-bonds) uses only the
  z-bond formula.""")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"""
  σ_bond = C₀ × [cm1²·Z_mono + s_phi²·Z_dipo] / v_g²   (per z-bond)

  FDTD/Born shape ratios (per z-bond):
    σ_xx: CV = {cv(R_xx):.1f}%  (monopole channel)
    σ_xy: CV = {cv(R_xy):.1f}%  (dipole channel)
    σ_tot: CV = {cv(R_tot):.1f}%

  Flat ring integrand: sin²(k)·σ_ring ≈ const (CV={cv(integrand):.1f}%) because:
    1. cos²(k/2) in sin²(k) cancels 1/cos²(k/2) in v_g²
    2. sin²(k/2) × N_eff ≈ const (CV={cv(balance_n):.1f}%)
    3. Born vertex V ≈ const at α≥0.25 (CV={cv(V):.1f}% at α=0.30)

  Two levels:
    Per-bond: Born with v_g normalization (CV=4.6%). Old 1/sin²(k) was wrong.
    Ring: N_eff(k) = collective interference, ∝ 1/sin²(k/2).
      Ring non-Born (α^2.56) comes from N_eff, not per-bond deviation.""")
