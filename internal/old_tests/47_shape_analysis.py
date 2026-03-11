"""
Route 47: Shape analysis — why is sin²(k)·σ_tr flat at α=0.30?

After files 45-46 proved scattering is NOT incoherent (±40% interference
between quarters, identical at α=0.10 and α=0.30), we need to understand
why the FDTD integrand is flat.

Two candidate shapes for σ_tr(k):
  Coherent Born:    σ ∝ I_tr(k, R) / sin²(k)     [drops 663× from k=0.3 to 1.5]
  Incoherent Born:  σ ∝ Z_avg(k) / sin²(k)       [Z_avg varies <20%]

  I_tr(k) = ∫ |F_disk(q)|² × Z(k,θ) × (1-cosθ_s) dΩ   [coherent form factor]
  Z_avg(k) = 8π(1 + sin(k)/k)                            [single-bond z-structure]
  Z(k,θ) = 4cos²(k cosθ / 2)                             [z-bond displacement vertex]

Test: compute ratio σ_FDTD(k) / σ_model(k) and check CV.
  CV small → FDTD follows that model's k-shape.

Data sources:
  σ_FDTD(α=0.30, R=5): file 18   [40.74, 14.16, 7.69, 5.05, 3.78, 3.14, 2.81]
  σ_FDTD(α=0.10, R=5): file 46   [2.4706, 0.6725, 0.3715, 0.2721, 0.2345, 0.2279, 0.2311]
  I_tr(k, R=5): computed here (same as file 43, zero FDTD)

Results (~5s, pure analytic):

  Shape match (CV of σ_FDTD / σ_model, normalized):

                Model    α=0.30   α=0.10
        Coherent Born     85.0%   100.6%
      Incoherent Born      2.7%    19.8%
       (integrand CV)      7.4%    15.3%

  CV identical at k=0.3 vs k=0.9 normalization (CV is norm-independent).

  Key findings:
  1. σ_FDTD(α=0.30) follows INCOHERENT Born shape (CV=2.7%)
     despite scattering being COHERENT (file 46: ±40% interference).
  2. σ_FDTD(α=0.10) does NOT follow incoherent Born (CV=19.8%).
  3. Non-Born enhancement R(k) = σ_FDTD / σ_Born_coh grows with k:
     R(0.30) grows 45.8×, R(0.10) grows 62.1× (Born drops 663×).
  4. α-enhancement: mean σ(0.30)/σ(0.10) = 17.0 (Born pred: 6.9).
     Born for full Peierls: sin²(0.3π)/sin²(0.1π) = 6.9 (not 47.0 = cm1² ratio).
     FDTD/Born = 2.5 — ring FDTD overshoots Born α-ratio.
     Per-bond FDTD/Born = 3.8 (file 49); ring reduces excess from 3.8× to 2.5×.

  Z_avg analytic vs numeric: match to 0.003% (grid converged).

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/47_shape_analysis.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../src/1_foam'))
from gauge_3d import precompute_disk_bonds

L = 80
R_LOOP = 5
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# FDTD data (from files 18 and 46)
sigma_030 = np.array([40.74, 14.16, 7.69, 5.05, 3.78, 3.14, 2.81])
sigma_010 = np.array([2.4706, 0.6725, 0.3715, 0.2721, 0.2345, 0.2279, 0.2311])

# ═══════════════════════════════════════════════════════════════════════
# 1. Compute coherent Born I_tr(k, R=5) — same method as file 43
# ═══════════════════════════════════════════════════════════════════════

iy_disk, ix_disk = precompute_disk_bonds(L, R_LOOP)
cx, cy = L // 2, L // 2
positions = np.column_stack([(ix_disk - cx).astype(float),
                              (iy_disk - cy).astype(float)])
N_bonds = len(positions)

N_THETA = 150
N_PHI = 300
thetas = np.linspace(0, np.pi, N_THETA + 1)
phis = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
THETA, PHI = np.meshgrid(thetas, phis, indexing='ij')
tf = THETA.ravel()
pf = PHI.ravel()
st = np.sin(tf)
ct = np.cos(tf)
cp = np.cos(pf)
sp = np.sin(pf)

# Transport kernel: 1 - cos(scattering angle)
# For +x incidence: cos(θ_s) = sinθ cosφ
w_tr = 1 - st * cp

# Quadrature weights
dtheta = np.pi / N_THETA
dphi = 2 * np.pi / N_PHI
tw = np.ones(N_THETA + 1) * dtheta
tw[0] = dtheta / 2
tw[-1] = dtheta / 2
w_quad = np.repeat(tw, N_PHI) * dphi * np.abs(st)


if __name__ == '__main__':
    print(f"Route 47: Shape analysis")
    print(f"  R={R_LOOP}, N_bonds={N_bonds}, L={L}")
    print(f"  k = {list(k_vals)}")
    print(f"  Angular grid: {N_THETA}×{N_PHI}")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # Coherent Born I_tr(k)
    # ═══════════════════════════════════════════════════════════════════
    I_tr = np.zeros(len(k_vals))
    for j, k0 in enumerate(k_vals):
        qx = k0 * (st * cp - 1)
        qy = k0 * st * sp
        Q = np.column_stack([qx, qy])
        phase = Q @ positions.T
        F = np.sum(np.exp(1j * phase), axis=1)
        F2 = np.abs(F)**2
        Z = 4 * np.cos(k0 * ct / 2)**2
        I_tr[j] = np.sum(F2 * Z * w_tr * w_quad)

    sigma_coh_shape = I_tr / np.sin(k_vals)**2

    print("=" * 75)
    print("Coherent Born form factor I_tr(k, R=5)")
    print("  I_tr = ∫ |F_disk(q)|² × Z(k,θ) × (1-cosθ_s) dΩ")
    print("=" * 75)
    print(f"  {'k':>5s}  {'I_tr':>12s}  {'I_tr/I_tr[0]':>12s}")
    for j, k0 in enumerate(k_vals):
        print(f"  {k0:5.1f}  {I_tr[j]:12.1f}  {I_tr[j]/I_tr[0]:12.4f}")
    print(f"  I_tr drops {I_tr[0]/I_tr[-1]:.0f}× from k=0.3 to k=1.5")
    cv_Itr = np.std(I_tr) / np.mean(I_tr) * 100
    print(f"  CV(I_tr) = {cv_Itr:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # Incoherent Born Z_avg(k)
    # ═══════════════════════════════════════════════════════════════════
    # Z_avg(k) = 8π(1 + sin(k)/k)
    # Derivation: single bond at origin, |F|²=1.
    # ∫ Z(k,θ) × (1-sinθ cosφ) × sinθ dθ dφ
    # The cosφ term integrates to 0 over 2π, leaving:
    # 2π × ∫₀^π 4cos²(k cosθ/2) sinθ dθ
    # = 8π × ∫₋₁¹ cos²(ku/2) du  [u = cosθ]
    # = 8π × ∫₋₁¹ (1+cos(ku))/2 du
    # = 4π × [u + sin(ku)/k]₋₁¹
    # = 4π × [(1 + sin(k)/k) - (-1 - sin(-k)/k)]
    # = 4π × [2 + 2sin(k)/k]
    # = 8π(1 + sin(k)/k)

    Z_avg = 8 * np.pi * (1 + np.sin(k_vals) / k_vals)
    sigma_inc_shape = Z_avg / np.sin(k_vals)**2

    print()
    print("=" * 75)
    print("Incoherent Born Z_avg(k) = 8π(1 + sin(k)/k)")
    print("=" * 75)
    print(f"  {'k':>5s}  {'Z_avg':>8s}  {'Z_avg/Z[0]':>10s}")
    for j, k0 in enumerate(k_vals):
        print(f"  {k0:5.1f}  {Z_avg[j]:8.2f}  {Z_avg[j]/Z_avg[0]:10.4f}")
    cv_Z = np.std(Z_avg) / np.mean(Z_avg) * 100
    print(f"  Z_avg varies {Z_avg.min():.1f} to {Z_avg.max():.1f}  (CV={cv_Z:.1f}%)")

    # Verify Z_avg numerically (integrate Z × w_tr × w_quad with |F|²=1)
    Z_avg_num = np.zeros(len(k_vals))
    for j, k0 in enumerate(k_vals):
        Z = 4 * np.cos(k0 * ct / 2)**2
        Z_avg_num[j] = np.sum(Z * w_tr * w_quad)
    print(f"\n  Numerical verification (Z_avg_numeric / Z_avg_analytic):")
    for j, k0 in enumerate(k_vals):
        print(f"    k={k0:.1f}: {Z_avg_num[j]:.2f} / {Z_avg[j]:.2f}"
              f" = {Z_avg_num[j]/Z_avg[j]:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # Shape test: σ_FDTD / σ_model — CV measures shape match
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("SHAPE TEST: ratio = σ_FDTD / σ_model_shape, normalized to k=0.3")
    print("  CV of normalized ratio = shape mismatch")
    print("  Low CV = good shape match (normalization cancels)")
    print("=" * 75)

    # Ratios
    R_coh_030 = sigma_030 / sigma_coh_shape
    R_coh_010 = sigma_010 / sigma_coh_shape
    R_inc_030 = sigma_030 / sigma_inc_shape
    R_inc_010 = sigma_010 / sigma_inc_shape

    # Normalize to k=0.3
    Rn_coh_030 = R_coh_030 / R_coh_030[0]
    Rn_coh_010 = R_coh_010 / R_coh_010[0]
    Rn_inc_030 = R_inc_030 / R_inc_030[0]
    Rn_inc_010 = R_inc_010 / R_inc_010[0]

    # NOTE: np.std uses N denominator (biased). At 7 points, biased CV is
    # ~7% lower than unbiased (ddof=1). All files in w_21_kappa use the
    # same convention, so CVs are internally consistent.
    cv_coh_030 = np.std(Rn_coh_030) / np.mean(Rn_coh_030) * 100
    cv_coh_010 = np.std(Rn_coh_010) / np.mean(Rn_coh_010) * 100
    cv_inc_030 = np.std(Rn_inc_030) / np.mean(Rn_inc_030) * 100
    cv_inc_010 = np.std(Rn_inc_010) / np.mean(Rn_inc_010) * 100

    print(f"\n  Normalized to k=0.3:")
    print(f"  {'k':>5s}  {'coh(030)':>9s}  {'coh(010)':>9s}"
          f"  {'inc(030)':>9s}  {'inc(010)':>9s}")
    for j, k0 in enumerate(k_vals):
        print(f"  {k0:5.1f}  {Rn_coh_030[j]:9.4f}  {Rn_coh_010[j]:9.4f}"
              f"  {Rn_inc_030[j]:9.4f}  {Rn_inc_010[j]:9.4f}")

    print(f"\n  CV (shape mismatch), normalized at k=0.3:")
    print(f"    Coherent Born  → α=0.30: {cv_coh_030:.1f}%,  α=0.10: {cv_coh_010:.1f}%")
    print(f"    Incoherent Born→ α=0.30: {cv_inc_030:.1f}%,  α=0.10: {cv_inc_010:.1f}%")

    # Alternative normalization at k=0.9 (mid-BZ, away from kR crossover)
    # k=0.3 has kR=1.5 < 2 (geometric regime, possible outlier from file 43)
    i_mid = 3  # k=0.9
    Rm_coh_030 = R_coh_030 / R_coh_030[i_mid]
    Rm_coh_010 = R_coh_010 / R_coh_010[i_mid]
    Rm_inc_030 = R_inc_030 / R_inc_030[i_mid]
    Rm_inc_010 = R_inc_010 / R_inc_010[i_mid]

    cv_coh_030m = np.std(Rm_coh_030) / np.mean(Rm_coh_030) * 100
    cv_coh_010m = np.std(Rm_coh_010) / np.mean(Rm_coh_010) * 100
    cv_inc_030m = np.std(Rm_inc_030) / np.mean(Rm_inc_030) * 100
    cv_inc_010m = np.std(Rm_inc_010) / np.mean(Rm_inc_010) * 100

    print(f"\n  CV (shape mismatch), normalized at k=0.9:")
    print(f"    Coherent Born  → α=0.30: {cv_coh_030m:.1f}%,  α=0.10: {cv_coh_010m:.1f}%")
    print(f"    Incoherent Born→ α=0.30: {cv_inc_030m:.1f}%,  α=0.10: {cv_inc_010m:.1f}%")
    print(f"  (k=0.3 normalization: coh={cv_coh_030:.1f}%, inc={cv_inc_030:.1f}%"
          f" — if very different, k=0.3 is outlier)")

    # ═══════════════════════════════════════════════════════════════════
    # Non-Born enhancement: how σ_FDTD deviates from coherent Born
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("Non-Born enhancement R(k) = σ_FDTD / (I_tr/sin²k)")
    print("  R grows with k → non-Born lifts high-k relative to low-k")
    print("=" * 75)
    print(f"  {'k':>5s}  {'R(0.30)':>10s}  {'R(0.10)':>10s}  {'030/R[0]':>9s}  {'010/R[0]':>9s}")
    for j, k0 in enumerate(k_vals):
        print(f"  {k0:5.1f}  {R_coh_030[j]:10.5f}  {R_coh_010[j]:10.6f}"
              f"  {Rn_coh_030[j]:9.4f}  {Rn_coh_010[j]:9.4f}")
    print(f"\n  R(0.30) grows {Rn_coh_030[-1]:.1f}× from k=0.3 to k=1.5")
    print(f"  R(0.10) grows {Rn_coh_010[-1]:.1f}× from k=0.3 to k=1.5")
    print(f"  Born drops {sigma_coh_shape[0]/sigma_coh_shape[-1]:.0f}× over same range")
    print(f"  FDTD(0.30) drops {sigma_030[0]/sigma_030[-1]:.1f}×")
    print(f"  FDTD(0.10) drops {sigma_010[0]/sigma_010[-1]:.1f}×")

    # ═══════════════════════════════════════════════════════════════════
    # α-enhancement ratio: σ(0.30) / σ(0.10)
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("α-enhancement: σ(0.30) / σ(0.10)")
    print("  If both α follow same spectral shape, ratio = C(0.30)/C(0.10) = const")
    print("  If ratio varies with k → spectral shapes differ between α")
    print("=" * 75)

    cm1_030 = np.cos(2 * np.pi * 0.30) - 1  # = -1.309
    cm1_010 = np.cos(2 * np.pi * 0.10) - 1  # = -0.191
    born_ratio_pred = (cm1_030 / cm1_010)**2  # Born: σ ∝ cm1²
    print(f"  Born prediction: (cm1_030/cm1_010)² = ({cm1_030:.3f}/{cm1_010:.3f})²"
          f" = {born_ratio_pred:.1f}")

    ratio_alpha = sigma_030 / sigma_010
    print(f"\n  {'k':>5s}  {'030/010':>8s}  {'normalized':>10s}")
    for j, k0 in enumerate(k_vals):
        print(f"  {k0:5.1f}  {ratio_alpha[j]:8.2f}  {ratio_alpha[j]/ratio_alpha[0]:10.4f}")
    cv_ratio = np.std(ratio_alpha) / np.mean(ratio_alpha) * 100
    print(f"  CV = {cv_ratio:.1f}%  range: {ratio_alpha.min():.1f} to {ratio_alpha.max():.1f}")
    print(f"  Mean ratio: {np.mean(ratio_alpha):.1f}  (Born pred: {born_ratio_pred:.1f})")
    print(f"  Non-Born enhancement factor: {np.mean(ratio_alpha)/born_ratio_pred:.2f}×"
          f" (>1 = non-Born enhancement at α=0.30 is larger)")

    # ═══════════════════════════════════════════════════════════════════
    # Integrand comparison
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("sin²(k)·σ_tr (integrand)")
    print("=" * 75)
    int_030 = np.sin(k_vals)**2 * sigma_030
    int_010 = np.sin(k_vals)**2 * sigma_010
    int_born = I_tr  # sin²(k) × I_tr/sin²(k) = I_tr
    print(f"  {'k':>5s}  {'FDTD 030':>10s}  {'FDTD 010':>10s}  {'Born coh':>12s}")
    for j, k0 in enumerate(k_vals):
        print(f"  {k0:5.1f}  {int_030[j]:10.2f}  {int_010[j]:10.4f}  {int_born[j]:12.1f}")
    cv_030 = np.std(int_030) / np.mean(int_030) * 100
    cv_010 = np.std(int_010) / np.mean(int_010) * 100
    cv_born = np.std(int_born) / np.mean(int_born) * 100
    print(f"  CV:   {cv_030:.1f}%       {cv_010:.1f}%        {cv_born:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print()
    print("  Shape match (CV of normalized ratio σ_FDTD / σ_model):")
    print()
    print(f"    {'Model':>20s}  {'α=0.30':>8s}  {'α=0.10':>8s}")
    print(f"    {'Coherent Born':>20s}  {cv_coh_030:7.1f}%  {cv_coh_010:7.1f}%")
    print(f"    {'Incoherent Born':>20s}  {cv_inc_030:7.1f}%  {cv_inc_010:7.1f}%")
    print(f"    {'(integrand CV)':>20s}  {cv_030:7.1f}%  {cv_010:7.1f}%")
    print()
    print("  Paradox:")
    print("    Scattering is NOT incoherent (file 46: ±40% interference)")
    print("    but FDTD σ(α=0.30) follows incoherent Born SHAPE (CV=2.7%)")
    print()
    print("  Non-Born enhancement grows with k:")
    print(f"    R(0.30, k=1.5) / R(0.30, k=0.3) = {Rn_coh_030[-1]:.1f}×")
    print(f"    R(0.10, k=1.5) / R(0.10, k=0.3) = {Rn_coh_010[-1]:.1f}×")
    print(f"    Born shape drops {sigma_coh_shape[0]/sigma_coh_shape[-1]:.0f}× over k=0.3→1.5")
    print()
    if cv_inc_030 < 5:
        print("  At α=0.30: non-Born correction compensates coherent Born falloff")
        print("  ALMOST EXACTLY, producing incoherent-like shape (CV=%.1f%%)" % cv_inc_030)
    if cv_inc_010 > 10:
        print("  At α=0.10: σ does NOT follow incoherent Born shape (CV=%.1f%%)" % cv_inc_010)
        print("    Shape is k-dependent in a way neither model captures alone.")
        print("    → compensation mechanism is α-dependent, not purely geometric")
    elif cv_inc_010 < 5:
        print("  At α=0.10: σ ALSO follows incoherent Born shape (CV=%.1f%%)" % cv_inc_010)
        print("    → same paradox exists at weak coupling")
