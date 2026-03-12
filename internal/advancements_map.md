# Advancements Map — st_vortex_transport

**Date:** Mar 2026
**Tests:** 10 files, 192 tests (~13s analytic + 6 min FDTD)
**Paper:** PRL submitted (Mar 2026), Temporary ID es2026mar11_956

---

## A. Per-Bond Born Mechanism (test_1, 21 tests)

1. **Z_mono approximately constant** (CV < 7%), Z_dipo grows 22x across BZ
2. **Z_mono + Z_dipo = 16pi** (exact sum rule)
3. **Born formula exact at α=0.25**: σ_bond = C0·V/v_g²
4. **Born underestimates FDTD by 2.3x** (MS enhancement)
5. **FDTD/Born shape match** for σ_tot, σ_xx, σ_xy (CV thresholds)
6. **|cm1| = |s_phi| at α=0.25** (monopole-dipole crossover)
7. **V(k) const above threshold** (CV < 6% for α ≥ 0.25)
8. **CV(V) asymmetric**: steep drop below α=0.25, gentle rise above
9. **1/v_g² varies less than 1/sin²(k)** (lattice normalization)
10. **cm1 dominates at α≥0.30**, s_phi at α=0.10, crossover at 0.25
11. **Diagonal = displacement coupling** (V_diag const); offdiag vertex varies strongly

## B. Born Disk Exponent (test_2, 11 tests)

12. **Angular grid converged** (N=50 vs 100, R=3,5)
13. **Coherent limit**: discrete N_eff → N², continuum N_eff → (πR²)² at k→0
14. **Born N_eff exponent -5/2** at R=5 and across R=3,5,7,9
15. **Convergence to -5/2 at R=200** using (δ,η) grid (within 1%)
16. **Total exponent -3/2** (cone area), transport weight adds -1
17. **Universality**: random disk, continuum Airy, square disk all give same exponent

## C. Multiple Scattering (test_3, 31 tests)

18. **MS exponent in [-2.15, -1.85]** at R=5,7,9
19. **MS captures 75-110% of FDTD correction**
20. **Born integrand CV > 30%**, MS integrand CV < 25%
21. **Exponent shift proportional to |V|**, C in [0.20, 0.45]
22. **Random disk gives same enhancement** as lattice disk
23. **Static 1/r gives larger shift** than physical exp(ikr)/r; phase reduces cooperation
24. **1st Born correction captures < 50%** of shift; series oscillates, needs full resummation
25. **Off-diagonal 1/r gives > 3x the shift** of G_00-only; G_00 fraction < 20%
26. **|lambda_eff| decreases with k** (power law ~k^{-0.77}); near-field (r≤3) dominates 60%+
27. **C grows monotonically with density**: UV-cutoff dependence
28. **Enhancement exponent CV < 15%** over 5 random seeds
29. **Enhancement robust to ±0.1a positional noise**; ±0.2a still positive
30. **Physical G converges at r_cut=3**; static G does NOT converge (+29% at r_cut=5)
31. **r_cut=3 gives exponent within 20% of full range** (oscillatory convergence)
32. **σ_tot ≥ σ_tr at all k**; ratio grows with k (forward peaking)
33. **MS suppresses at low k** (enh=0.61 < 1, not cooperative enhancement)
34. **Forward coherence broken**: |Σ Tb|²/|Σ Vb|² = 0.54
35. **Suppression monotonic in R**: 0.61 → 0.54 → 0.47 at R=5,7,9

## D. Assembly (test_4, 14 tests)

36. **Algebraic cancellation**: sin²(k) cancels cos²(k/2) from 1/v_g², leaves sin²(k/2)
37. **sin²(k/2) · N_eff CV < 8%** at R=5,7,9; improves with R
38. **sin²(k) · σ_ring CV < 10%** (flat integrand)
39. **Residual CV ~6%** (sinc² deviation); total 7-10% (not zero)
40. **sin²(k/2) better than k²** (margin > 0.5%); exponent near -2.0
41. **σ_tr ~ R^p with p=1.63** (sub-geometric, between 1.5 and 1.9)

## E. Coupling Requirements (test_5, 12 tests)

42. **Born integrand CV > 30%** without MS (~19x variation)
43. **α=0.05 gives V(k) CV > 25%** (weak coupling not flat)
44. **V(k) CV decreases with α** to 0.25, stays < 6% above
45. **Strain coupling = 0 on z-bond** (geometric null Δu=0); displacement coupling nonzero
46. **(R-I)(R^T-I) = 2(1-cos(2πα))·I**: polarization-independent; N_pol=2

## F. Mechanism Elimination (test_6, 25 tests)

47. **Scalar T-matrix correction < 25%** (Born regime, |DK·G_anti| < 0.25)
48. **Vectorial T-matrix stronger by ~1.24x** but still Born (|eig·G| < 0.5)
49. **No resonance**: |λ_max| in range 0.1-0.7 at all k; 0.861 at R=9
50. **σ_xy grows > 40x**; σ_xx varies < 2x; no xx/xy compensation
51. **Disk shift > line > annulus**; interior matters (R=9 disk >> annulus)
52. **Tr(AA†) slope ≈ 0** (NOT -2); k^{-2} is emergent, not structural in VG

## G. Null Gauging (test_7, 13 tests)

53. **NNN integrand CV > 20%**; NN flatter than NNN
54. **σ_NNN > σ_NN at all k**
55. **κ_NNN > κ_NN**; κ monotonic in α; NNN overshoots κ=1 at α=0.30
56. **κ_NN from trapezoidal integral matches kappa_table** (< 3% at all α)
57. **AB formula fails**: CV > 20%, amplitude 6.5× below FDTD, wrong shape
58. **NN z-bond phase = 1 at all k** (Δx=0); NNN phase varies

## H. FDTD Convergence (test_8, 11 tests, ~6 min)

59. **L=100 vs L=120 within 5%** (converged)
60. **L=80 overestimates k=0.3 by 44%** (PML too close)
61. **Integrand CV < 15% at L=100**; improves from L=80
62. **DW=15 vs DW=20 within 5%**; all DW spread < 2%
63. **sx=6/8/12 within 5%** at k=0.9,1.5 (bandwidth-independent)

## I. Forward Decoherence Mechanism (test_9, 42 tests)

64. **MS/Born ratio uniform in angle**: mean=0.537, CV=4.5% at k=0.3
65. **|T_jj/V|²=1.347** (diagonal enhances); cross=-137.0 (destructive); |cross|/diagonal=0.73
66. **Cross-term scales 26× from α=0.05 to 0.30** (why weak coupling shows no decoherence)
67. **Shift R-independent**: +0.41,+0.45,+0.40,+0.41 (CV=4.2%)
68. **Shift ∝ |V|**: C=0.334±0.012 (CV=3.7%)
69. **C_disk > C_annulus > C_line** (interior bonds matter; disk/line ratio=3.6×)
70. **C_full < C_static** at all R (ratio 1.44-1.56); physical G converges, static diverges
71. **Mean-field coupling |V·⟨ΣG⟩|**: power law p=-0.94; spectral radius < 1
72. **Power decomposition**: cross/born = 2 × diag × offdiag × cos(Φ); exponent sum verified
73. **Crossover k***: 0.61 (R=3) → 1.46 (R=9)
74. **|G_ij|² k-independent**; cancel_ratio 0.37→0.93; VG coherence 0.83→0.20
75. **|cos(Φ)| exponent**: -0.35 (R=3) → -0.05 (R=9) → 0
76. **cross_12 sign change at k≈0.35**: -11% at k=0.3, +79% at k=1.5

## J. Forward Decoherence Phase 2 (test_10, 12 tests)

77. **Crossover k* ~ R^{0.59}**: 0.69 (R=3) → 1.56 (R=12); monotonic
78. **Eigenvalue dominance**: |λ₁|/|λ₂| = 2.52 at k=0.3 (single mode), 1.01 at k=1.5 (uniform)
79. **Shift methods agree**: +0.480 vs +0.474 (1.2% diff); F² invariant
80. **Born exponent near -5/2** (cross-check with test_2)
81. **1/r-weighted median pe=2.0** (vs unweighted 4.0); 62.5% of weight at pe<3
82. **MS integrand CV monotonically decreases** with α: 39.5%→26.9% (no minimum at 0.25)
83. **N_eff_MS/N_eff_FDTD = 1.05** (mean, CV=14%); MS/FDTD gap is per-bond normalization
84. **Born σ_tr ~ R^{3/2}**; MS reduces R-scaling (p_ms < p_born)

---

## Key Results (paper-level)

| # | Result | Section |
|---|--------|---------|
| 38 | **Flat Boltzmann integrand**: sin²(k)·σ_tr CV = 5.6% at α=0.30 | Results |
| 15 | **Born exponent -5/2** verified at R=200 (within 0.4%) | Mechanism |
| 18 | **MS shifts exponent to ≈-2.0** (+0.45 shift) | Mechanism |
| 34 | **Forward coherence broken**: ratio 0.54 at k=0.3 | Mechanism |
| 36 | **Algebraic cancellation**: sin²(k)/v_g² = 4sin²(k/2) | Mechanism |
| 56 | **κ_NN from integral**: 0.844 at α=0.30 (k≤1.5) | Results |
| 83 | **N_eff mechanism validated**: MS/FDTD ratio 1.05 | SM §9.6 |

---

*84 numbered results from 192 tests across 10 files.*
