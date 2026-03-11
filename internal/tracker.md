# Test Gaps — Referee-Proofing Tracker

**Date:** Mar 2026
**Source:** Reviewer strategic analysis of `tests/` (115 tests, 8 files)
**Purpose:** Systematic gaps a referee could exploit. Each gap = one new test class.

---

## Summary

| # | Test | Type | Priority | Status |
|---|------|------|----------|--------|
| 1 | TestContinuumDiskLimit | analytic + MS | high | pending |
| 2 | TestDispersionPerturbation | analytic | medium | parked — cos² cancellation is algebraic identity |
| 3 | TestBoxSizeConvergence | FDTD | high | **done** — test_8 |
| 4 | TestPMLConvergence | FDTD | medium | **done** — test_8 |
| 5 | TestWavepacketBandwidth | FDTD | medium | **done** — test_8 |
| 6 | TestPropagationRange | MS | medium | **done** — test_3 |
| 7 | TestPositionalNoise | MS | high | **done** — test_3 |
| 8 | TestDimensionality | analytic | low | pending |
| 9 | TestEnergyConservation | MS | high | **done** — test_3 |
| 10 | TestSeedIndependence | MS | low | **done** — test_3 |
| 11 | TestEigenvalueNeffFlat | MS | — | **done** — test_6 (null test, reviewer suggestion disproved) |
| 12 | TestPhaseRegularization | MS | — | **done** — test_3 (from propagation range investigation) |

---

## Completed Tests

### 6. TestPropagationRange (test_3, 2 tests)

r_cut=2: p=0.323, r_cut=3: 0.352, r_cut=5: 0.360, r_cut=∞: 0.354.
Near-field (r≤3) gives 99% of full enhancement. Exponent converges fast.

### 7. TestPositionalNoise (test_3, 2 tests)

clean: p=0.354, ±0.1a: 0.353 (|Δp/p|=0.2%), ±0.2a: 0.350 (1.0%).
Lattice order is NOT essential for the mechanism.

### 9. TestEnergyConservation (test_3, 2 tests)

σ_tot/σ_tr: 1.66 (k=0.3) → 7.38 (k=1.5). All > 1.
Ratio grows with k (scattering becomes forward-peaked).

### 10. TestSeedIndependence (test_3, 2 tests)

p_enh: 0.343, 0.354, 0.357, 0.335, 0.320 over 5 seeds. CV=3.9%.
All seeds give positive enhancement.

### 12. TestPhaseRegularization (test_3, 3 tests)

Physical G(r) = exp(ikr)/(4πc²r) converges at r_cut ≈ 3 (|Δp/p| = 0.6%).
Static G(r) = 1/(4πc²r) does NOT converge: +29% growth from r_cut=3 to r_cut=5.
At short range (r_cut=2): physical > static (phases add constructively).
Phase oscillations self-regularize the Green function, cutting off distant contributions.

### 11. TestEigenvalueNeffFlat (test_6, 5 tests)

Reviewer proposed: extract N_eff from Tr(AA†) = Σ 1/|1-λ_i|², predict k^{-2}.
**Result:** slope ≈ 0.03 (constant), NOT -2. Tr(AA†) ≈ 1.4N (Born regime).
k^{-2} is emergent from coherent phase sums + transport weighting, not structural in VG.

---

## Pending Tests

### 1. TestContinuumDiskLimit

**Referee question:** "Does the exponent survive at large N?"

Dense random disk (N=200-500 bonds via Monte Carlo), compute σ_tr via MS T-matrix. Verify MS exponent converges toward -2.0 as N grows. Currently tested at R=3,5,7,9 (lattice disk). Gap: no continuum-limit test.

**Where:** test_2_born_disk.py or test_3_multiple_scattering.py
**Effort:** medium (large VG matrix, slow but doable)

### 2. TestDispersionPerturbation — PARKED

**Referee question:** "Is the flat integrand an artifact of simple cos dispersion?"

**Status:** Parked. The cos²(k/2) cancellation between sin²(k) and 1/v_g² is an algebraic identity for ω² = 4c²sin²(k/2), already tested in test_4 TestAlgebraicCancellation. Perturbing the dispersion breaks the identity by O(ε), which is expected, not insightful.

### 3. TestBoxSizeConvergence (test_8, 5 tests)

L=80, L=100, L=120 at k=[0.3, 0.9, 1.5]. All computed live (no hardcoded data).
L=100 vs L=120: Δ = 2.7%, 0.2%, 0.4% (converged).
L=80 vs L=100: k=0.3 overestimated by 44% (PML at 5 units from sphere), k≥0.9 by ~8%.
Integrand CV: L=80 = 10.0%, L=100 = 6.4%. Flatness improves with box size.

### 4. TestPMLConvergence

**Referee question:** "Are PML reflections contaminating your results?"

Vary PML layer thickness, check σ_tr stability.

**Where:** alongside TestBoxSizeConvergence
**Effort:** high (needs FDTD runs)

### 5. TestWavepacketBandwidth

**Referee question:** "Does spectral width matter?"

Run FDTD with 2-3 different wavepacket bandwidths. Check σ_tr(k) shape is bandwidth-independent.

**Where:** FDTD-dependent
**Effort:** high (FDTD runs)

### 8. TestDimensionality

**Referee question:** "Is -2 specific to disks or generic?"

Compare Born/MS exponents for line (1D), disk (2D), sphere (3D). Line should give different exponent. Currently test_6 tests geometry dependence qualitatively but not exponent systematics.

**Where:** test_6_mechanism_elimination.py
**Effort:** medium (sphere geometry not implemented yet)

---

## Notes

- Tests 3, 4, 5 require FDTD infrastructure. If old investigation data exists (files in old_tests/), can use stored values instead of re-running.
- Test 1 uses existing MS machinery but with large N (slow). Doable.
- Test 8 needs new geometry (sphere_bonds) in helpers/geometry.py.
- Test 2 parked — algebraic identity, not a meaningful robustness test.
