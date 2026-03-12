# Tests Map

## Test Files (192 tests, ~13s + 6 min FDTD)

### test_1_born_perbond.py (21 tests)

Per-bond Born mechanism: sigma_bond = C0 * V / v_g^2.

- **TestMonopoleDipole** -- Z_mono approximately constant (CV < 7%), Z_dipo grows 22x, Z_mono + Z_dipo = 16pi, limits at k→0
- **TestBornShape** -- Born formula C0·V/v_g² exact at α=0.25; Born underestimates FDTD by 2.3x (MS enhancement); FDTD/Born shape match for sigma_tot, sigma_xx, sigma_xy (CV thresholds)
- **TestAlphaThreshold** -- |cm1| = |s_phi| at alpha = 0.25, V(k) const above threshold, varies below; CV(V) asymmetric: steep drop below alpha=0.25, gentle rise above (< 6%)
- **TestBornNormalization** -- 1/v_g^2 varies less than 1/sin^2(k)
- **TestDiagonalOffdiagDecomposition** -- cm1 dominates at alpha>=0.30, s_phi dominates at alpha=0.10, crossover at 0.25, diagonal is displacement coupling (V_diag const), offdiag vertex varies strongly

### test_2_born_disk.py (11 tests)

Born exponent -5/2 for filled disk of gauged bonds.

- **TestBornExponent** -- Angular grid converged (N=50 vs 100, R=3,5); coherent limit: discrete N_eff → N² and continuum N_eff → (πR²)² at k→0; Born N_eff exponent at R=5 and across R=3,5,7,9
- **TestBornExponentLargeR** -- Convergence to -5/2 at R=200 using (delta,eta) grid (within 1%)
- **TestTransportDecomposition** -- Total exponent -3/2 (cone area), transport weight adds -1
- **TestUniversality** -- Random disk, continuum Airy, square disk all give similar exponent

### test_3_multiple_scattering.py (31 tests)

Multiple scattering correction: Born -5/2 shifts to MS -2.0.

- **TestMSExponent** -- MS exponent in [-2.15, -1.85] at R=5,7,9; R=3 wider tolerance; MS captures 75-110% of FDTD correction
- **TestIntegrandFlatness** -- Born integrand CV > 30%, MS integrand CV < 25%
- **TestShiftLinearity** -- Exponent shift proportional to |V|, C in [0.20, 0.45]
- **TestRandomDisk** -- Random disk gives same enhancement as lattice disk
- **TestPhaseRemoval** -- Static 1/r gives larger shift than physical exp(ikr)/r; phase reduces cooperation
- **TestBornSeriesConvergence** -- 1st Born correction captures < 50% of shift; series oscillates, needs full resummation
- **TestShiftDecomposition** -- Off-diagonal 1/r gives > 3x the shift of G_00-only; G_00 fraction < 20%
- **TestSingleModeAnalysis** -- |lambda_eff| decreases with k (power law ~k^{-0.77}); near-field (r<=3) dominates 60%+
- **TestCUVCutoff** -- C grows monotonically with density: 1× < 2× < 4× (UV-cutoff dependence)
- **TestSeedIndependence** -- Enhancement exponent CV < 15% over 5 random seeds; all seeds give positive enhancement
- **TestPositionalNoise** -- Enhancement exponent changes < 15% with ±0.1a noise; ±0.2a still positive
- **TestPhaseRegularization** -- physical G converges at r_cut=3 (|Δp/p|<5%); static G does NOT converge (+29% at r_cut=5); short-range phases are constructive
- **TestPropagationRange** -- r_cut=3 gives exponent within 20% of full range; spread < 0.15 (oscillatory convergence)
- **TestEnergyConservation** -- sigma_tot >= sigma_tr at all k; ratio grows with k (forward peaking)
- **TestForwardDecoherence** -- MS suppresses at low k (enh=0.61 < 1, not cooperative enhancement); tilt upward 0.61→1.18 (positive shift); forward coherence broken (|Σ Tb|²/|Σ Vb|²=0.54); suppression monotonic in R (0.61→0.54→0.47)

### test_4_assembly.py (14 tests)

Assembly: Born -5/2 + MS +1/2 produces flat integrand sin^2(k) * sigma_ring.

- **TestAlgebraicCancellation** -- sin^2(k) = 4 sin^2(k/2) cos^2(k/2), v_g^2 = c_lat^2 cos^2(k/2), cancellation leaves sin^2(k/2)
- **TestNeffBalance** -- sin^2(k/2) * N_eff CV < 8% at R=5,7,9; improves with R; R=3 too small (~12%)
- **TestFlatIntegrand** -- sin^2(k) * sigma_ring CV < 10%, Born vertex V(k) const, chain decomposition matches integrand
- **TestResidualCV** -- sinc^2(k/2) deviation ~6%, total residual 7-10% (not zero)
- **TestNeffStructure** -- sin^2(k/2) better than k^2 (margin > 0.5%), exponent near -2.0, sin^2(k/2)/k^2 approx 1/4, N_eff grows with R, σ_tr ~ R^p with mean p=1.63 (sub-geometric, between 1.5 and 1.9)

### test_5_coupling_requirements.py (12 tests)

Coupling requirements: displacement coupling + strong alpha + polarization independence.

- **TestBornNotFlat** -- Born integrand CV > 30% (~19x variation, needs MS correction)
- **TestWeakCouplingNotFlat** -- alpha=0.05 gives V(k) CV > 25%; alpha=0.10 integrand CV > 15%
- **TestAlphaThresholdRequired** -- V(k) CV decreases with alpha to 0.25, stays < 6% above; alpha=0.50 limit (R=-I, pure monopole) flatter than 0.10; alpha=0.30 flatter than alpha=0.10
- **TestStrainCouplingZero** -- (Documentary) Plane wave in +x: Delta_u = 0 on z-bond (geometric null), strain force = 0 for any dK, displacement coupling K1*cm1*u nonzero (contrast)
- **TestPolarizationIndependence** -- (R-I)(R^T-I) = 2(1-cos(2pi*alpha))*I; polarization-independent for physical (linear, circular) and random u; uz decoupled (N_pol=2)

### test_6_mechanism_elimination.py (25 tests)

Mechanism elimination: flatness is NOT from single-bond T-matrix or resonance or xx/xy compensation.

- **TestScalarTMatrixFails** -- |DK * G_anti| < 0.25 at all k (Born regime); scalar T-matrix correction < 25%
- **TestVectorialTMatrixFails** -- DK eigenvalues K1*(cm1 +/- i*s_phi) incl. α=0.50 degenerate limit, vectorial stronger than scalar by ~1.24x, but still Born (|eig*G| < 0.5)
- **TestNotResonance** -- |lambda_max| of VG in range 0.1-0.7 at all k; no resonance at R≤9 (0.861 at R=9)
- **TestNoCompensation** -- sigma_xy grows > 40x; sigma_xx varies < 2x; sigma_xy/sigma_xx < 1% at low k
- **TestGeometryDependence** -- Disk shift > line > annulus; R=9 disk >> annulus (interior matters); single bond no shift; disk grows with R; line grows slowly (spread < 0.15)
- **TestEigenvalueNeffFlat** -- Tr(AA†) slope ≈ 0 (NOT -2) at all R; Tr(AA†) ≈ 1.4N < 2N (Born regime). k^{-2} is emergent, not structural in VG.

### test_7_null_gauging.py (13 tests)

Null test: flat integrand is NN-specific, NNN gauging and AB formula fail.

- **TestNNNNotFlat** -- NNN integrand CV > 20%; NN flatter than NNN
- **TestNNNHigherThanNN** -- sigma_NNN > sigma_NN at all k
- **TestKappaNNvsNNN** -- kappa_NNN > kappa_NN; kappa monotonic in alpha; α=0.30 k≤1.5: κ_NN<1 but κ_NNN>1 (NNN overshoots)
- **TestKappaFromIntegral** -- κ_NN from trapezoidal ∫sin²(k)σ_tr dk matches stored kappa_table (< 3% at all α)
- **TestABFails** -- AB sigma ~ 1/k gives integrand CV > 20%; AB amplitude 6.5× below FDTD; AB shape does not match FDTD
- **TestNNPhaseArgument** -- (Documentary) NN z-bond dx=0: phase exp(ik*0)=1 at all k; NNN dx=1: phase varies (Re < 0.1 at k=1.5); NN vertex k-independent

### test_8_fdtd_convergence.py (11 tests, ~6 min)

FDTD convergence: box size L=80, L=100, L=120. All tests run actual FDTD — no hardcoded data.

- **TestBoxSizeConvergence** -- L=100 vs L=120 within 5% (converged); L=80 overestimates k=0.3 by 44% (PML too close); L=80 bias < 15% at k>=0.9; integrand CV < 15% at L=100; CV improves from L=80 to L=100; L=100 matches stored sigma_ring[5]
- **TestPMLConvergence** -- DW=15 vs DW=20 within 5%; all DW spread < 2% (stable); k=0.3 excluded (tested via box size)
- **TestWavepacketBandwidth** -- sx=6/8/12 within 5% at k=0.9,1.5 (bandwidth-independent)

### test_9_forward_mechanism.py (42 tests)

Forward decoherence mechanism: microscopic decomposition of how MS shifts Born -5/2 toward -2.

- **TestAngularUniformity** (F2) -- MS/Born ratio uniform in angle: mean=0.537, CV=4.5% at k=0.3. Max < 1 everywhere. k=1.5: CV=11.7% (low-k specific). Transport/total ratio diff=2.8% (uniform → weight-independent)
- **TestTMatrixDecomposition** (F3) -- |T_jj/V|²=1.347 (diagonal enhances); cross=-137.0 (destructive); |cross|/diagonal=0.73. α scaling: 2.8% (α=0.05) vs 73% (α=0.30), 26× ratio
- **TestTransportShiftRIndependence** (F4) -- Shift: +0.41 (R=3), +0.45 (R=5), +0.40 (R=7), +0.41 (R=9). CV=4.2%. Random disk: shift=0.27 (CV=9.4%). CV stable at 15 k-points (12.6%). Shift ∝ |V| (α=0.40 > 2× α=0.20)
- **TestGeometryDependence** (F7) -- C_disk=0.270 > C_annulus=0.121 > C_line=0.075. Disk/line ratio = 3.6×
- **TestCFullVsCStatic** (F8) -- C_full < C_static at all R (ratio 1.44–1.56); physical G converges at r_cut=3, static G diverges
- **TestMeanFieldCoupling** (F9) -- |V·⟨ΣG⟩| = 0.667 (k=0.3) → 0.157 (k=1.5), power law p = -0.94. Spectral radius < 1 at all k
- **TestPowerDecomposition** (F10) -- cross/born = 2 × diag_amp × offdiag_amp × cos(Φ). Exponent sum: +0.01+(-0.48)+(-0.16) = -0.63 ✓. Crossover k (enh=1): 0.61 (R=3) → 1.46 (R=9)
- **TestBornSeriesAlternation** (F11) -- |G_ij|² k-independent. cancel_ratio: 0.37 (k=0.3) vs 0.93 (k=1.5). VG coherence: 0.83 → 0.20. Eigenvalue |λ| CV: 41% (k=0.3, clustered) → 24% (k=1.5, spread)
- **TestCosPhiConvergence** (F12) -- |cos(Φ)| exponent: -0.35 (R=3) → -0.05 (R=9) → 0
- **TestCross12SignChange** (F13) -- cross_12 = 2Re⟨VG*·VG²⟩: -11% at k=0.3, +79% at k=1.5. Sign change at k≈0.35. Path excess phases: frac(k_eff·pe > π) = 7.5% (k=0.3) → 68% (k=1.5)

### test_10_forward_p2.py (12 tests)

Forward decoherence phase 2: scaling, consistency, eigenvalue structure, κ validation.

- **TestCrossoverScaling** (B1) -- Crossover k (enh=1) ~ R^0.59. k*=0.69 (R=3) → 1.56 (R=12). Monotonic in R
- **TestEigenvalueDominance** (B4) -- |λ_max|/|λ_second| = 2.52 at k=0.3 (single dominant mode), 1.01 at k=1.5 (no dominant mode)
- **TestShiftMethodConsistency** (A3+A4) -- Shifts from ms.py (no F²) and test_9 style (with F²) agree: +0.480 vs +0.474 (1.2% diff). F² invariant. Born exponent near -5/2 (links to test_2)
- **TestPathExcessConcentration** (B3) -- 1/r-weighted median pe = 2.0 (vs unweighted 4.0). Short-range (pe<3) = 62.5% of 1/r³ weight (vs 39.5% by count)
- **TestIntegrandCVMonotonic** (A2) -- MS integrand CV monotonically decreases with α: 39.5% (0.20) → 26.9% (0.40). No minimum at α=0.25
- **TestKappaConsistency** (A1) -- MS/FDTD gap (~15×) is per-bond normalization, not collective. N_eff_MS/N_eff_FDTD = 1.05 (mean). Ring/bond gap ratio ≈ 1.0
- **TestRScaling** (B6) -- Born σ_tr ~ R^{3/2} at fixed k (p ∈ [1.3, 1.7]). MS reduces R-scaling (p_ms < p_born): forward decoherence strengthens with R

## helpers/

- **config.py** -- Spring constants (K1, K2), lattice sound speed, default alpha/R, FDTD box params, k-grids, V_ref
- **born.py** -- Peierls coupling, V_eff, Z_monopole, Z_dipole, sigma_bond_born
- **geometry.py** -- Bond geometries: disk_bonds, ring_bonds, annulus_bonds, random_disk, line_bonds
- **lattice.py** -- Dispersion relation, group velocity, k_eff, 3D BZ grid (omega_k2), G_00 self-energy
- **ms.py** -- Multiple scattering: build_G_matrix, build_VG, T_matrix, sigma_tr_ms, eigenvalues_VG, make_dOmega
- **stats.py** -- cv (coefficient of variation), log_log_slope (power law exponent), N_eff_from_sigma

## data/

- **sigma_bond.py** -- FDTD sigma_tr(k) for single z-bond at alpha=0.30: total, sigma_xx, sigma_xy
- **sigma_ring.py** -- FDTD sigma_tr(k) for NN-gauged ring at various R (3,5,7,9) and various alpha (0.10-0.50)
- **kappa_table.py** -- kappa(alpha) tables for NN and NNN gauging at multiple k-cutoffs

## Import-time self-consistency checks

All helpers/ and data/ files contain assert statements that run at import time (zero cost, fail immediately if data is corrupted):

- **config.py** -- K1=2K2, c_lat^2=K1+4K2, grid lengths
- **lattice.py** -- omega^2(0)=0, v_g(0)=c_lat, omega^2=4c^2 sin^2(k/2)
- **born.py** -- Z_mono+Z_dipo=16pi, V_eff(alpha_ref)=V_ref, |cm1|=|s_phi| at alpha=0.25
- **geometry.py** -- disk(0)=1pt, disk(1)=5pts, line symmetry, ring subset of disk
- **stats.py** -- CV(const)=0, CV([1,3])=50%, slope(x^2)=2
- **sigma_bond.py** -- sigma_xx+sigma_xy=sigma_bond, positivity, sigma_xx>sigma_xy
- **sigma_ring.py** -- sigma_ring[5]=sigma_alpha_nn[0.30], monotonic in R and alpha
- **kappa_table.py** -- kappa_nn monotonic in alpha, kappa_nnn>kappa_nn
