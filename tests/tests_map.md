# Tests Map

## Test Files (127 tests, ~5s + 6 min FDTD)

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

### test_3_multiple_scattering.py (27 tests)

Multiple scattering correction: Born -5/2 shifts to MS -2.0.

- **TestMSExponent** -- MS exponent in [-2.15, -1.85] at R=5,7,9; R=3 wider tolerance; MS captures 75-110% of FDTD correction
- **TestIntegrandFlatness** -- Born integrand CV > 30%, MS integrand CV < 25%
- **TestShiftLinearity** -- Exponent shift proportional to |V|, C in [0.20, 0.45]
- **TestRandomDisk** -- Random disk gives same enhancement as lattice disk
- **TestPhaseRemoval** -- Static 1/r gives larger shift than physical exp(ikr)/r; phase reduces cooperation
- **TestBornSeriesConvergence** -- 1st Born correction captures < 50% of shift; series oscillates, needs full resummation
- **TestShiftDecomposition** -- Off-diagonal 1/r gives > 3x the shift of G_00-only; G_00 fraction < 20%
- **TestSingleModeAnalysis** -- |lambda_eff| decreases with k (power law ~k^{-0.77}); near-field (r<=3) dominates 60%+
- **TestCUVCutoff** -- C grows with scatterer density (UV-cutoff dependence)
- **TestSeedIndependence** -- Enhancement exponent CV < 15% over 5 random seeds; all seeds give positive enhancement
- **TestPositionalNoise** -- Enhancement exponent changes < 15% with ±0.1a noise; ±0.2a still positive
- **TestPhaseRegularization** -- physical G converges at r_cut=3 (|Δp/p|<5%); static G does NOT converge (+29% at r_cut=5); short-range phases are constructive
- **TestPropagationRange** -- r_cut=3 gives exponent within 20% of full range; spread < 0.15 (oscillatory convergence)
- **TestEnergyConservation** -- sigma_tot >= sigma_tr at all k; ratio grows with k (forward peaking)

### test_4_assembly.py (13 tests)

Assembly: Born -5/2 + MS +1/2 produces flat integrand sin^2(k) * sigma_ring.

- **TestAlgebraicCancellation** -- sin^2(k) = 4 sin^2(k/2) cos^2(k/2), v_g^2 = c_lat^2 cos^2(k/2), cancellation leaves sin^2(k/2)
- **TestNeffBalance** -- sin^2(k/2) * N_eff CV < 8% at R=5,7,9; improves with R; R=3 too small (~12%)
- **TestFlatIntegrand** -- sin^2(k) * sigma_ring CV < 10%, Born vertex V(k) const, chain decomposition matches integrand
- **TestResidualCV** -- sinc^2(k/2) deviation ~6%, total residual 7-10% (not zero)
- **TestNeffStructure** -- sin^2(k/2) better than k^2 (margin > 0.5%), exponent near -2.0, sin^2(k/2)/k^2 approx 1/4, N_eff grows with R

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

### test_7_null_gauging.py (12 tests)

Null test: flat integrand is NN-specific, NNN gauging and AB formula fail.

- **TestNNNNotFlat** -- NNN integrand CV > 20%; NN flatter than NNN
- **TestNNNHigherThanNN** -- sigma_NNN > sigma_NN at all k
- **TestKappaNNvsNNN** -- kappa_NNN > kappa_NN; kappa monotonic in alpha; α=0.30 k≤1.5: κ_NN<1 but κ_NNN>1 (NNN overshoots)
- **TestABFails** -- AB sigma ~ 1/k gives integrand CV > 20%; AB amplitude 6.5× below FDTD; AB shape does not match FDTD
- **TestNNPhaseArgument** -- (Documentary) NN z-bond dx=0: phase exp(ik*0)=1 at all k; NNN dx=1: phase varies (Re < 0.1 at k=1.5); NN vertex k-independent

### test_8_fdtd_convergence.py (6 tests, ~6 min)

FDTD convergence: box size L=80, L=100, L=120. All tests run actual FDTD — no hardcoded data.

- **TestBoxSizeConvergence** -- L=100 vs L=120 within 5% (converged); L=80 overestimates k=0.3 by 44% (PML too close); L=80 bias < 15% at k>=0.9; integrand CV < 15% at L=100; CV improves from L=80 to L=100; L=100 matches stored sigma_ring[5]
- **TestPMLConvergence** -- DW=15 vs DW=20 within 5%; all DW spread < 2% (stable); k=0.3 excluded (tested via box size)
- **TestWavepacketBandwidth** -- sx=6/8/12 within 5% at k=0.9,1.5 (bandwidth-independent)

## helpers/

- **config.py** -- Spring constants (K1, K2), lattice sound speed, default alpha/R, FDTD box params, k-grids, V_ref
- **born.py** -- Peierls coupling, V_eff, Z_monopole, Z_dipole, sigma_bond_born
- **geometry.py** -- Bond geometries: disk_bonds, ring_bonds, annulus_bonds, random_disk, line_bonds
- **lattice.py** -- Dispersion relation, group velocity, k_eff, 3D BZ grid (omega_k2), G_00 self-energy
- **ms.py** -- Multiple scattering: build_G_matrix, build_VG, T_matrix, sigma_tr_ms, eigenvalues_VG
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
