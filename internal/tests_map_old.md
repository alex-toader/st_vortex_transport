# W21: Tests Map

**62 test files** for σ_tr(k) computation on Peierls vortex ring.
Each file is self-contained: run instructions in header, results recorded after run.

---

## Infrastructure

| File | Tests |
|------|-------|
| elastic_3d.py | Scalar Laplacian (NN+NNN), PML damping, Verlet FDTD |
| scattering_3d.py | Gaussian wave packet, sphere sampling, σ_tr integration |
| gauge_3d.py | Peierls vortex ring (NN + optional NNN), Dirac disk geometry |
| parallel_fdtd.py | Multiprocessing wrapper for FDTD |

## A. Pipeline Validation (files 1-5, 8-9, 24, 33)

| File | What it verifies |
|------|-----------------|
| 1 | 3D dispersion ω²(k) matches analytic for NN+NNN on simple cubic |
| 2 | Lattice module: dispersion (5 dirs), Newton 3, PML absorption, FDTD energy |
| 3 | 2D AB pipeline: σ_tr matches analytic 2sin²(πα)/k at α=0.5 |
| 4 | gauge_3d: α=0 identity, disk bond count, R(2πα) matrix, α↔1−α |
| 5 | scattering_3d: packet propagation, sphere sampling, σ_tr convergence |
| 8 | uz decoupling: pure uz → σ_tr=0 exact; mixed uz+ux → uz unscattered |
| 9 | Reviewer checks: uz=0, y-incidence isotropy, elastic FFT, n_steps convergence |
| 24 | α=0 → σ_tr=0 exact at all k. Rules out pipeline artifacts at high k |
| 33 | α↔1−α symmetry (exact to 1e-15). Angular grid 13×24 vs 25×48 (diff < 2%) |

## B. σ_tr Spectrum & κ(α) (files 6-7, 10, 14-18, 22-23, 28)

| File | What it measures |
|------|-----------------|
| 6 | Diagnostics: σ_tr vs R scaling, polarization conversion (uy/ux), uz fraction |
| 7 | κ extraction from σ_tr(k) via trapezoidal integration |
| 10 | α scan: κ(α) at α=0.1-0.5, 6 k-pts |
| 14 | Gauss-Legendre quadrature eliminates tail extrapolation |
| 15 | Extended k: σ_tr at k=1.1, 1.3, 1.5 |
| 16 | α scan extended: κ(α) with 9 k-pts |
| 17 | Flat integrand discovery: sin²(k)·σ_tr ≈ const at α≥0.2 |
| 18 | R scan: σ_tr(k) at R=3,5,7,9. Source for R^{3/2} scaling |
| 22 | α rescan: κ(α) with 13 k-pts (full BZ, k=0.3-3.0) |
| 23 | NNN α scan: κ_NNN vs κ_NN at 4 α values, 13 k-pts |
| 28 | NNN fine α scan: κ_NNN at k≤0.9 and k≤1.5. κ=1 crossing |

## C. R-Scaling & Geometry (files 12-13, 18-21, 27)

| File | What it measures |
|------|-----------------|
| 12 | +z incidence (along ring axis): σ_tr ≈ 0 (holonomy inaccessible) |
| 13 | Direction scan: σ_tr(θ_inc) from +z to +x. In-plane isotropy 3% |
| 18 | R scan: σ_tr at R=3,5,7,9. R^{3/2} stationary phase fit |
| 19 | Fine k-grid (Δk=0.05): no oscillations → incoherent-like at R=3,5,7 |
| 20 | Large R=15 (L=160): σ_tr scaling extends to R=15, p≈1.6 |
| 21 | k>1.5: σ_tr rises at k≈1.7 (not power-law decay). Minimum confirmed |
| 27 | R^{3/2} factorization at α=0.05 vs 0.30. R-part universal, k-part α-dependent |

## D. Systematics & Robustness (files 25-26, 29-32)

| File | What it tests |
|------|--------------|
| 25 | Gauge invariance: Dirac surface shift ±2. Violation < 2.6% (NN), < 2.1% (NNN) |
| 26 | Born limit: CV(sin²·σ) at α=0.05, 0.10, 0.30. Flat only at α≥0.2 |
| 29 | K₁/K₂=1.5, 2.0, 3.0: flat integrand persists (CV 3-14%), κ=0.75-1.23 |
| 30 | Born R=1: α-exponent p→2.05 at kR=1.5 (Born limit). CV=37-77% (not flat) |
| 31 | 3D flux tube vs 2D AB: α=0 exact, α=0.30 within 10% |
| 32 | 2D monopole: Peierls/AB ratio=6-18× (not AB). 3D L=80 vs L=120 convergence |

## E. Coupling Mechanism (files 34-38, 40-42)

| File | What it tests | Key result |
|------|--------------|------------|
| 34 | Mass sphere (strain coupling, 1302 bonds) | CV=89-152%. NOT flat |
| 35 | Diagonal (cm1·I) vs off-diagonal (s_phi·J) on ring | Diag CV=8.8%, offdiag CV=23.7% |
| 36 | Decomposition at α=0.05-0.30 | Diag flat only at α≥0.30 |
| 37 | Displacement vs strain coupling on ring | Displacement CV=8.9%, strain=0 (null) |
| 38 | Ring orientation: xy, xz, yz planes | yz (axis along prop) CV=42.5%. Strain null |
| 40 | NNN selective: dx=0 vs dx=±1 bonds | Phase prediction DISPROVED |
| 41 | Non-Born ratio: σ(K=-0.5)/σ(K=-1.3) vs Born | 2.5× enhancement, k-independent |
| 42 | Displacement on sphere + half-ring arc | Sphere CV=35-46% (not flat). Arc CV=3.6% |

## F. Coherence & Interference (files 43-48)

| File | What it tests | Key result |
|------|--------------|------------|
| 43 | Born form factor |F(q)|² of Dirac disk | Coherent Born CV=147%. Incoherent Born CV=2.7% |
| 44 | Arc scaling: σ_tr vs N_bonds (angular arcs) | p_arc=0.79, sub-linear |
| 45 | Additivity: Q1-Q4 quarters at α=0.30 | ±40% interference. NOT incoherent |
| 46 | Additivity at α=0.10 (47× weaker coupling) | Same ±40%. Interference is GEOMETRIC |
| 47 | Shape: σ_FDTD vs coherent/incoherent Born | Incoherent shape CV=2.7% (α=0.30) despite coherent scattering |
| 48 | Cross-term: σ_diag + σ_cross from files 45-46 | cross/full = -0.45 (geometric, α-independent) |

## G. Per-Bond Decomposition (files 49-56)

| File | What it tests | Key result |
|------|--------------|------------|
| 49 | Single bond (R_loop=0): full Peierls at α=0.10, 0.30 | CV=96% (α=0.10), 71% (α=0.30). Per-bond non-Born, α-dependent |
| 50 | Single bond: diagonal-only (cm1·I, no s_phi) | CV=70% (α=0.10), 69% (α=0.30). Non-Born from displacement, α-independent |
| 51 | Identity test: σ_bond × I_tr vs Z_avg + functional form | σ_bond ≈ const(k) (CV=20%). Identity FAILS (CV=145%). N_eff drops 23× |
| 52 | Scalar T-matrix single bond (analytic) | |DK·G|≈0.11-0.19 ≪ 1 (Born). CV=106% vs FDTD 20%. SCALAR FAILS |
| 53 | Polarization decomposition: σ_xx, σ_xy separate | σ_xx≈const (CV≈62-66%), α-independent. Compensation FAILS |
| 54 | Vectorial 2×2 T-matrix single bond | |T_xx|²≈const (2% variation). CV=109% vs FDTD 14%. VECTORIAL FAILS |
| 55 | Born mechanism: monopole/dipole + v_g normalization | Per-bond Born CV=4.6%. Ring flat (CV=7.4%) from N_eff ∝ 1/sin²(k/2). CV(V) scan |
| 56 | N_eff structure: k,R dependence, Born vs FDTD | N_eff ∝ 1/k² (CV=2.3%). Model N²/(1+c·k²·R^q). Born (kR)^{-2.4} vs FDTD (kR)^{-2.0} |

## H. Forward Cone Analysis (files 57-59)

| File | What it tests | Key result |
|------|--------------|------------|
| 57 | Born N_eff exponent + correction characterization | Born -5/2, correction = 0.56×√ω (R-indep, CV=2.9%). Single-bond T-matrix wrong direction → collective |
| 58 | Multiple scattering mechanism confirmation | T=(I-VG)^{-1}V with continuum G(r). MS shift 91-104% of FDTD. Integrand CV: Born 35% → MS 10% → FDTD 7% |
| 59 | Eigenvalue analysis of VG (Parts A-G) | NOT resonance. Shift 80% from inter-bond 1/r (not G₀₀). Single-mode λ_eff captures 77-100%. C≈0.31 is lattice geometric constant |
| 60 | Born -5/2 derivation + gap closure (Parts A-F) | -5/2 = -3/2 (cone) + (-1) (transport). ANALYTIC. -2.0 NOT exact (crossing at α≈0.29). CV=10.7% from sinc² + non-power-law |
| 61 | Pair sum S(k) + geometry of +1/2 (Parts A-D) | Disk p_enh grows → +1/2 at R→∞. Line +0.12, annulus +0.11 (saturated). Near-field (r<3) = 66-77% |
| 62 | Final sanity checks (Parts A-C) | Eigenvalues smooth (no fractal). Random disk p_enh=0.33±0.016≈lattice 0.35. C set by UV cutoff a=1 (not lattice symmetry) |

## I. Zero-Compute Analysis (file 39)

| File | What it computes |
|------|-----------------|
| 39 | CV vs R (F11), phase prediction (F12), α-exponent Born mixing (F15), polarization independence (F16) |

---

## Claim → File Index

| Claim | Files |
|-------|-------|
| σ_tr(k) spectrum (principal result) | 17, 18, 22 |
| Flat integrand sin²(k)·σ_tr ≈ const | 17, 26, 29 |
| κ(α) = O(1) at α≈0.3 | 10, 16, 22, 23, 28 |
| R^{3/2} stationary phase | 18, 20, 27 |
| Not Born (σ∝α^{2.56}) | 26, 30, 41 |
| AB fails (k, α, R axes) | 31, 32 |
| Coherent with incoherent-like shape | 43, 45, 46, 47 |
| Flat = displacement coupling + ring topology | 34, 37, 42 |
| Diagonal (cm1) dominates at α≥0.25 | 35, 36 |
| Strain null mechanism | 37, 38 |
| NNN/NN: same holonomy, different CV | 23, 28, 40 |
| Per-bond non-Born (CV≈69-71%) | 49, 50 |
| σ_bond ≈ const(k), not algebraic identity | 51 |
| Scalar T-matrix Born at all α, fails for σ_bond | 52 |
| σ_xx (same-pol) ≈ const, compensation fails | 53 |
| Vectorial T-matrix Born, fails for σ_xx | 54 |
| Per-bond Born σ=C₀·V/v_g² (DERIVED). Ring N_eff ∝ 1/k² (EMPIRIC) | 55, 56 |
| N_eff = N²/(1+c·k²·R^q), q+p_R ≈ 2β | 56 |
| Born exponent -5/2 = -3/2 (cone) + (-1) (transport). ANALYTIC | 57, 60 |
| Non-Born correction = 0.56×√ω, R-independent, collective | 57 |
| Single-bond T-matrix wrong direction (eliminated) | 57 |
| MS T=(I-VG)⁻¹V explains +1/2 correction (91-104% of FDTD) | 58 |
| Integrand CV: Born 35% → MS 10% → FDTD 7% | 58 |
| NOT resonance (|λ_max|=0.3-0.6, far from 1) | 59 |
| Shift ≈ 0.31×|V| (linear, R-independent) | 59 |
| α-threshold derived: |V|=1.26 → α=0.29 | 59 |
| Shift 80% from inter-bond 1/r, only 9% from G₀₀ self-energy | 59 |
| Single-mode λ_eff captures 77-100% of shift (exact at R≥7) | 59 |
| |λ_eff| ~ k^{-0.77}: phase coherence at low k → shift | 59 |
| C ≈ 0.31 is lattice-specific (~80% from r ≤ 3) | 59 |
| Flat integrand is collective (ring) | 35, 44, 48, 49, 50, 51 |
| Interference is geometric (α-independent) | 45, 46, 48 |
| α=0 → σ_tr=0 exact | 24 |
| α↔1−α symmetry exact | 33 |
| K₁/K₂ robustness | 29 |
| Gauge violation < 2.6% | 25 |
| Angular grid converged (< 2%) | 33 |
| -2.0 NOT exact: crossing p_MS(α)=-2 at α≈0.29 | 60 |
| N_eff ∝ 1/k² is consequence of Born+MS, not independent | 60 |
| Residual CV=10.7%: sinc²(6%) + non-power-law(4%) + deficit(1%) | 60 |
| +1/2 requires filled 2D disk (line/annulus give only +0.12) | 61 |
| Near-field (r<3) provides 66-77% of S(k) | 59, 61 |
| Enhancement is geometric (random disk ≈ lattice disk) | 62 |
| C set by UV cutoff a=1, not lattice symmetry | 62 |
| Eigenvalue spectrum smooth (no fractal/localization) | 62 |

---

*Mar 2026*
