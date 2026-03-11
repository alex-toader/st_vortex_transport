# Paper Skeleton v2: σ_tr of a Disclination Loop

**Date:** Mar 2026
**Previous:** `11_paper_skeleton.md` (v1, pre-mechanism-chain)

---

## Top View

**One sentence:** We compute the phonon transport cross-section of a
disclination loop in a 3D lattice and explain why the Boltzmann integrand
sin²(k)·σ_tr(k) becomes nearly constant, leading to κ = O(1).

**Two stories:**
1. COMPUTATION — σ_tr(k) spectrum, κ(α) curve. New, nobody has done this.
2. UNDERSTANDING — why sin²(k)·σ_tr ≈ const. Mechanism chain: Born -5/2 + MS +1/2.

Story 2 is the intellectual contribution. Without it, paper = "we computed a number."

---

## Structure

### §1 Introduction

Argument flow:
1. Phonon drag on defects — established field (Ziman, Klemens)
2. Disclination loops — unstudied. Peierls holonomy = new type of scatterer
3. This paper: σ_tr(k) full BZ, mechanism chain, κ(α)
4. Motivation: in the stochastic mechanics framework, D = cR/(2κ) and κ = 1
   gives the Schrödinger diffusion constant D = ℏ/(2m). Computing κ from
   first-principles scattering tests whether this is structurally possible.
   (Context, not framework dependency — the scattering results stand alone.)
5. Paper structure

### §2 Model

2.1 Lattice + equation of motion (cubic, NN+NNN, K₁=2K₂ isotropy)
2.2 Disclination loop (Peierls gauge, Dirac disk, holonomy R(2πα))
2.3 FDTD + PML
2.4 Scattering measurement (wave packet, sphere recording, σ_tr definition)

### §3 Transport formula

3.1 Boltzmann integral: state result κ = (R/2π²) ∫ sin²(k) σ_tr dk.
    Derivation in Appendix B (Debye DOS → ω→k change of variable → lattice dispersion).
3.2 Polarization: uz decoupled, N_pol = 2
3.3 Full BZ coverage (13 k-pts, no tail extrapolation)

### §4 Results

4.1 σ_tr(k) spectrum (principal figure)
   - Decrease to k ≈ 1.7, rise above (polarization conversion)
   - Sub-geometric: σ_tr ~ R^{3/2} (stationary phase on loop contour)
   - Coherent scattering with incoherent-like angular profile: ±40% interference
     between ring segments, yet angular shape matches single-bond prediction
     within 3% — non-Born enhancement compensates coherent form factor falloff.

4.2 Flat integrand
   - sin²(k)·σ_tr ≈ const at α ≥ 0.2 (NN gauging, CV = 7.5%)
   - NOT flat at weak coupling (α = 0.05: CV = 34%)
   - NOT flat with NNN gauging (CV = 24-35% at all α)
   - As we show in §6, this arises from an algebraic cancellation between the
     lattice group velocity and the per-bond scattering amplitude, combined
     with a multiple-scattering correction to the Born exponent.

4.3 κ(α) curve
   - Smooth, monotonic, O(1) at α ~ 0.2-0.3
   - κ=1 crossing at α ≈ 0.20-0.33 (gauging × cutoff bracket)
   - K₁/K₂ robust (κ = 0.75-1.23)
   - AB comparison: fails on all 3 axes (k, α, R). Brief.

### §5 Systematics

5.1 NN vs NNN gauging (dominant: 2.1-4.9×)
5.2 k-cutoff (1.9-2.7×)
5.3 Gauge invariance (< 2.6%)
5.4 Near-field, angular grid, K₁/K₂ (small)
5.5 Uncertainty budget table

### §6 Mechanism

**The intellectual core.** Why is sin²(k)·σ_tr ≈ const?

6.1 Per-bond scattering (DERIVED, file 55)
   - σ_bond = C₀ · V / v_g², Born with lattice Green function
   - FDTD/Born CV = 4.6% — per-bond Born is CORRECT
   - V = cm1²·Z_mono + s_phi²·Z_dipo. At α ≥ 0.25: monopole dominates, V ≈ const(k)
   - cos²(k/2) from v_g cancels with sin²(k) = 4sin²(k/2)cos²(k/2). ALGEBRAIC.
   - Remaining: sin²(k/2) · V · N_eff ≈ const. V ≈ const → need N_eff ∝ 1/sin²(k/2)

6.2 Born exponent (ANALYTIC, files 57, 60)
   - Dirac disk of N ≈ πR² bonds → asymmetric forward cone
   - σ_total ~ (kR)^{-3/2}: cone widths δ ~ 1/√(kR), η ~ 1/(kR)
   - Transport weight (1-cosθ) ~ δ² adds -1 → σ_tr_Born ~ (kR)^{-5/2}
   - Verified: R=200, p = -2.491 (0.4% from -5/2)
   - This is UNIVERSAL: same for lattice disk, random disk, square disk

6.3 Multiple scattering correction (NUMERICAL, files 58-59)
   - T = (I - VG)^{-1}V with continuum G(r) = exp(ik_eff r)/(4πc²r)
   - Shifts Born -5/2 → MS ≈ -2.0 (shift +0.45 at α=0.30)
   - 91-104% of FDTD at all R tested
   - Shift = C · |V|, C ≈ 0.31 (UV-cutoff dependent, not lattice-symmetry dependent)
   - NOT resonance: |λ_max| = 0.3-0.6, far from 1
   - 80% from inter-bond 1/r propagation, only 9% from self-energy
   - Geometric: random disk gives same enhancement (ratio 0.92)
   - Direction UNIVERSAL (disk shape), magnitude LATTICE-SPECIFIC (UV cutoff a=1)
   - Born series oscillates (alternating sign corrections) → direct summation diverges
     at strong coupling. T-matrix resummation (I-VG)^{-1}V is essential, not optional.
     This is the physical argument for why MS treatment is necessary.

6.4 Assembly
   - sin²(k) · σ_ring = [4sin²(k/2)cos²(k/2)] · [C₀V/cos²(k/2)] · N_eff
   - cos²(k/2) cancels: ALGEBRAIC (DERIVED)
   - V ≈ const at α ≥ 0.25: DERIVED (monopole dominance)
   - 4sin²(k/2) · N_eff ≈ const: NUMERICAL from Born -5/2 + MS +0.5 ≈ -2.0
   - N_eff ∝ 1/sin²(k/2) (lattice) fits better than 1/k² (continuum): CV 5.8% vs 7.1%
     at L=100. Lattice dispersion correction to the exponent.
   - Flat integrand = algebraic cancellation + self-consistency of convergent κ integral.
     The +1/2 MS shift is the ONLY value giving convergent integrand at both BZ ends:
     shift < +1/2 → integrand diverges at k→π; shift > +1/2 → diverges at k→0.
     This is not coincidence — it is the self-consistent exponent for finite κ.
   - Balance point at α ≈ 0.29. Residual CV = 10.7%. Window α ∈ [0.20, 0.40]: CV < 15%.

### §7 Discussion

7.1 What is universal vs model-specific
   - UNIVERSAL (geometric): Born -5/2 from disk, MS +1/2 direction, R^{3/2} scaling
   - CUBIC-SPECIFIC: cos²(k/2) cancellation (from v_g), V ≈ const threshold, C = 0.31
   - On other lattices (Kelvin, WP): mechanism direction preserved, exact κ needs recalcul
   - Flat integrand is cubic-specific. κ = O(1) is probably robust.

7.2 Phonon drag literature
   - Point defects: Rayleigh σ ∝ ω⁴
   - Dislocations: σ ∝ ω (vibrating string)
   - Superfluid vortices: σ per unit length (Cleary 1968, Sonin 1997)
   - This work: to our knowledge, no previous σ_tr computation for a disclination
     loop exists. The flat integrand sin²(k)·σ_tr ≈ const has no known precedent.

7.3 Stochastic mechanics
   - D = cR/(2κ), κ = 1 → D = ℏ/(2m) → Schrödinger
   - κ = O(1) at α ≈ 0.2-0.3 — structurally compatible without fine-tuning
   - This is the physical motivation for computing κ (stated in §1). The result
     confirms structural compatibility. Whether κ = 1 exactly is a separate question
     requiring α from defect energetics and a physical k-cutoff.

7.4 Open directions
   - C ≈ 0.31: characterized (UV cutoff) but not derived analytically
   - -2.0 is approximate: crossing at α ≈ 0.29, window α ∈ [0.20, 0.40] gives CV < 15%
   - Vectorial elasticity on real foam lattices — separate project
   - κ = 1 exact: requires α from holonomy/energy + physical k-cutoff

---

## Figures

1. σ_tr(k) spectrum — 4 α values, full BZ (principal)
2. sin²(k)·σ_tr integrand — flat at α ≥ 0.2 vs non-flat at α = 0.05
3. Two-panel mechanism figure (data: file 58, R=5):
   (a) N_eff(k) log-log: Born (slope -5/2), MS (-2.0), FDTD (-2.0)
   (b) Integrand sin²(k)·σ_tr: Born CV=35%, MS CV=10%, FDTD CV=7%
5. κ(α) — NN + NNN, two k-cutoffs, horizontal line at κ=1
6. σ_tr vs R — log-log, R^{3/2} fit
7. Gauge invariance — spread vs k (can go to appendix if tight on space)

---

## Key Changes from v1

| Item | v1 skeleton | v2 |
|------|------------|-----|
| "Incoherent scattering" | Used throughout | Coherent with incoherent-like angular profile (defined) |
| Mechanism | "open question" in §4.3, §6.4 | Own section §6, partially resolved |
| Section order | Results → Mechanism → Systematics | Results → Systematics → Mechanism (conventional) |
| §7.3 stochastic mech | "Reader can skip" | Motivation (§1) + result (§7.3), integrated |
| AB comparison | Full §4.3 in v1 | Brief in §4.3, not separate section |
| Universality | Not discussed | §7.1: geometric vs cubic-specific |
| Born -5/2 derivation | Not present | §6.2 (analytic) |
| MS +1/2 | Not present | §6.3 (numerical) |
| Assembly | Not present | §6.4 (algebraic + numerical) |
| Figure 5 | Flow chart diagram | N_eff log-log plot (Born/MS/FDTD) |
| New figure | — | Born/MS/FDTD integrand comparison |
| Forward reference | — | §4.2 → §6 (prevents "mystery result") |
| One-sentence summary | "First computation..." | "We compute and explain why..." |

---

## Page Estimate

| Section | Pages |
|---------|-------|
| §1 Introduction | 1.5 |
| §2 Model | 2.0 |
| §3 Transport formula | 0.5 |
| §4 Results | 2.5 |
| §5 Systematics | 1.5 |
| §6 Mechanism | 3.0 |
| §7 Discussion | 1.5 |
| References | 0.5 |
| Appendix A (tests) | 1.0 |
| Appendix B (κ derivation) | 1.0 |
| **Total** | **~14.5** |

---

## Referee Preparation

| Question | Answer |
|----------|--------|
| Why disk? | Dirac disk is the gauge surface of the disclination loop — topological consequence, not design choice. All bonds crossing this surface carry Peierls coupling. |
| Is -2 exact? | No. Crossing p_MS = -2.0 at α ≈ 0.29. Window [0.20, 0.40] gives CV < 15%. Quantitative coincidence, not symmetry. |
| How lattice-dependent is κ? | Mechanism direction (Born -5/2 + MS positive shift) is geometric/universal. Constants (C, v_g form, threshold) are lattice-specific. κ = O(1) probably robust; exact value needs recalculation per lattice. |
| Flat integrand only NN? | Yes. NNN: CV = 24-35% at all α. Both gaugings give identical holonomy; difference is lattice-scale gauge distribution. Presented as discovery specific to NN, not universal. |
| Why stochastic mechanics? | Physical motivation for computing κ — provides the only known context where κ = 1 has meaning. Results stand alone as phonon scattering computation. |
| Simple cubic — how general? | Simplest 3D lattice with tunable isotropy. Mechanism has universal and lattice-specific parts (§7.1). Generalization to foam lattices is future work. |

---

*Mar 2026*
