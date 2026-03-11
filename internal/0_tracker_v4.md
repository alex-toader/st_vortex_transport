# W21: κ from First-Principles Scattering — Tracker v4

**Date:** Mar 2026
**Previous:** `0_tracker.md` (routes 1-18), `0_tracker_v2.md` (17-25), `0_tracker_v3.md` (25-50)
**Paper skeleton:** `11_paper_skeleton.md`

---

## Central Result

First computation of σ_tr(k) for a disclination loop (vortex ring) in 3D elastic lattice.
13 wavenumbers spanning the full BZ.

**Principal results:**
- σ_tr(k): decrease to k≈1.7, rise at higher k (polarization conversion)
- Flat integrand: sin²(k)·σ_tr ≈ const at α ≥ 0.2 (NN gauging, CV=7.5%)
- Coherent scattering with incoherent-like SHAPE (files 45-47)
- AB prediction fails on all 3 axes (k, α, R)

**Application (§6.3):** κ(α) = drag prefactor. κ = O(1) at α ~ 0.3.
Chain: κ = 1 → γ = c/R → D = cR/2 → D = ℏ/(2m) → Schrödinger.

**Physical model:** generic elastic medium on cubic lattice with NN+NNN springs
(K₁=2K₂ for isotropy). Defect via Peierls gauge coupling.

---

## Data Tables

### κ(α) — NN+NNN gauging (13 k-pts file 23, 7 k-pts file 28)

| α | κ_NNN(k≤0.9) | κ_NNN(k≤1.5) | κ_NNN(k≤2.1) |
|---|-------------|--------------|--------------|
| 0.10 | 0.099 | 0.266 | — |
| 0.15 | 0.240 | 0.621 | — |
| 0.20 | 0.451 | 1.136 | — |
| 0.25 | 0.720 | 1.782 | 2.783 |
| 0.30 | 1.018 | 2.490 | 3.845 |
| 0.40 | — | 3.722 | 5.650 |
| 0.50 | — | 4.205 | 6.339 |

κ_NNN = 1 crossing: α ≈ 0.30 (k≤0.9) or α ≈ 0.19 (k≤1.5).

### κ(α) — NN-only gauging (comparison)

| α | κ_NN(k≤0.9) | κ_NN(k≤1.5) | κ_NN(k≤2.1) | κ_NN(k≤3.0) |
|---|------------|-------------|-------------|-------------|
| 0.10 | 0.025 | 0.056 | — | — |
| 0.15 | 0.084 | 0.164 | — | — |
| 0.20 | 0.189 | 0.355 | — | — |
| 0.25 | 0.333 | 0.625 | 0.898 | 1.211 |
| 0.30 | 0.495 | 0.944 | 1.341 | 1.752 |
| 0.40 | — | 1.545 | 2.188 | 2.760 |
| 0.50 | — | 1.798 | 2.552 | 3.185 |

### Flat integrand CV

| α | CV_NN | CV_NNN | Verdict |
|---|-------|--------|---------|
| 0.05 | 34.2% | — | NOT flat |
| 0.10 | 15.3% | 34.9% | NN marginal, NNN not flat |
| 0.15 | 15.9% | 31.0% | Neither flat |
| 0.20 | 15.5% | 27.7% | NN marginal, NNN not flat |
| 0.25 | 11.9% | 25.5% | NN roughly flat, NNN not flat |
| 0.30 | 7.5% | 24.2% | NN flat, NNN not flat |
| 0.40 | 3.9% | — | NN very flat |
| 0.50 | 5.6% | — | NN flat |

### Systematics on κ

| Source | Effect on κ | Status |
|--------|-----------|--------|
| NN vs NNN gauging | 2.1-4.8× (k-dependent) | **Dominant** |
| k-cutoff (k≤0.9 vs k≤1.5) | 1.5-2.4× | Large |
| Near-field (r_m) | ~7% | Small |
| Gauge violation (Dirac shift) | < 2.1% (NNN) | Small |
| Recording (k≤2.1) | < 0.2% | Negligible |

Combined: κ=1 at α ∈ [0.19, >0.30] depending on gauging × cutoff.

---

## Mechanism Understanding (Mar 2026)

### What is established

1. **Empirical factorization:** σ_tr(k,R) ≈ A(α) × R^{3/2} / sin²(k).
   R-part: stationary phase on curved contour (universal, any α).
   k-part: 1/sin²(k) requires strong coupling α ≥ 0.2 (NN gauging).

2. **R^{3/2} = stationary phase.** ℓ_c ~ √(Rλ), N_patches ~ √(kR), σ ~ R^{3/2}.
   Measured: p=1.51-1.61 at kR>2. Crossover geometric→SP at kR≈2. (Files 18, 27)

3. **Not Born.** σ_tr ∝ α^{2.56}, not α². Non-Born enhancement R(k) grows 46×
   from k=0.3→1.5. Born amplitude mixing (p=1.5-1.9) always below data (p=2.2-2.7).
   Excess ~0.7 is T-matrix effect. (Files 26, 35)

4. **AB fails on all 3 axes.** k: 1/sin²(k), not 1/k. α: C/sin²(πα) varies 44%.
   R: ~R^{1.7}, not R¹. AB comparison is apples-to-oranges (files 31-32).

5. **Coupling × geometry table:**

   | Coupling | Geometry | CV | File |
   |----------|----------|-----|------|
   | strain | sphere | 89-152% | 34 |
   | strain | ring | 0% (null) | 37 |
   | displacement | ring | 7.5% | 37 |
   | displacement | sphere | 35-46% | 42 |

   Flat integrand = displacement coupling + ring topology. Neither alone suffices.

6. **Diagonal/off-diagonal on ring:** diagonal (cm1·I) CV=8.8%, off-diagonal
   (s_phi·J) CV=23.7%. Diagonal dominates at α>0.25 (equality at α=0.25). (File 35)
   Weak coupling: diagonal is worst (CV=62% at α=0.05). (File 36)

7. **Strain null mechanism.** Ring axis ⊥ propagation: strain coupling is zero
   on z-bonds (wave constant in z → Δu=0). yz-plane ring (axis along prop):
   CV=42.5%. (File 38)

8. **NNN/NN: same holonomy, different coupling.** Both give R(2πα). NNN adds
   312 bonds (K₂=0.5). NNN/NN ratio: 1.33 (k=0.3) → 3.56 (k=1.5).
   Phase prediction DISPROVED by selective gauging (file 40). (Files 28, 40)

9. **Coherent scattering with incoherent-like shape.** ±40% interference
   between quarter-arcs (files 45-46). Interference is GEOMETRIC (identical
   at α=0.10 and α=0.30). Despite coherence, σ_tr follows Z_avg/sin²(k)
   shape at α=0.30 (CV=2.7%). Non-Born R(k) compensates coherent Born
   form factor falloff. (Files 43-47)

10. **Per-bond σ_tr ≈ const(k).** Best fit: constant (CV=20.3%), power law k^{0.29}.
    Grows only 1.6× over k=0.3-1.5. Born predicts 1/sin²(k) (varies 11.5×).
    σ_bond ≈ const is the non-Born per-bond property that enables flat integrand.
    (File 51, α=0.30)

    σ_bond(k): 0.060, 0.054, 0.057, 0.062, 0.071, 0.081, 0.096

    Diagonal-only (cm1·I) per bond: CV α-INDEPENDENT (70.0% vs 69.0%, file 50).
    Full Peierls per bond: CV α-DEPENDENT (96.3% vs 70.8%, file 49).
    The α-dependence in full Peierls comes from s_phi·J (rotation channel):
      α=0.10: |s_phi/cm1|=3.08, rotation dominates → adds 26pp to CV (70→96%)
      α=0.30: |s_phi/cm1|=0.73, displacement dominates → adds <2pp (69→71%)

11. **Flat integrand is COLLECTIVE.** Ring interference reduces per-bond CV.

    Diagonal-only: 1 bond (69%) → ring 81 bonds (8.8%), factor ~8×. (Files 50, 35)
    Full Peierls:  1 bond (71%) → quarter (15.5%) → half (4.8%) → ring (7.4%).
    Hierarchy from file 48 applies to full Peierls, not diagonal-only.

12. **N_eff decomposition (file 51).** σ_ring = σ_bond × N_eff(k).
    N_eff: 681→261→135→81→53→39→29 (drops 23.3×). NOT incoherent.
    N_eff ≈ (I_tr/Z_avg) × g(k), g(k) grows 1→2.1 (CV=21.3%).
    σ_bond × I_tr ≠ Z_avg (CV=145%). Not algebraic identity.
    Non-Born ring (R_ring grows 45.8×) > per-bond (R_bond grows 21.8×).
    Ring interference amplifies non-Born enhancement by ~2× at high k.

    Balance: sin²·σ_ring = sin²·σ_bond × N_eff = (18.3× up) × (23.3× down) = 1.27×.
    Adjacent destructive (×0.32), distant constructive (×1.72).
    Cross/full fraction = -0.45 (geometric, α-independent). (Files 48-50)

    α-ratio σ(0.30)/σ(0.10) at each level:

    | Level | ratio | Born | FDTD/Born | Note |
    |-------|-------|------|-----------|------|
    | 1 bond, diagonal-only | 35.5 | 47.0 | 0.76 | below Born (cm1² ratio) |
    | 1 bond, full Peierls | 26.0 | 6.9 | 3.79 | above Born (sin²πα ratio) |
    | ring 81, full Peierls | 17.0 | 6.9 | 2.48 | above Born, ring reduces excess |

    Born basis changes with coupling: diagonal-only uses cm1²/cm1² = 47.0;
    full Peierls uses sin²(0.3π)/sin²(0.1π) = 6.9 (rotation is relatively
    stronger at α=0.30). Displacement non-Born suppresses α-ratio below Born
    (0.76×). Adding rotation: FDTD drops modestly (35.5→26.0) but Born drops
    dramatically (47.0→6.9) → FDTD/Born flips above 1. Ring interference
    reduces FDTD/Born from 3.79 to 2.48.

13. **Polarization decomposition (file 53).** σ_bond = σ_xx + σ_xy + σ_xz.
    σ_xz = 0 (uz decoupled). Compensation hypothesis FAILS.

    α=0.30: σ_xx ≈ const (0.060→0.080, CV=66%), σ_xy grows 58× (0.03%→16.5%).
    α=0.10: σ_xx ≈ const (0.0013→0.0016, CV=62%), σ_xy grows 50× (7.6%→77.9%).

    σ_xx is the non-Born channel: Born predicts 13.6× decrease, FDTD gives
    1.2-1.4× INCREASE. CV(σ_xx) ≈ 62-66% at BOTH α — α-INDEPENDENT.
    This matches diagonal-only CV≈69% (file 50): σ_xx ≈ σ_diag per bond.

    The α-dependence in σ_tot comes entirely from σ_xy weight:
      α=0.30: xy is 0.5-16.5% of total → small perturbation → CV≈71%
      α=0.10: xy is 7.6-77.9% of total → dominates at high k → CV≈96%
    Threshold α≈0.25 = where σ_xy becomes small enough that σ_xx dominates.

    σ_xx REVERSAL: Born predicts σ_xx ∝ Z_avg/sin²(k) (decreasing).
    FDTD shows σ_xx slightly increasing. The non-Born reversal in same-pol
    scattering is the elementary mechanism — NOT polarization compensation.
    Scalar T-matrix (file 52) doesn't capture it → must be vectorial T-matrix.

14. **Born mechanism with correct lattice normalization (file 55 v3).**

    σ_bond = C₀ × [cm1²·Z_mono + s_phi²·Z_dipo] / v_g²

    Two ingredients:
    a) Monopole/dipole source decomposition (z-bond, x-incident):
       x-channel: MONOPOLE (same forces at both sites), Z_mono = 8π(1+sinc(k)) ≈ const (CV=5.9%)
       y-channel: DIPOLE (opposite forces), Z_dipo = 8π(1-sinc(k)) grows 22×

    b) Born normalization: σ ∝ V/v_g², where v_g = √(K₁+4K₂)·cos(k/2).
       The old 1/sin²(k) was WRONG — included spurious 1/(4sin²(k/2)) factor.
       1/v_g² varies 1.83×, not 1/sin²(k) which varies 11.4×.

    NOTE: Form factors are z-bond specific (ring has only z-bonds).
    For x-bonds the form factor is 4cos²((k+q_x)/2) ≠ 4cos²(q_z/2).

    FDTD/Born ratios (per z-bond): σ_xx CV=4.8%, σ_xy CV=11.0%, σ_tot CV=4.6%.
    Per-bond Born with v_g normalization matches FDTD at ~5% level.
    Ring-level non-Born (σ_ring ∝ α^2.56, file 26) comes from N_eff (interference).

    Flat integrand chain: sin²(k)·σ_ring
    = 4sin²(k/2)·cos²(k/2) · C₀·V/cos²(k/2) · N_eff
    = 4sin²(k/2) · C₀·V · N_eff
    cos²(k/2) cancels between sin²(k) and 1/v_g².
    Remaining: sin²(k/2)·N_eff ≈ const (CV=5.4%).
    N_eff ∝ 1/sin²(k/2) = lattice analog of continuum 1/k².

    CV(V) scan: α=0.10→28.2%, α=0.20→4.6%, α=0.25→0.0%, α=0.30→2.7%, α=0.50→5.9%.
    α-threshold at 0.25 exactly: |cm1|=|s_phi| → CV(V)=0.

### Current picture

Two levels — per-bond DERIVED, ring-level EMPIRIC:

**Per-bond (COMPLETE, from first principles):**
  σ_bond = C₀ × [cm1²·Z_mono + s_phi²·Z_dipo] / v_g²
  Born with v_g normalization. FDTD/Born CV=4.6%.
  Old 1/sin²(k) was wrong (spurious 1/(4sin²(k/2))). Correct: 1/cos²(k/2).
  Monopole/dipole from Peierls coupling on z-bond.
  V ≈ const at α≥0.25 (monopole dominates). CV(V): α=0.25→0%, α=0.30→3%.

**Ring-level (SEMI-EMPIRIC, not derived):**
  σ_ring = σ_bond · N_eff. N_eff ∝ 1/sin²(k/2) (collective interference).
  Ring non-Born (σ ∝ α^2.56) comes from N_eff, not per-bond deviation.

**Flat integrand:**
  sin²(k)·σ_ring ≈ const (CV=7.4%).
  Algebric: cos²(k/2) in sin²(k) cancels 1/cos²(k/2) in v_g². DERIVED.
  Remaining: 4sin²(k/2) × V × N_eff ≈ const.
  V ≈ const at α≥0.25: DERIVED (monopole dominance).
  sin²(k/2)·N_eff ≈ const (CV=5.4%): CIRCULAR — follows tautologically from
  σ_ring ∝ 1/sin²(k) (empiric) combined with σ_bond ∝ 1/cos²(k/2) (Born).
  Not an independent confirmation.

**What is and isn't explained:**
  DERIVED: per-bond Born shape, v_g normalization, monopole/dipole, α-threshold.
  EMPIRIC: N_eff ∝ 1/sin²(k/2), i.e., why σ_ring ∝ 1/sin²(k).
  OPEN: why collective interference of 81 z-bonds gives exactly N_eff ∝ 1/sin²(k/2).
  This is equivalent to the original question "why is the integrand flat?"
  reformulated as a ring-level interference question.

### Central open question — PARTIALLY RESOLVED (file 55 v3)

WHY is sin²(k)·σ_ring ≈ const (CV=7.4%)?

RESOLVED (per-bond, ~60%):
  σ_bond = C₀·V/v_g² (Born with correct lattice Green function).
  FDTD/Born CV=4.6%. The cos²(k/2) cancellation is algebraic.
  V ≈ const at α≥0.25 from monopole dominance.
  α-threshold: |cm1|=|s_phi| at α=0.25 exactly.

OPEN (ring-level, ~40%):
  N_eff ∝ 1/sin²(k/2). Why? Equivalent to asking why σ_ring ∝ 1/sin²(k).
  Empirically solid (files 17, 18). Consistent with stationary phase (R^1.6).
  But not derived analytically from the ring geometry.
  sin²(k/2)·N_eff ≈ const is a reformulation, not a derivation.

CLOSED: T-matrix route. Per-bond Born already correct.

### Mechanism Chain — for paper/skeleton (files 55-59)

Complete chain from first principles to κ = O(1):

**Step 1 — Per-bond coupling (file 55, DERIVED)**
Each Peierls bond on the vortex ring creates a displacement scatterer:
  σ_bond = C₀ × [cm1²·Z_mono + s_phi²·Z_dipo] / v_g²
Born level. FDTD/Born CV = 4.6%. No free parameters.
At α ≥ 0.25: monopole dominates, V ≈ const(k) (CV < 3%).

**Step 2 — Disk geometry → Born -5/2 (files 57, 60, ANALYTIC)**
Dirac disk of N ≈ πR² bonds scatters coherently. Forward cone:
  Born N_eff ~ (kR)^{-5/2}  (transport)
-5/2 = -3/2 (asymmetric cone area) + (-1) (transport weight).
Cone widths: η ~ 1/(kR), δ ~ 1/√(kR). σ_total ~ (kR)^{-3/2}.
Transport (1-cosθ) ~ δ² ~ 1/(kR) suppresses forward → σ_tr ~ (kR)^{-5/2}.
Verified: R=200, kR>15: p = -2.491 (0.4% from -5/2). Correction O(1/(kR)).

**Step 3 — Multiple scattering → +1/2 correction (files 58-59, NUMERICAL)**
T = (I - VG)^{-1}V shifts Born exponent:
  MS N_eff ~ (kR)^{-2.0}  (shift +0.45 at α=0.30)
91-104% of FDTD at all R=3,5,7,9. Integrand CV: Born 35% → MS 10% → FDTD 7%.
Shift = C × |V|, C ≈ 0.31 ± 0.03 (R-independent).
Shift is 80% from inter-bond 1/r propagation, only 9% from self-energy G₀₀.
Single-mode (Rayleigh quotient λ_eff) captures 77-100% (exact at R≥7).
|λ_eff| ~ k^{-0.77}: phase coherence at low k → more cooperation → shift.
NOT resonance (|λ_max| = 0.3-0.6). Born series oscillates → full resummation needed.

**Step 4 — Flat integrand (partial: algebraic + numerical)**
sin²(k) · σ_ring = sin²(k) · σ_bond · N_eff
cos²(k/2) cancels between sin²(k) = 4sin²(k/2)cos²(k/2) and 1/v_g² = 1/cos²(k/2).
This is an algebraic identity. Remaining: 4sin²(k/2) · V · N_eff.
V ≈ const (Step 1, DERIVED).
sin²(k/2) · N_eff ≈ const requires N_eff ∝ 1/sin²(k/2) ≈ 4/k².
This is NOT derived — it follows NUMERICALLY from Steps 2+3:
Born -5/2 + MS +0.45 ≈ -2.0 at α=0.30 (file 60 Part E: p_MS = -1.97).
The flat integrand is approximate (CV=7-10%), not an identity.
It is a quantitative coincidence at α≈0.30, not forced by symmetry.

**Step 5 — κ = O(1) (file 22, NUMERICAL)**
κ(α=0.30) = 1.02 (k ≤ 0.9), 2.49 (k ≤ 1.5).
Threshold: |V| = (p_Born - 2)/C = 1.26 → α = 0.29 (self-consistent).

**Status:**

| Step | Content | Status |
|------|---------|--------|
| 1 | Per-bond Born: σ_bond = C₀V/v_g² | DERIVED (analytic, verified 4.6%) |
| 2 | Born exponent -5/2 from disk geometry | ANALYTIC (cone -3/2 + transport -1, file 60) |
| 3 | MS shift +1/2 from T=(I-VG)^{-1}V | NUMERICAL (91-104%, C=0.31 not analytic) |
| 4 | cos²(k/2) cancellation + N_eff ≈ k^{-2} | PARTIAL (cancellation=identity, N_eff=numerical from 2+3) |
| 5 | κ = O(1) at α ≈ 0.30 | NUMERICAL (from steps 1-4) |

**Characterized:** C ≈ 0.31 is UV-cutoff dependent (a=1), not lattice-symmetry dependent.
C ~ 0.28 + 0.12·log(density). Not derivable analytically, but origin understood (file 62).
**Closed:** per-bond T-matrix (wrong direction, file 57), resonance (|λ| far from 1,
file 59), strain coupling (null, file 37), AB factorization (fails 3 axes, files 31-32).

---

## Open Investigations

### Mechanism — do next (priority order)

**I11 — DONE (file 51).** σ_bond × I_tr ≠ Z_avg (CV=145%). Not algebraic.
N_eff ≈ I_tr/Z_avg with 2× drift (CV=21.3%). R_ring > R_bond by ~2× at high k.

**I12 — DONE (file 51).** σ_bond grows 1.6× over full BZ (CV=20.3%).
Old Born with 1/sin²(k) predicted 11.5× — WRONG normalization.
Correct Born with 1/v_g² predicts 1.7× — matches FDTD (file 55 v3).

**I14 — DONE (file 52).** Scalar T-matrix FAILS. |DK·G| ≤ 0.21 → Born at all α.

**I16 — DONE (file 53).** Polarization decomposition. Compensation FAILS.
σ_xx ≈ const (CV≈62-66%, α-independent). σ_xy grows 50-58× but is small
at α≥0.25. α-threshold = |cm1|>|s_phi| ⟺ α=0.25 exactly.

**I17 — DONE (file 54).** Vectorial 2×2 T-matrix FAILS. |DK·G|_eig ≤ 0.24.
|T_xx|² varies only 2%. CV(σ_xx_T)=109% vs FDTD 14%. σ_xx≈const is NOT
a T-matrix effect (scalar or vectorial). Near-field or lattice effect.

**I18 — DONE (file 55 v3).** Born mechanism per z-bond: σ_bond = C₀·V/v_g².
FDTD/Born: σ_xx CV=4.8%, σ_tot CV=4.6%. Ring flatness from N_eff ∝ 1/sin²(k/2).
Corrected: sigma_ring data (file 18), removed direction-independence claim,
distinguished per-bond Born from ring non-Born, added CV(V) scan.
Z_mono direction-independent (Z_x=Z_z to <0.001%). File 55 v1 Part C was wrong.
Flat integrand: cos²(k/2) cancels, sin²(k/2)·N_eff≈const (CV=5.6%).

**I3 — g(k) at R=3,5,7,9. Zero compute. DO THIRD.**
g(k) = N_eff/(I_tr/Z_avg) from files 18+43. File 51 gave g growing 1→2.1
at R=5. If g(k) is R-independent → per-bond T-matrix correction. If varies
with R → collective effect. Reframed from R_coh after file 51 results.

**I13 — Open arc, 81 bonds straight line. ~20 min FDTD. DO FOURTH.**
Displacement K_eff=-1.3 on 81 z-bonds in a row (not curved, not closed).
CV ≈ 7.5% → flatness from N large (statistical)
CV ≫ 7.5% → closure essential (topological)

**I2 — α-scan of g(k). Zero compute. DO FIFTH.**
Files 28/36 have σ_FDTD at α=0.05-0.30. Compute g(k,α): does the 2×
drift depend on α? Localizes α-crossover for ring correction.

**I4 — g(k) at variable K_eff. Zero compute. DO SIXTH.**
File 41 has σ_FDTD at K_eff=-0.5,-1.0,-1.3 (displacement coupling on ring).
Does g(k) correction depend on coupling strength?

**I9 — Cross-term at displacement pure. ~20 min FDTD. SEVENTH.**
File 37: displacement K_eff=-1.3 gives CV=8.9%. Run quarter additivity.
If cross/full ≈ -0.45 → interference geometry is universal.

### Mechanism — remaining

**I5 — Displacement at K_eff=-0.191 on ring. ~20 min FDTD.**
Equivalent to cm1 at α=0.10. Compare with Peierls α=0.10 on ring.
Separates rotation effect from coupling magnitude on ring flatness.

**I6 — Square ring test. ~20 min FDTD.**
Same perimeter, different symmetry. If CV similar → flat integrand from
closed loop topology, not circular symmetry. Needs new gauge function.

**I8 — Eighth-ring CV from file 44. Zero compute.**
Extend hierarchy: eighth → quarter → half → full. Tests whether half
is optimal flattening level.

**I9** — Moved to "do next" section above.

**I13** — Moved to "do next" section above.

**I10 — Half-ring paradox at R=3 and R=9. ~20 min FDTD.**
Half CV < full CV at α=0.30 (4.8% vs 7.4%). Test R-dependence.

### Theory (high impact, zero compute)

**T7/T1 — CLOSED by I14+I17 (files 52, 54).** Both scalar and vectorial
T-matrix are Born (|DK·G| ≤ 0.24). |T_xx|² varies only 2%. T-matrix
(single-site) cannot produce σ_xx ≈ const.

**F17 — CLOSED.** Neither T-matrix |T|² nor polarization compensation
produces σ_xx ≈ const. The effect is beyond single-site scattering theory.

**Route 22 — RESOLVED (file 55 v3, deepened in file 56).** Per-bond Born with
v_g normalization: σ_bond = C₀·V/v_g². FDTD/Born CV=4.6%. Ring non-Born
(α^2.56) is collective (N_eff), not per-bond. sin²(k)·σ_ring CV=7.4%.

N_eff deep dive (file 56): N_eff ∝ 1/k² (CV=2.3%), better than 1/sin²(k/2)
(CV=5.4%). Model N_eff = N²/(1+c·k²·R^q), c≈2.1, q≈2.37. Formula: q = 2β - p_R
(β≈1.96, p_R=1.6), predicted q=2.32 vs fit 2.37 (Δ=0.05).

Asymmetric forward cone (file 57): Born exponent is EXACTLY -5/2 (transport),
-3/2 (total). Cause: 2D planar disk → Q_⊥ grows linearly in ε_φ but
quadratically in ε_θ → forward cone elongated, solid angle ∝ 1/(kR)^{3/2}.
Transport weight adds exactly -1 → Born = -5/2. FDTD = -2.0 (correction +1/2).
Born integrand CV = 35% (NOT flat). FDTD CV = 7.4% (flat).
Flat integrand requires non-Born correction from -5/2 to -2.
Cross-check: discrete R=5 vs Airy Δ=0.017. Q_⊥/|ε_φ| = k (varies with k),
geometric ratio √(2kR) is universal. 2D shapes universal (disk=square); 1D
orientation-dependent (along x: -1.72, along y: -1.04).
Correction characterization (file 57 Parts F-G): N_eff_FDTD = 0.56 × √ω × N_eff_Born.
C ≈ 0.56, R-independent (R^{-0.004}), CV=2.9%. √ω = √(2c·sin(k/2)).
Single-bond T-matrix: |T/V|² ≈ 1.33-1.56 at Q=0, drops to ~1.0 at Q=1.5.
Enhancement stronger at low k → exponent MORE negative (-2.53 vs Born -2.41).
T-mat shift -0.12 vs FDTD +0.45 (wrong direction).
ELIMINATED: single-bond T-matrix (wrong direction), finite-size (C R-independent).
Correction is collective multiple scattering.

### Route 22b: √ω mechanism — CONFIRMED (file 58)

**RESOLVED:** N_eff correction comes from collective multiple scattering.
File 58: T = (I - V·G)^{-1}·V with continuum G(r) = exp(ik_eff r)/(4πc²r).

N_eff exponents (V² normalization):

| R | Born | MS | FDTD | MS shift | FDTD shift | MS/FDTD |
|---|------|-----|------|----------|-----------|---------|
| 3 | -2.22 | -1.81 | -1.78 | +0.41 | +0.45 | 91% |
| 5 | -2.42 | -1.97 | -1.97 | +0.45 | +0.45 | 100% |
| 7 | -2.37 | -1.97 | -1.98 | +0.40 | +0.39 | 104% |
| 9 | -2.41 | -1.99 | -1.96 | +0.41 | +0.44 | 94% |

Integrand CV (R=5): Born 35.1% → MS 9.6% → FDTD 7.4%.
MS flattens integrand from 35% to 9.6% — within 2.2% of FDTD.

k_eff = ω/c = 2sin(k/2) (continuum, verified: kv overshoots 17%).
C_MS ~ R^{-0.25} (mild R-dep, FDTD R-independent — remaining ~10% from lattice).
OPEN: why exactly +1/2 analytically (eigenvalue structure of VG for disk).

### Route 22c: +1/2 mechanism — RESOLVED (file 59, Parts A-G)

**File 59 results (7 parts):**

**Part A — Eigenvalue spectrum.** |λ_max| = 0.30-0.60 (far from 1). NOT resonance.

**Part B — Phase removal.** Without exp(ikr): shift +0.83. With phase: +0.45.
Phase REDUCES shift. Mechanism: phase randomization at high k.

**Part C — α scan.** Shift = C × |V|, C = 0.314 ± 0.030 (R-independent).
Flat integrand: |V| = 1.26 → α = 0.29 (self-consistent).

**Part D — Born series.** Oscillates (27%, 152%, 100%). Full resummation needed.

**Part E — Clean decomposition (CORRECTED).**

| G type | shift | % |
|--------|-------|---|
| G₀₀ diagonal only | +0.04 | 9% |
| Off-diag 1/r only | +0.36 | 80% |
| Physical (both) | +0.45 | 100% |
| Uniform G (max) | +0.69 | — |

Old "floor +0.21 from G₀₀" was WRONG (eps=5 still had short-range 1/r⁶).
True self-energy contribution is only 9%. Shift is 80% from inter-bond 1/r.

**Part F — Geometry.** Disk >> line, annulus. Disk interior matters.

**Part G — Single-mode (Rayleigh quotient).**
λ_eff = ⟨b|VG|b⟩/⟨b|b⟩ projects VG onto incident mode.
|λ_eff| ~ k^{-0.77}: phases coherent at low k → cooperation → enhancement.
Single-mode captures: 77% (R=2) → 80% (R=5) → 100% (R≥7).
~80% of λ_eff from r ≤ 3 (short-range lattice pairs dominate).
Continuum disk integral fails (off by 4× and different shape).

**Physical picture:** At low k, phases exp(ikr) are coherent across disk →
bonds cooperate → strong MS enhancement. At high k, phases randomize →
cooperation drops. This k-dependent cooperation shifts N_eff from -5/2 to -2.
C ≈ 0.31 is a lattice geometric constant from the discrete pair sum.

**CLOSED:** C ≈ 0.31 is lattice-specific (not derivable from continuum integrals).
No further analytic route — the constant comes from short-range lattice structure.

### Route 22d: Born -5/2 derivation + gap closure — RESOLVED (file 60)

Born exponent -5/2 derived analytically:
- σ_total ~ (kR)^{-3/2} from asymmetric forward cone (δ ~ 1/√(kR), η ~ 1/(kR))
- Transport weight adds -1 → σ_tr ~ (kR)^{-5/2}
- Verified at R=200, kR>15: p = -2.491 (0.4% from -5/2)

Gap 2 resolved: -2.0 is NOT exact. Crossing point p_MS(α) = -2.0 at α ≈ 0.29.
Window α ∈ [0.20, 0.40] gives CV < 15%.

Gap 4 resolved: N_eff ∝ 1/k² is consequence of Born (-5/2) + MS (+1/2), not independent.

Residual CV = 10.7%: sinc²(k/2) (~6%), non-power-law (~4%), p_MS deficit (~1%).

### Route 22e: Geometry of +1/2 — RESOLVED (file 61)

**Result:** +1/2 enhancement requires filled 2D disk. Line and annulus give
only +0.12 (saturated, does not grow with size). Disk p_enh grows:
R=3: +0.25, R=5: +0.29, R=9: +0.34, R=15: +0.44, R=25: +0.61.
Extrapolation: 1/R → 0.53 (≈ +1/2), 1/√R → 0.69 (both approximate).

**Idea 1 — S(k) pair sum.** Computed. |S| ~ k^{-0.77} at R=5. Matches λ_eff
from file 59. S(k) is not a clean power law (exponent varies with R).
C ≈ 0.31 cannot be derived analytically from S(k).

**Idea 2 — Coherence length.** |S| ~ k^{-0.77}, not k^{-1} as ℓ_c ~ 1/k
would predict. Qualitatively correct direction, not quantitative.

**Idea 3 — Disk maximizes Σ 1/r_ij.** Confirmed by file 61 Part C.
Filled disk p_enh grows with R. Annulus and line do not.
Interior bonds (short-range 1/r coupling) are essential.

C ≈ 0.31 remains a numerical lattice constant. Not analytically derivable,
but its geometric origin is understood: filled disk interior at r ≤ 3.

### Remaining gaps — assessment + sanity checks

**Gap 1 — C ≈ 0.31 is a lattice constant.** CLOSED (file 62 Part B+C).
C is UV-cutoff dependent (lattice spacing a=1), NOT lattice-symmetry dependent:
random disk at same density gives C = 0.270 = lattice C. Higher density →
C grows as C ~ 0.28 + 0.12·log(density) — 1/r UV divergence. Any medium with
nearest-neighbor spacing a=1 gives same C, regardless of lattice type.

**Gap 2 — p_MS ≈ -1.97, not exactly -2.** CLOSED (file 60 Part E+F).
Flat integrand is NOT a symmetry — balance point at α ≈ 0.30. CV = 10.7%
decomposed: sinc²(6%) + non-power-law(4%) + deficit(1%).

**Gap 3 — Why disk maximizes effect.** CLOSED (file 62 Part A+B).
Eigenvalue spectrum smooth (no fractal/localization). Random disk gives
p_enh = 0.32 vs lattice 0.35 (ratio 0.91). Effect is geometric (disk shape),
not lattice-structure-specific.

**Sanity checks — file 62 (4s):**

| Check | Result | Verdict |
|-------|--------|---------|
| S1: eigenvalue density | Smooth, bulk [0.1, 0.3], one dominant mode at low k | No fractal/localization |
| S2: random disk | p_enh = 0.33 ± 0.016 vs lattice 0.35 (ratio 0.92, 10 trials) | Geometric, not lattice-specific |
| S3: continuum density | Same N: C = 0.270 = lattice. C ~ 0.28 + 0.12·log(density) | UV cutoff (a=1) sets C, not symmetry |

### For paper (before submission)

**Q — Fine α scan for κ(α).** α=0.35-0.50 for smooth figure. ~30 min.

**W4 — NN/NNN presentation decision.** NN default vs NNN default.
Flat integrand is NN-specific; holonomy identical; NNN adds ~3× coupling.

**Mechanism discussion rewrite (in §4.1/§4.2)** — old "incoherent" formulation
disproved. New: coherent scattering produces incoherent-like shape via non-Born
compensation. Per-bond is non-Born (file 50), ring collective is flat.
(Skeleton §4 numbering: §4.1 spectra, §4.2 κ(α), §4.3 AB, §4.4 application)

### Quick tests (low effort, useful for paper)

| Test | Description | Effort |
|------|------------|--------|
| F3d | α=0.5: verify σ_full = σ_diag (R(π)=-I, s_phi=0) | 5 min |
| P6 | σ_tr(α=0.5) maximum check | 5 min |
| O2 | DT=0.20,0.30 sensitivity | 5 min |
| O3 | sx=6,10 sensitivity | 5 min |
| O4 | r_m=15,20,25 extended | 5 min |
| T4 | Ring translation invariance (off-center) | 5 min |
| T5 | α_cross precision (α=0.29 or 0.31) | 5 min |
| R2 | k_min vs R (R=3,7) — lattice or geometric? | 15 min |
| S1 | Predict R=11, then measure | 15 min |

### From existing data (zero compute)

| Test | Description |
|------|------------|
| F20 | Non-Born ratio σ(K=-0.5)/σ(K=-1.3) vs Born — k-dependent? (file 37) |
| B5 | p(α-exponent) vs kR collapse from R=1,3,5,7,9,15 data |
| G3 | N_bonds vs σ_tr: ∝N or ∝N²? (existing data) |
| P3 | Multipolar decomposition C_l(α) from sphere data — Born→non-Born transition |
| P5 | Angular decorrelation vs kR — stationary phase confirmation |

### Medium effort (enrich paper)

| Test | Description | Effort |
|------|------------|--------|
| F9 | Uniform holonomy over sphere surface (topology test) | 20 min |
| F10 | Ring +z incidence (axial, holonomy inaccessible) | 15 min |
| F21 | Form factor |F(q)|² of Dirac disk numerically | 1h |
| F23 | Sphere displacement dx=0 only (strain null + displacement on sphere) | 20 min |
| P1 | Polarization-resolved σ_xx, σ_xy from existing data | 30 min |
| P4 | Dirac surface locality (finite disk R_disk) | 20 min |
| T8 | Time-resolved σ_tr(t) — scattering timescale | 30 min |
| N2 | Open arc vs closed ring — does closure matter? | 30 min |

---

## Parked

| Direction | Why |
|-----------|-----|
| E3 (holonomy measurement) | NNN is default, NN is comparison |
| O (thermal noise) | Trivial (difference cancels noise) |
| Test B (L=120 at k=1.9) | Less relevant without foam BZ |
| Route 16 (3-comp rotation) | Too complex, doesn't change O(κ) |
| Route 10 (direct drag) | Full BZ coverage makes it less urgent |
| R=20 scaling test | Needs L≥200, too expensive |
| Defect shapes (ellipse, trefoil) | Future paper — tests universality |
| Two coaxial rings | Future paper — defect interaction |
| Multiple rings (defect gas) | Future paper — macroscopic transport |
| SU(2) holonomy | Separate paper |
| Ring thickness (2-bond) | Changes gauge construction |
| Other lattice types | Different project |
| Temperature-dependent γ(T) | Future paper |
| Moving vortex Doppler | Different physics |
| κ=1 exact (W22) | Requires: (1) α from holonomy/energy, (2) physical k-cutoff, (3) whether κ=1 is coincidence or constraint |
| Real foams (W22+) | Kelvin/WP/C15 with vectorial elasticity. New gauge_3d.py per geometry. Born -5/2 + MS +1/2 direction likely universal (geometric), but exact κ value needs recalcul: different v_g (no cos²(k/2) cancellation), different Peierls coupling (connectivity ≠ 2), bending/twisting modes. Separate project |

---

## Paper Status

**Skeleton:** `11_paper_skeleton.md`

### Ready to write

| Section | Status | Notes |
|---------|--------|-------|
| §1 Introduction | Ready | Lead with σ_tr spectrum, not κ |
| §2 Setup | Ready | NNN default |
| §3 Drag formula | Ready | Unchanged |
| §4.1 σ_tr(k) spectra | Ready | Principal result |
| §4.2 κ(α) curve | Ready | Includes flat integrand discussion |
| §4.3 AB comparison | Ready | All 3 axes fail |
| §4.4 κ(α) as application | Ready | §6.3 connection |
| Mechanism discussion | **NEEDS REWRITE** | In §4.1/§4.2. Old "incoherent" disproved → collective ring (files 48-50) |
| §5 Systematics | Ready | K1/K2 done (file 29), angular grid done (file 33) |
| §6.1-6.2 Discussion | Ready | |
| §6.3 Stochastic mechanics | Ready | κ=1 chain as application |

### Figures needed

1. σ_tr(k) spectra — 4 α values, full BZ (principal figure)
2. sin²(k)·σ_tr integrand — flat at α≥0.2
3. κ(α) curve — NNN default, NN comparison (needs Q for smooth curve)
4. σ_tr vs R — log-log, R^{3/2} fit, kR crossover
5. Gauge invariance — spread vs k

---

## File Index

### Core infrastructure
| File | Content |
|------|---------|
| elastic_3d.py | Scalar Laplacian, PML, FDTD |
| scattering_3d.py | Wave packets, sphere integration |
| gauge_3d.py | Peierls vortex ring (NN + NNN) |

### Completed routes (chronological, grouped)

**σ_tr spectrum + κ (files 17-28):**
17-20: R-scaling, 7-13 k-pts. 23: 13 k-pts NN+NNN.
25: gauge violation. 26: Born limit (weak α). 27: R^{3/2} factorization.
28: fine α scan (NN+NNN, 7 k-pts).

**Validation (files 29-33):**
29: K₁/K₂ sensitivity. 30: Born R=1. 31: flux tube AB.
32: 2D monopole + convergence. 33: α↔1-α symmetry + angular grid.

**Mechanism — coupling × geometry (files 34-42):**
34: mass sphere (NOT flat). 35: diagonal/off-diagonal decomposition.
36: decomposition α-scan. 37: displacement vs strain coupling.
38: ring rotation (strain null). 40: NNN selective gauging.
41: displacement K_eff scan. 42: sphere displacement + half-ring arc.

**Coherence investigation (files 43-50):**
43: Born form factor. 44: arc scaling. 45: additivity test (α=0.30).
46: additivity Born (α=0.10). 47: shape analysis.
48: cross-term spectral shape. 49: single bond CV.
50: diagonal-only single bond.

**Per-bond mechanism (files 51-56):**
51: σ_bond≈const, N_eff decomposition. 52: scalar T-matrix (Born, FAILS).
53: polarization decomposition (compensation FAILS). 54: vectorial T-matrix (FAILS).
55: Born mechanism v3 — per-bond Born + collective N_eff → flat ring.
56: N_eff structure — 1/k² scaling, Born vs FDTD, global model N²/(1+c·k²·R^q).

---

## Relations

**Earlier κ bridge** (`release/3_bombardment/8_kappa_bridge.md`): 2D surrogate.
W21 supersedes — AB factorization fails at 44%.

**ST_ model connection.** W21 computes the missing piece in the bombardment chain
(Sector 3, `release/3_bombardment/1_bombardment.md`):
  γ = κc/R → D = ℏ/(2κm) → Schrödinger.
The prefactor κ depends on σ_tr(k) of the Peierls vortex ring — the lattice
realization of the Z₂ disclination loop (`release/2_particles/3_disclination_loop.md`).
Old κ bridge used 2D AB surrogate; W21 computes 3D directly.
Key ST_ files:
  - `release/3_bombardment/1_bombardment.md` §2: γ=κc/R derivation, §4: D=ℏ/(2κm)
  - `release/3_bombardment/8_kappa_bridge.md`: old factorized κ (superseded by W21)
  - `release/3_bombardment/6_P2b_bath_gauge_coupling.md`: α=1/2 from Z₂ holonomy
  - `release/2_particles/3_disclination_loop.md`: vortex ring construction, holonomy=-1
  - `release/0_project_status.md`: Sector 3 status, κ=1 prediction (C8)
The paper IS the mechanism: without understanding WHY sin²(k)·σ_tr ≈ const,
κ=O(1) is a numerical observation, not a derivable result.

---

*Mar 2026*
