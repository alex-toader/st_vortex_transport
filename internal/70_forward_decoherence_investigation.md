# 70 — Forward Decoherence Investigation

Mar 2026

## Questions

1. **Why exponent -2?** Shift is +0.5 = exactly 1/2. Coincidence or geometric?
2. **Angular structure** — does MS broaden the cone or suppress the peak?
3. **R → ∞ limit** — does exponent saturate at -2 or keep shifting?

## Findings

### F1. Total power enhancement grows with R (no angular weighting)

Enhancement ratio Σ|Tb|²/Σ|Vb|² at α=0.30 (NO transport weight):

| R  | N    | enh(k=0.3) | enh(k=1.5) | slope_enh |
|----|------|------------|------------|-----------|
|  3 |   29 | 0.747      | 1.340      | +0.308    |
|  5 |   81 | 0.611      | 1.183      | +0.354    |
|  9 |  253 | 0.473      | 1.016      | +0.411    |
| 15 |  709 | 0.360      | 0.960      | +0.522    |
| 25 | 1961 | 0.265      | 0.970      | +0.716    |

Total power suppression at low k keeps growing. Slope keeps growing.
But this is NOT the transport-relevant quantity — see F4.

### F2. Angular structure — MS does NOT reshape the cone (Q2 answered)

At k=0.3, R=5: MS/Born ratio is CONSTANT at all θ (0.539–0.554).
MS suppresses uniformly in angle. Transport-weighted 50%/90% angles
identical for Born and MS.

"Forward decoherence" reduces the overall coherent amplitude uniformly,
NOT by reshaping the angular pattern.

### F3. T-matrix decomposition — the real mechanism

T = T_diag + T_offdiag. At k=0.3, R=5:
- T_diag: enhances (|T_jj/V|² ≈ 1.35) — self-energy
- T_offdiag: forward sum OPPOSES diagonal (+38 vs -90)
- Cross term in total power: -137 (negative, 73% of diagonal)

Mechanism:
1. Born vertex V = -1.31 (negative)
2. Self-energy: T_jj = V/(1-VG₀₀) ≈ -1.52 (still negative, larger |T|)
3. Rescattering: Σ_{j≠i} T_ij·b_j ∝ V²·G_offdiag > 0 (OPPOSITE sign)
4. Destructive interference between diagonal and off-diagonal amplitudes
5. Strongest at low k (phases aligned) → k-dependent suppression

Cross term at k=0.3 grows with R:
|cross|/N = 1.3(R=3) → 2.6(R=15). At R=15: |cross| > diagonal.

### F4. Transport weight makes the shift R-independent (KEY FINDING)

With CORRECT transport weight (1 - sinθ cosφ = 1 - cosθ_scattering):

| R  | N    | p_Born | p_MS   | shift  | integrand_CV |
|----|------|--------|--------|--------|-------------|
|  3 |   29 | -2.22  | -1.81  | +0.41  | 15.7%       |
|  5 |   81 | -2.42  | -1.97  | +0.45  | 12.5%       |
|  7 |  149 | -2.37  | -1.97  | +0.40  | 12.4%       |
|  9 |  253 | -2.41  | -1.99  | +0.41  | 13.2%       |
| 12 |  441 | -2.40  | -1.99  | +0.42  | 12.7%       |
| 15 |  709 | -2.42  | -1.97  | +0.45  | 11.9%       |

**Shift = +0.448 ± 0.03, R-INDEPENDENT.** Converged at all angular grids (N=50..200).

Physical picture:
- Total power: forward decoherence grows with R (F1), NOT saturating
- Transport cross-section: shift is CONSTANT because transport weight
  (1-cosθ_s) filters out the forward cone where Born/MS differ most
- σ_tr is robust; integrand CV is stable at ~12% across all R

**The flat integrand works at ALL R, not just R=5.**

### F5. Shift is linear in |V|, C ≈ 0.34

| α    | |V|    | p_MS   | shift  | C=shift/|V| |
|------|--------|--------|--------|-------------|
| 0.10 | 0.191  | -2.356 | +0.058 | 0.305       |
| 0.20 | 0.691  | -2.189 | +0.226 | 0.326       |
| 0.25 | 1.000  | -2.079 | +0.335 | 0.335       |
| 0.30 | 1.309  | -1.966 | +0.448 | 0.342       |
| 0.35 | 1.588  | -1.863 | +0.552 | 0.348       |
| 0.40 | 1.809  | -1.779 | +0.636 | 0.352       |
| 0.50 | 2.000  | -1.705 | +0.710 | 0.355       |

C slowly increases from 0.305 to 0.355. Not exactly constant, but close.

p_MS = -2.0 at α ≈ 0.28 (where shift = 0.415 = p_Born + 2.0).
The integrand is flattest at α ≈ 0.25-0.30 (combining N_eff shift with V(k) constancy).

### F6. T-matrix integrand CV is α-independent

T-matrix (scalar V, no V(k) variation) gives integrand CV ≈ 12.4% at
ALL α from 0.20 to 0.35. The α-dependence of FDTD integrand CV (3.7% at
α=0.25, 5.6% at α=0.30) comes from V(k) variation, not N_eff shift.

Two independent contributions to integrand flatness:
1. N_eff shift: +0.45 at α=0.30 (from MS) — gives CV~12%
2. V(k) constancy: CV=0% at α=0.25 (from monopole dominance)
Combined: partial cancellation → FDTD CV = 5.6%

### Q1 Summary: Why ≈ -2?

Exponent = -2.5 + C·|V(α)|. Not exactly -2. The shift +0.45 at α=0.30
gives exponent -1.97. Whether -2 is "exact" or approximate remains open.
The C coefficient (0.34) is empirical; no analytic derivation exists.

### F7. C depends on geometry: local coordination matters

| Geometry   | C      | <n(r≤3)> |
|------------|--------|----------|
| disk R=5   | 0.342  | 20.5     |
| disk R=7   | 0.307  | 22.6     |
| disk R=9   | 0.316  | 23.8     |
| line L=9   | 0.092  | 4.7      |
| line L=15  | 0.089  | 5.2      |
| annulus R=5| 0.109  | 5.8      |
| annulus R=9| 0.124  | 5.4      |

Disk C is 3.5× larger than line C. Roughly proportional to coordination
number within cooperation range (r≤3), with sublinear scaling C ~ n^0.86.

### F8. Phase oscillation makes C universal (R-independent)

| R  | C_full | C_static |
|----|--------|----------|
|  3 | 0.313  | 0.457    |
|  5 | 0.342  | 0.635    |
|  7 | 0.307  | 0.642    |
|  9 | 0.316  | 0.555    |
| 12 | 0.319  | 0.576    |

- C_full (with exp(ikr)): stable at 0.32 ± 0.02 — R-INDEPENDENT
- C_static (1/r only): fluctuates 0.46–0.64 — R-DEPENDENT
- Phase oscillation limits cooperation range → makes C a local property

### F9. Mean-field picture

Mean-field coupling |V·<Σ G_ij>| at each k:
- k=0.3: 0.67 (strong coupling → suppression)
- k=0.9: 0.22 (intermediate)
- k=1.5: 0.16 (weak → Born regime)

Coupling decreases as ~k^{-0.9} (phases cancel at large k).
This k-dependent coupling → k-dependent suppression → exponent shift.

Spectral radius of VG at k=0.3: 0.60 (close to mean-field: 0.67).
System is NOT resonant (all |λ| < 1), but NOT purely perturbative either.

## Summary of mechanism

1. Born vertex V < 0 (Peierls coupling)
2. Self-energy T_jj = V/(1-VG₀₀) enhances each bond (|T/V|² ≈ 1.35)
3. Off-diagonal rescattering V²G produces amplitudes with OPPOSITE sign to V
4. At low k: VG off-diagonal elements are COHERENT (coh=0.83) → Born series
   terms alternate systematically → destructive interference → suppress |R_ij|
   (cancel_ratio=0.37, cross_12 = -11%)
5. At high k: phases randomize (coh=0.20) → no alternation → Born terms add
   constructively → enhance |R_ij| (cancel_ratio=2.0, cross_12 = +79%)
6. This k-dependent |R_ij|² (incoh ~ k^{+0.48}) combines with Fresnel zone
   cooperation (N_coop ~ k^{-1.45}) to give offdiag_amp ~ k^{-0.48}
7. Cross term = diag_amp × offdiag_amp × cos(Φ), where cos(Φ) → const at
   large R. So cross ~ k^{-1/2} approximately.
8. enh = D - |cross| → positive slope → exponent shift from -5/2 toward -2
9. Transport weight (1-cosθ_s) filters forward cone → shift is R-independent
   even though underlying exponents (N_coop, incoh) keep evolving with R
10. Result: shift = C·|V| with C ≈ 0.34 (empirical coefficient)

### F10. Analytic chain: N_coop × incoh → cross → shift

Decompose (Tb)_i = T_ii·b_i (diagonal) + Σ_{j≠i} T_ij·b_j (off-diagonal).

Total power = diag_pow + offdiag_pow + cross. The cross term (destructive
interference between diagonal and off-diagonal) carries the k-dependence.

**The cross term decomposes as:**

cross/born = 2 × (diag_amp/√born) × (offdiag_amp/√born) × cos(Φ)

where Φ is the effective angle between diag and offdiag N-vectors.

Power law exponents at R=5, α=0.30:

| Quantity                    | Exponent | Meaning                          |
|-----------------------------|----------|----------------------------------|
| diag_amp/√born              | k^+0.01  | Self-energy, constant            |
| offdiag_amp/√born           | k^-0.48  | √(N_coop × incoh)               |
| \|cos(Φ)\|                  | k^-0.16  | Phase correlation (→ 0 at R→∞)  |
| \|cross\|/born              | k^-0.63  | = sum of three exponents above   |
| enh(k)                      | k^+0.35  | Total power enhancement          |

**offdiag_amp decomposes further:**

offdiag_pow = N_coop × N(N-1) × incoh. So offdiag_amp ∝ √(N_coop × incoh).

| Quantity | Exponent | Mechanism                                      |
|----------|----------|-------------------------------------------------|
| N_coop   | k^-1.45  | Fresnel zone cooperation (keeps growing with R) |
| incoh    | k^+0.48  | Born series alternation (keeps growing with R)  |
| product  | k^-0.97  | = offdiag_pow exponent ✓                        |

### F11. Born series alternation: the mechanism behind incoh

incoh = ⟨|T_ij|²⟩ = V² × ⟨|R_ij|²⟩ where R = (I-VG)^{-1} (resolvent).

Key facts:
- |G_ij|² is k-INDEPENDENT (= 0.0000886 at all k)
- All k-dependence of incoh comes from the RESOLVENT structure
- R_ij = (VG)_ij + (VG²)_ij + (VG³)_ij + ... (Born series)

Cancel ratio = ⟨|R_ij|⟩ / ⟨Σ_n |(VG)^n_ij|⟩ (resolvent vs sum of absolute terms):

| k   | cancel_ratio | VG coherence | interpretation               |
|-----|-------------|--------------|------------------------------|
| 0.3 | 0.37        | 0.83         | Strong cancellation          |
| 0.9 | 0.82        | 0.28         | Near-incoherent              |
| 1.5 | 0.93        | 0.20         | Constructive addition        |

At low k, exp(ikr) phases are aligned → VG off-diagonal elements are
coherent (coherence=0.83) → Born series terms alternate systematically
(since V<0: n=1 negative, n=2 positive, n=3 negative...) → destructive
interference → small |R_ij| → small incoh.

At high k, phases randomize → VG elements are incoherent (coherence=0.20)
→ no systematic alternation → constructive addition → large |R_ij|
→ large incoh.

The cross_frac = (|R|² - Σ|VG^n|²) / |R|² confirms:
- k=0.3: -0.35 (35% destructive interference between Born orders)
- k=0.9: +0.34 (34% constructive interference)
- k=1.5: +0.51 (51% constructive interference)

### F12. Convergence with R

| R  | N    | p_Ncoop | p_incoh | p_cosΦ | p_offamp | p_cross | p_enh  |
|----|------|---------|---------|--------|----------|---------|--------|
|  3 |   29 |  -1.351 |  +0.410 | -0.349 |   -0.470 |  -0.806 | +0.308 |
|  5 |   81 |  -1.449 |  +0.480 | -0.155 |   -0.484 |  -0.627 | +0.354 |
|  7 |  149 |  -1.466 |  +0.511 | -0.088 |   -0.478 |  -0.554 | +0.380 |
|  9 |  253 |  -1.497 |  +0.532 | -0.048 |   -0.482 |  -0.519 | +0.411 |
| 12 |  441 |  -1.553 |  +0.552 | -0.023 |   -0.500 |  -0.512 | +0.461 |
| 15 |  709 |  -1.628 |  +0.567 | -0.011 |   -0.531 |  -0.531 | +0.522 |

Key convergence results:
- **cos(Φ) → 0**: rapidly, by R=12 already -0.023. The effective phase
  between diag and offdiag becomes k-independent.
- **p_Ncoop + p_incoh drifts**: -0.94 (R=3) → -1.06 (R=15). Both keep
  growing in magnitude. offdiag_amp drifts from -0.47 to -0.53.
  The apparent stability at -0.50 was a coincidence at R=5-9.
- **p_enh keeps growing**: total power shift is NOT R-independent (→ F1).
  Only the transport-weighted shift converges (→ F4).
- Neither N_coop nor incoh individually converge to simple fractions.
  The +1/2 for incoh and -3/2 for N_coop are approximate, not exact.

### F13. Cross_12 sign change: microscopic origin of incoh k-dependence

The leading Born correction to ⟨|R_ij|²⟩ is cross_12 = 2Re⟨(VG)_ij* (VG²)_ij⟩.
This involves the 3-point Green function correlator G_ij* G_il G_lj, whose
phase is k_eff × (r_il + r_lj - r_ij) = k_eff × Δ_path (path excess).

By triangle inequality, Δ_path ≥ 0. At low k, k_eff × Δ ≈ 0 → cos ≈ 1,
but since V < 0, V³ < 0, making cross_12 < 0 (DESTRUCTIVE).
At high k, the phase oscillates → average becomes positive → cross_12 > 0.

| R  | k_sign_chg | cross_12/|VG|² at k=0.3 | at k=1.5 |
|----|------------|-------------------------|----------|
|  3 | no change  | +0.032                  | +0.829   |
|  5 | k=0.35    | -0.107                  | +0.789   |
|  7 | k=0.38    | -0.165                  | +0.758   |
|  9 | k=0.41    | -0.235                  | +0.724   |
| 12 | k=0.44    | -0.317                  | +0.684   |

The sign change shifts to higher k as R grows (larger ⟨Δ_path⟩ → lower k
threshold for destructive phase). At R=3, all Born corrections are
constructive; the destructive regime appears only for R ≥ 5.

Perturbation theory to order 2 captures ~60% of the incoh exponent
(0.31 vs 0.48 at R=5). Higher Born orders amplify the cross_12 pattern.

### Status of the analytic chain

**DERIVED (geometric/structural):**
1. Direction of shift (positive) — from cross < 0 and |cross| decreasing
2. incoh mechanism — Born series alternation from VG phase coherence:
   cross_12 sign change from destructive (low k) to constructive (high k)
3. cos(Φ) → const — phase between diag/offdiag stabilizes at large R
4. Angular uniformity — MS rescales uniformly, doesn't reshape cone

**SEMI-QUANTITATIVE (mechanism clear, exponent approximate):**
5. N_coop ~ k^{-3/2} at moderate R (but keeps growing at large R)
6. incoh ~ k^{+1/2} at moderate R (keeps growing, not exact 1/2)
7. offdiag_amp ≈ k^{-1/2} at R=5-9 (drifts to -0.53 at R=15)
8. cross ≈ k^{-1/2} (follows from 7 + cos→const)
9. Total power Δp ~ 0.3-0.5 depending on R, not converging

**EMPIRICAL:**
10. C = 0.34 (exact coefficient in Δp = C|V|)
11. Transport shift R-independence (1-cosθ filtering compensates
    for drift of underlying exponents, no quantitative derivation)
12. Perturbation theory (order 2) gives ~60% of incoh exponent

### Status of questions

Q1 (why -2?): Exponent = -5/2 + C|V|. C ≈ 0.34 from Fresnel zone
cooperation. Not exactly -2 — it's -1.97 at α=0.30. The value -2 is
approximate, not geometric.

Q2 (angular structure): ANSWERED. MS rescales uniformly, no cone reshaping.

Q3 (R→∞ limit): Total power shift keeps growing (F1). Transport shift
is R-INDEPENDENT at +0.45 (F4). Flat integrand works at all R.

## Tests consolidated

File: tests/test_9_forward_mechanism.py (33 tests, all passing)

| Class                          | Tests | What it verifies                    |
|--------------------------------|-------|-------------------------------------|
| TestAngularUniformity          | 2     | MS/Born ratio constant (CV=4.5%)    |
| TestTMatrixDecomposition       | 4     | Diag enhances, cross term negative  |
| TestTransportShiftRIndependence| 3     | Shift +0.45 R-independent, CV~12%   |
| TestGeometryDependence         | 3     | C_disk > C_annulus > C_line (3.6×)  |
| TestCFullVsCStatic             | 4     | Phases reduce C, convergence r=3    |
| TestMeanFieldCoupling          | 3     | Coupling ~k^{-0.94}, no resonance   |
| TestPowerDecomposition         | 4     | Exponents sum, offdiag≈-0.5, enh>0  |
| TestBornSeriesAlternation      | 4     | |G|²=const, cancel ratio, coherence |
| TestCosPhiConvergence          | 2     | cos(Φ) exp→0 with R                 |
| TestCross12SignChange           | 4     | cross_12 sign change, fraction >0.5 |

Total test suite: 160 tests passing (tests 1-7 + 9, excluding 8=FDTD).

## Key results for paper

1. **T-matrix decomposition**: diagonal enhances, off-diagonal destructively
   interferes. This is the microscopic mechanism.
2. **Transport weight robustness**: shift is R-independent because transport
   weight filters the forward cone. Flat integrand works at all R.
3. **Shift linearity**: Δp = C|V|, C ≈ 0.34 (slowly varying).
4. **Angular uniformity**: MS doesn't reshape the cone, just rescales amplitude.
5. **Phase regularization**: oscillating exp(ikr) limits range → C is local/universal.
6. **Geometry dependence**: C scales with local coordination (disk > annulus > line).
7. **Fresnel zone scaling**: N_coop ~ k^{-3/2} (geometric, same as Born σ_tot).
   The -3/2 exponent connects the shift to 2D disk Fresnel zone geometry.
8. **Born series alternation**: the cross_12 term 2Re⟨VG*·VG²⟩ switches sign
   from -11% (k=0.3, destructive) to +79% (k=1.5, constructive).
   This drives incoh's k-dependence. VG coherence: 0.83 (k=0.3) → 0.20 (k=1.5).
9. **offdiag_amp ≈ k^{-1/2} at moderate R** (R=5-9, ±0.03), from
   N_coop × incoh balance. Drifts to -0.53 at R=15 — not exactly -1/2.
10. **cos(Φ) → const at large R**: the effective diag-offdiag phase becomes
    k-independent, simplifying the chain to cross ≈ offdiag_amp.
11. **R-independence of transport shift**: the underlying exponents (N_coop,
    incoh) do NOT individually converge, but the transport weight (1-cosθ)
    compensates for their drift. This is empirically verified but not derived.
