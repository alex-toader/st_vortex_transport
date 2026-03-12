# W21: κ from First-Principles Scattering — Tracker v5

**Date:** Mar 2026
**Previous:** `0_tracker_v4.md` (full history, routes 1-62, files 17-62)

---

## Architecture

Nivel 1 — FDTD engine (src/, fără dependențe interne)
1. src/elastic_3d.py — laplacian, damping, FDTD loop
2. src/scattering_3d.py — wave packet, sphere measurement, sigma integration
3. src/gauge_3d.py — vortex force (importă elastic_3d)

Nivel 2 — helpers (analitic, fără FDTD)
4. tests/helpers/config.py — constante (K1, K2, k_vals, α)
5. tests/helpers/lattice.py — dispersie, vg (importă config)
6. tests/helpers/stats.py — cv, log_log_slope (standalone)
7. tests/helpers/geometry.py — disk_bonds, line_bonds (standalone)
8. tests/helpers/born.py — Peierls, Born σ (importă config, lattice)
9. tests/helpers/ms.py — Green, T-matrix (importă config, lattice, born)

Nivel 3 — data (generat de FDTD, importă doar numpy)
10. tests/data/sigma_bond.py — FDTD single z-bond
11. tests/data/sigma_ring.py — FDTD ring at various R, α
12. tests/data/kappa_table.py — κ(α) post-processing

Nivel 4 — teste (192 tests, ~13s + 6 min FDTD)
13. test_1 — Born per-bond (21 tests)
14. test_2 — Born disk exponent (11 tests)
15. test_3 — multiple scattering (31 tests)
16. test_4 — assembly (14 tests)
17. test_5 — coupling requirements (12 tests)
18. test_6 — mechanism elimination (25 tests)
19. test_7 — null gauging (13 tests)
20. test_8 — FDTD convergence (11 tests, ~6 min)
21. test_9 — forward decoherence mechanism (42 tests)
22. test_10 — forward decoherence phase 2 (12 tests)

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

---

## Mechanism Chain (complete, tests 1-9)

### Step 1 — Per-bond coupling (DERIVED)
σ_bond = C₀ × [cm1²·Z_mono + s_phi²·Z_dipo] / v_g²
Born with v_g normalization. FDTD/Born CV = 4.6%. No free parameters.
At α ≥ 0.25: monopole dominates, V ≈ const(k) (CV < 3%).

### Step 2 — Disk geometry → Born -5/2 (ANALYTIC)
N ≈ πR² bonds. Forward cone: Born N_eff ~ (kR)^{-5/2} (transport).
-5/2 = -3/2 (asymmetric cone area) + (-1) (transport weight).
Verified: R=200, p = -2.491 (0.4% from -5/2).

### Step 3 — Multiple scattering → +1/2 correction (NUMERICAL)
T = (I - VG)^{-1}V shifts Born exponent from -5/2 toward -2.
Shift = C × |V|, C ≈ 0.34 ± 0.02 (R-independent, transport-weighted).
MS captures 91-104% of FDTD at all R=3,5,7,9.
Integrand CV: Born 35% → MS 10% → FDTD 7%.

### Step 4 — Flat integrand (PARTIAL)
sin²(k) · σ_ring = sin²(k) · σ_bond · N_eff.
cos²(k/2) cancels between sin²(k) and 1/v_g² — ALGEBRAIC IDENTITY.
Remaining: 4sin²(k/2) · V · N_eff ≈ const.
V ≈ const at α≥0.25: DERIVED (monopole dominance).
sin²(k/2) · N_eff ≈ const: NUMERICAL (from Steps 2+3, not independent).

### Step 5 — κ = O(1) (NUMERICAL)
κ(α=0.30) = 1.02 (k ≤ 0.9), 2.49 (k ≤ 1.5).
Threshold: |V| = (p_Born - 2)/C = 1.26 → α = 0.29 (self-consistent).

| Step | Status |
|------|--------|
| 1. Per-bond Born σ_bond = C₀V/v_g² | DERIVED (verified 4.6%) |
| 2. Born exponent -5/2 from disk | ANALYTIC (cone -3/2 + transport -1) |
| 3. MS shift +1/2 from T=(I-VG)^{-1}V | NUMERICAL (C=0.34 not analytic) |
| 4. cos²(k/2) cancellation + N_eff≈k^{-2} | PARTIAL (cancellation=identity, N_eff=numerical) |
| 5. κ = O(1) at α ≈ 0.30 | NUMERICAL (from steps 1-4) |

---

## Forward Decoherence (test_9, 42 tests)

### What is established

1. **T-matrix decomposition**: diagonal enhances (|T/V|²≈1.35), off-diagonal
   destructively interferes. Cross term = -137 at k=0.3 (73% of diagonal).
2. **Born series alternation**: VG coherence 0.83 (k=0.3) → 0.20 (k=1.5).
   cancel_ratio 0.37 → 0.93. cross_12 = -11% → +79%.
3. **Path excess phases**: frac(k_eff·pe > π) = 7.5% (k=0.3) → 68% (k=1.5).
   Geometric origin of cross_12 sign change.
4. **Angular uniformity**: MS/Born ratio CV=4.5% at k=0.3 (uniform rescaling).
   Transport/total diff = 2.8% (weight-independent at low k).
5. **Transport shift R-independent**: +0.45 ± 0.03, CV=4.2% from R=3 to R=15.
6. **Shift ∝ |V|**: C = shift/|V| = 0.334, CV=3.7% over α=0.15–0.40.
7. **Phase regularization**: C_full R-independent (0.32±0.02), C_static fluctuates.
8. **Geometry**: C_disk > C_annulus > C_line (3.6×). Interior matters.
9. **Eigenvalue clustering**: |λ| CV=41% (k=0.3) → 24% (k=1.5).
10. **cos(Φ) → const**: exponent -0.35 (R=3) → -0.05 (R=9).

### What is NOT derived

- **C = 0.34**: empirical, no analytic derivation
- **Transport shift R-independence**: verified, not derived
- **Exponent -2**: approximate (-1.97 at α=0.30), not geometric

### Power decomposition chain (F10)

cross/born = 2 × diag_amp(k^+0.01) × offdiag_amp(k^-0.48) × cos(Φ)(k^-0.16)
Sum = -0.63 ✓. offdiag_amp drifts from -0.47 to -0.53 at R=15.

---

## Phase 2 — Forward Mechanism Gaps & Extensions

Tests: test_10_forward_p2.py (8 tests)

### A. Paper-critical gaps

**A1. κ MS vs FDTD — N_eff mechanism validated (tested)**
MS σ_tr is ~15× FDTD (varies 8-20× with k). But the gap is entirely per-bond
normalization (scalar MS vs vector FDTD). The collective N_eff is correct:
N_eff_MS/N_eff_FDTD = 1.05 (mean), CV=14%. Ring/bond gap ratio ≈ 1.0.
MS correctly predicts how N bonds interact; absolute σ has constant offset.
Tests: TestKappaConsistency (test_10, 2 tests).

**A4. Born exponent p_born vs -5/2 — DONE (tested)**
p_born = -2.55 from sigma_tr_born_ms, consistent with -5/2 from test_2.
Assert p_born ∈ [-2.8, -2.2] added to TestShiftMethodConsistency.
Links test_10 angular integration to test_2 Born exponent.

**A2. α=0.25 CV minim — CLOSED (negative result, tested)**
CV(sin²k·σ_ms) scade MONOTON cu α: 39.5% (0.20) → 26.9% (0.40).
V(k) CV=0 la α=0.25 NU domină integrandul total.
Coupling mai puternic → shift mai mare → mereu mai flat.
Test: TestIntegrandCVMonotonic (test_10).

**A3. Consistență shifts cu/fără F² — DONE (tested)**
Shifts: +0.480 (fără F²) vs +0.474 (cu F²). Diferență 1.2%.
F² = 4cos²(qz/2) adaugă ~k^{-2.3} la ambele → se anulează în shift.
N_eff absolut diferă ~130× dar shift invariant.
Test: TestShiftMethodConsistency (test_10).

### B. Investigații cu valoare fizică

**B1. Crossover k ~ R^p — DONE (tested)**
k* = 0.69 (R=3), 0.83 (R=5), 1.15 (R=7), 1.20 (R=9), 1.56 (R=12).
Fit: k* ~ R^{0.59}. Monotonic. La R≥25: k* → BZ edge.
Exponent 0.59 ≈ 3/5, posibil legat de cooperare range scaling.
Tests: TestCrossoverScaling (test_10, 2 tests).

**B2. Uniformitate angulară ca funcție continuă de k**
Status: OPEN
Doar k=0.3 (CV<5%) și k=1.5 (CV>10%) testate. La ce k CV trece prin ~7%?
Coincide cu crossover din F10? Leagă F2 de F10.

**B3. Path excess — 1/r weighting concentrează cooperare — DONE (tested)**
Raw distribution: median pe = 4.0 (NOT concentrated).
1/r³-weighted: median pe = 2.0, frac(pe<3) = 62.5% (vs 39.5% by count).
Cooperarea e locală nu din distribuția geometrică, ci din 1/r weighting în G.
Explică dc r_cut=3 converge (test_9 F8).
Tests: TestPathExcessConcentration (test_10, 2 tests).

**B4. Eigenvalue dominant — DONE (tested)**
k=0.3: |λ₁|/|λ₂| = 2.52 (single dominant mode, λ₂=λ₃ degenerat).
k=1.5: ratio = 1.01 (no dominant mode, uniform spectrum).
Justifică mean-field / single-mode picture la low k.
Tests: TestEigenvalueDominance (test_10, 2 tests).

**B6. σ_tr ~ R^{3/2} Born + MS R-scaling — DONE (tested)**
Born σ_tr ~ R^p at fixed k: p ≈ 1.5 (near 3/2 from stationary phase).
MS R-scaling is LOWER than Born (p_ms < p_born at all k): 1.35 vs 1.53 (k=0.5),
1.27 vs 1.49 (k=0.9). Forward decoherence strengthens with R → slower σ growth.
FDTD gives p ≈ 1.63 (higher than both), not captured by MS.
Tests: TestRScaling (test_10, 2 tests).

### C. Consolidare internă

**C1. _make_dOmega → make_dOmega (public)** — DONE
Renamed in ms.py. test_10 imports make_dOmega. test_3 updated.

**C2. Helper _compute_angular_ratio_cv**
Status: OPEN (low priority)

**C3. Refactoring _enh_exp_custom → ms.py**
Status: OPEN (low priority)

### Status

| Item | Status | Test |
|------|--------|------|
| A1 (κ N_eff validation) | DONE | TestKappaConsistency |
| A2 (CV vs α) | DONE | TestIntegrandCVMonotonic |
| A3 (shift consistency) | DONE | TestShiftMethodConsistency |
| A4 (p_Born vs -5/2) | DONE | TestShiftMethodConsistency |
| B1 (crossover ~ R^p) | DONE | TestCrossoverScaling |
| B2 (angular CV vs k) | OPEN | — |
| B3 (path excess) | DONE | TestPathExcessConcentration |
| B4 (eigenvalue dominant) | DONE | TestEigenvalueDominance |
| B6 (σ_tr ~ R^{3/2}) | DONE | TestRScaling |
| C1 (make_dOmega public) | DONE | — |
| C2 (helper angular) | OPEN | — |
| C3 (refactor enh_exp) | OPEN | — |

---

## Data Tables

### κ(α) — NN gauging (from FDTD)

| α | κ_NN(k≤0.9) | κ_NN(k≤1.5) | κ_NN(k≤2.1) | κ_NN(k≤3.0) |
|---|------------|-------------|-------------|-------------|
| 0.10 | 0.025 | 0.056 | — | — |
| 0.15 | 0.084 | 0.164 | — | — |
| 0.20 | 0.189 | 0.355 | — | — |
| 0.25 | 0.333 | 0.625 | 0.898 | 1.211 |
| 0.30 | 0.495 | 0.944 | 1.341 | 1.752 |
| 0.40 | — | 1.545 | 2.188 | 2.760 |
| 0.50 | — | 1.798 | 2.552 | 3.185 |

### κ(α) — NNN gauging

| α | κ_NNN(k≤0.9) | κ_NNN(k≤1.5) | κ_NNN(k≤2.1) |
|---|-------------|--------------|--------------|
| 0.10 | 0.099 | 0.266 | — |
| 0.15 | 0.240 | 0.621 | — |
| 0.20 | 0.451 | 1.136 | — |
| 0.25 | 0.720 | 1.782 | 2.783 |
| 0.30 | 1.018 | 2.490 | 3.845 |
| 0.40 | — | 3.722 | 5.650 |
| 0.50 | — | 4.205 | 6.339 |

### Flat integrand CV

| α | CV_NN | CV_NNN | Verdict |
|---|-------|--------|---------|
| 0.05 | 34.2% | — | NOT flat |
| 0.10 | 15.3% | 34.9% | NN marginal |
| 0.20 | 15.5% | 27.7% | NN marginal |
| 0.25 | 11.9% | 25.5% | NN roughly flat |
| 0.30 | 7.5% | 24.2% | NN flat |
| 0.40 | 3.9% | — | NN very flat |
| 0.50 | 5.6% | — | NN flat |

### Systematics on κ

| Source | Effect on κ | Status |
|--------|-----------|--------|
| NN vs NNN gauging | 2.1-4.8× | **Dominant** |
| k-cutoff (k≤0.9 vs k≤1.5) | 1.5-2.4× | Large |
| Near-field (r_m) | ~7% | Small |
| Gauge violation (Dirac shift) | < 2.1% (NNN) | Small |

---

## Pre-submission Items

### Ready to write

| Section | Status |
|---------|--------|
| §1 Introduction | Ready |
| §2 Setup | Ready |
| §3 Drag formula | Ready |
| §4.1 σ_tr(k) spectra | Ready |
| §4.2 κ(α) curve | Ready |
| §4.3 AB comparison | Ready |
| §4.4 κ(α) as application | Ready |
| §5 Systematics | Ready |
| §6 Discussion | Ready |
| Mechanism discussion | Ready (forward decoherence language in place) |

### Figures needed

1. σ_tr(k) spectra — 4 α values (principal figure)
2. sin²(k)·σ_tr integrand — flat at α≥0.2
3. κ(α) curve — NNN default, NN comparison
4. σ_tr vs R — log-log, R^{3/2} fit
5. Gauge invariance spread

### Pre-submission tests (from v4, still valid)

| Test | Description | Effort |
|------|------------|--------|
| Q | Fine α scan κ(α) for smooth figure | 30 min |
| W4 | NN/NNN presentation decision | decision |
| F3d | α=0.5: σ_full = σ_diag (s_phi=0) | 5 min |
| T5 | α_cross precision (α=0.29 or 0.31) | 5 min |

---

## Old Investigations (from v4, DONE or superseded)

I11-I18: all DONE (files 51-55). Per-bond Born mechanism resolved.
I3, I2, I4: g(k) investigations — superseded by forward decoherence (test_9).
I13: open arc — still valid but low priority.
I5, I6, I8, I9, I10: FDTD investigations — parked.
Routes 22a-e: all RESOLVED (files 55-62).
T7/T1, F17: CLOSED (T-matrix route eliminated).

For full details see `0_tracker_v4.md`.

---

## Parked

| Direction | Why |
|-----------|-----|
| Real foams (W22+) | Separate project — different v_g, connectivity, modes |
| κ=1 exact (W22) | Needs α from holonomy + physical k-cutoff |
| Other lattice types | Different project |
| Defect shapes | Future paper |
| Two/multiple rings | Future paper |
| SU(2) holonomy | Separate paper |
| Moving vortex Doppler | Different physics |
| Open arc (I13) | Low priority — test_9 covers mechanism |
| Square ring (I6) | Low priority |
| Half-ring paradox (I10) | Low priority |

---

*Mar 2026*
