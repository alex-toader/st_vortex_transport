# Test Reorganization Plan

**Date:** Mar 2026
**Source:** 61 files in `internal/old_tests/` (development order)
**Target:** `tests/` organized by paper section, with shared helpers

---

## Problems with old tests

1. **Every file duplicates** K1, K2, c_lat, BZ grid, disk_bonds(), cv(), FDTD config
2. **FDTD data hardcoded** — file 55 copies σ_bond from file 51, file 58 copies from file 18.
   If we re-run file 18 with a change, files 51, 55, 56, 58 all need manual updating.
3. **No assert** — most files print results but don't assert. A regression is invisible.
4. **Mixed concerns** — file 33 does both α↔1-α symmetry AND angular grid convergence.
5. **No imports between tests** — each is standalone. Good for exploration, bad for maintenance.

---

## Architecture

```
tests/
  helpers/
    __init__.py
    config.py           # K1, K2, c_lat, k_vals, standard FDTD params
    lattice.py          # BZ grid, omega_k2, dispersion, group velocity
    geometry.py         # disk_bonds(), ring_bonds(), random_disk()
    born.py             # Per-bond Born σ, monopole/dipole Z, form factors
    ms.py               # T=(I-VG)^{-1}V, N_eff, eigenvalue analysis
    fdtd.py             # Run FDTD + extract σ_tr (wraps parallel_fdtd)
    stats.py            # cv(), power law fit, log-log slope

  data/
    sigma_ring.py       # σ_tr(k) at R=3,5,7,9, α=0.30 (from file 18)
    sigma_bond.py       # σ_bond(k) at α=0.30 (from file 51)
    sigma_pol.py        # σ_xx, σ_xy per bond (from file 53)
    kappa_table.py      # κ(α) NN and NNN (from files 22, 23, 28)

  # By paper section:
  test_s2_infrastructure.py
  test_s4_spectrum.py
  test_s4_flat_integrand.py
  test_s4_kappa.py
  test_s5_systematics.py
  test_s6_born_perbond.py
  test_s6_born_disk.py
  test_s6_multiple_scattering.py
  test_s6_assembly.py
```

---

## helpers/ — what goes where

### config.py
```python
K1, K2 = 1.0, 0.5
c_lat = sqrt(K1 + 4 * K2)           # = sqrt(3)
ALPHA_REF = 0.30
R_LOOP = 5
L, DW, DS, DT = 80, 15, 1.5, 0.25
sx, r_m = 8.0, 20
N_THETA, N_PHI = 13, 24
k_vals_7 = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
k_vals_13 = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.4, 2.7, 3.0]
```

### lattice.py
- `omega_k2_grid(N_bz=64)` — BZ grid for G_00
- `dispersion(k)` — ω² = 2K1(1-cos k) + 4K2(2-2cos k)
- `group_velocity(k)` — v_g = c_lat * cos(k/2)
- `G_00(k, eps=0.005)` — self-energy from BZ sum

### geometry.py
- `disk_bonds(R)` — lattice points in disk of radius R
- `ring_bonds(R)` — perimeter only
- `random_disk(R, N, seed)` — uniform random in disk
- `annulus_bonds(R, width)` — annular region

### born.py
- `V_peierls(alpha)` — cm1, s_phi, V_eff
- `Z_monopole(k)`, `Z_dipole(k)` — angular integrals
- `sigma_bond_born(k, alpha)` — C₀ * V / v_g²
- `sigma_ring_born(k, R, alpha)` — coherent Born on disk

### ms.py
- `build_VG(dx, dy, k, alpha)` — V*G matrix
- `T_matrix(dx, dy, k, alpha)` — (I-VG)^{-1} V
- `sigma_ring_ms(k, R, alpha)` — MS σ_tr
- `N_eff_ms(k, R, alpha)` — N_eff from MS
- `eigenvalues_VG(dx, dy, k, alpha)` — spectrum of VG

### stats.py
- `cv(arr)` — coefficient of variation (%)
- `power_law_fit(x, y)` — log-log slope + residual
- `log_log_slope(k_arr, y_arr)` — exponent from fit

---

## data/ — FDTD reference values

Single source of truth. Each file has:
- The data arrays
- Source file reference
- Run conditions (R, L, α, gauging)

When we re-run FDTD, update HERE and all tests pick it up automatically.

---

## Test files — claims map

### test_s2_infrastructure.py (§2 Model)

| Claim | Assert | Old file |
|-------|--------|----------|
| Dispersion ω=2c sin(k/2) along 5 directions | rel_err < 1e-10 | 1, 2 |
| Speed spread < 0.2% over 1000 random dirs at k=1 | spread < 0.002 | 2 |
| Newton 3: F_ij = -F_ji | exact to 1e-15 | 2 |
| PML absorption > 99% | E_reflected/E_incident < 0.01 | 2 |
| FDTD energy conservation (no PML) | ΔE/E < dt² | 2 |
| α=0 → σ_tr=0 exactly at all k | σ_tr < 1e-10 | 24 |
| uz pure → σ_tr=0, uz+ux → uz unscattered | exact | 8 |
| α↔1-α symmetry to machine precision | rel_diff < 1e-14 | 33 |
| Gauge bond count: N_NN=81, N_NNN=393 at R=5 | exact | 4 |
| R(2πα) matrix correct | det=1, trace=2cos(2πα) | 4 |

~10 asserts. Mix of fast (analytic) + 2-3 FDTD runs (~2 min).

### test_s4_spectrum.py (§4.1 σ_tr spectra)

| Claim | Assert | Old file |
|-------|--------|----------|
| σ_tr(k) at α=0.30, R=5, 7 k-pts (NN) | match stored data ±1% | 18 |
| σ_tr(k) at α=0.10, 0.20, 0.50 (4 α) | match stored | 22 |
| σ_tr ~ R^{3/2}: exponent at R=3,5,7,9 | p ∈ [1.4, 1.7] | 18 |
| No oscillations (fine k-grid) | smooth | 19 |
| Polarization conversion uy/ux grows with k | monotonic | 6 |

FDTD-heavy. ~15-30 min total. Run infrequently.

### test_s4_flat_integrand.py (§4.2 Flat integrand)

| Claim | Assert | Old file |
|-------|--------|----------|
| sin²(k)·σ_tr CV < 10% at α=0.30, NN | CV < 0.10 | 17, 22 |
| CV > 30% at α=0.05 | CV > 0.25 | 26 |
| CV > 20% at all α, NNN gauging | CV > 0.20 | 28 |
| CV < 15% at K₁/K₂ = 1.5, 2.0, 3.0 | CV < 0.15 | 29 |

Uses stored data (fast) + 1-2 FDTD verifications.

### test_s4_kappa.py (§4.3 κ(α))

| Claim | Assert | Old file |
|-------|--------|----------|
| κ_NN(α=0.30, k≤1.5) ≈ 0.94 | ±5% | 22 |
| κ_NNN(α=0.30, k≤0.9) ≈ 1.02 | ±5% | 28 |
| κ monotonically increasing with α | diff > 0 | 22, 28 |
| AB fails: σ_tr ≠ 2sin²(πα)/k at any α | rel_diff > 20% | 31, 32 |

Mostly from stored data (fast).

### test_s5_systematics.py (§5 Systematics)

| Claim | Assert | Old file |
|-------|--------|----------|
| Gauge invariance: Dirac shift ±2, spread < 3% | spread < 0.03 | 25 |
| NNN/NN ratio: 1.3 (k=0.3) to 3.6 (k=1.5) | range check | 23 |
| Angular grid: 13×24 vs 25×48, diff < 2% | diff < 0.02 | 33 |
| n_steps scales with 1/v_g | correlation > 0.99 | 24 |

Mix: 1 FDTD (gauge shift test) + stored data.

### test_s6_born_perbond.py (§6.1 Per-bond Born)

| Claim | Assert | Old file |
|-------|--------|----------|
| σ_bond = C₀V/v_g², FDTD/Born CV < 6% | CV < 0.06 | 55 |
| Z_mono ≈ const (CV < 7%) | CV < 0.07 | 55 |
| Z_dipo grows 22× | ratio > 20 | 55 |
| At α≥0.25: V ≈ const (CV < 5%) | CV < 0.05 | 55 |
| |cm1|=|s_phi| at α=0.25 exactly | rel_diff < 1e-10 | 55 |

All analytic (0 seconds). Uses stored FDTD data for comparison.

### test_s6_born_disk.py (§6.2 Born exponent -5/2)

| Claim | Assert | Old file |
|-------|--------|----------|
| Born N_eff exponent → -5/2 at large R | p ∈ [-2.55, -2.45] at R=200 | 57, 60 |
| -5/2 = -3/2 (cone) + (-1) (transport) | decompose and verify | 60 |
| Universal: random disk ≈ lattice disk | ratio ∈ [0.8, 1.2] | 62 |
| Universal: square disk ≈ round disk | same exponent ± 0.1 | 57 |

Analytic (seconds). Matrix eigenvalue + Born integrals.

### test_s6_multiple_scattering.py (§6.3 MS correction)

| Claim | Assert | Old file |
|-------|--------|----------|
| MS exponent ≈ -2.0 at R=3,5,7,9 | p ∈ [-2.1, -1.9] | 58 |
| MS captures 90-105% of FDTD correction | ratio ∈ [0.9, 1.05] | 58 |
| CV(integrand): Born 35% → MS 10% | Born > 30%, MS < 15% | 58 |
| |λ_max| < 0.7 (not resonance) | max < 0.7 | 59 |
| Shift = C·|V|, C ∈ [0.25, 0.40] | range check | 59 |
| C is UV-cutoff dependent: C ~ log(density) | fit residual < 0.05 | 62 |
| Random disk p_enh/lattice ratio ∈ [0.8, 1.1] | ratio check | 62 |

Analytic (seconds). Matrix computations.

### test_s6_assembly.py (§6.4 Assembly)

| Claim | Assert | Old file |
|-------|--------|----------|
| cos²(k/2) cancellation is algebraic | identity check | 55 |
| sin²(k/2)·N_eff CV < 8% | CV < 0.08 | 55 |
| Flat integrand = Born -5/2 + MS +0.5 | decompose, total ≈ -2.0 | 60 |
| Balance at α ≈ 0.29: p_MS = -2.0 | crossing in [0.25, 0.35] | 60 |
| Residual CV ≈ 10%: decompose sinc² + non-power-law | CV components | 60 |

Analytic. Combines results from other test modules.

---

## Negative tests — "nu e artefact"

Reviewer-ul va suspecta: flat integrand e artefact numeric, bug de pipeline, sau
rezultat trivial. Fiecare ipoteză alternativă are un test negativ care o DISPROVE.

### test_null_pipeline.py — Pipeline nu produce flat din nimic

| Hypothesis | Test | Expected | Old file |
|------------|------|----------|----------|
| Pipeline artifact | α=0 → σ_tr at all k | σ_tr = 0.000000 (exact) | 24 |
| Polarization leak | uz pure wave → scattered field | σ_tr = 0 (exact) | 8 |
| Recording artifact | Mixed uz+ux → uz component | uz_scattered = 0 (exact) | 8 |
| n_steps insufficient | E_tail/E_peak at high k | < 5% at k ≤ 2.1 | 24 |

If pipeline had a bug producing flat integrand, α=0 would show it.

### test_null_geometry.py — Flat requires ring, not any defect

| Hypothesis | Test | Expected | Old file |
|------------|------|----------|----------|
| Any defect gives flat | Mass sphere (strain, 1302 bonds) | CV = 89-152% (NOT flat) | 34 |
| Any geometry gives flat | Sphere displacement (81 bonds) | CV = 35-46% (NOT flat) | 42 |
| Single bond gives flat | 1 Peierls bond, α=0.30 | CV = 71% (NOT flat) | 49 |
| Line gives flat | 81 bonds in a line | p_enh saturates at +0.12 | 61 |
| Annulus gives flat | Ring perimeter only | p_enh saturates at +0.11 | 61 |

Flat integrand requires filled disk topology. Sphere, line, annulus, single bond: all fail.

### test_null_coupling.py — Flat requires displacement + strong coupling

| Hypothesis | Test | Expected | Old file |
|------------|------|----------|----------|
| Strain coupling works | Strain on ring (∂u) | σ_tr = 0 (null, no signal) | 37 |
| Strain on yz-ring | Ring axis along propagation | CV = 42.5% (NOT flat) | 38 |
| Weak coupling flat | α=0.05, NN | CV = 34% (NOT flat) | 26 |
| Born gives flat | Born integrand (analytic) | CV = 35% (NOT flat) | 57 |

Strain coupling is null on the ring. Displacement coupling + α ≥ 0.2 required.

### test_null_gauging.py — Flat is NN-specific, not universal

| Hypothesis | Test | Expected | Old file |
|------------|------|----------|----------|
| NNN also flat | NNN gauging, α=0.30 | CV = 24% (NOT flat) | 28 |
| NNN flat at any α | NNN α=0.10-0.30 | CV = 25-35% at all α | 28 |
| AB prediction works | 2D AB: σ=2sin²(πα)/k | Fails k-axis (44%), α-axis, R-axis | 31, 32 |

Both gaugings give identical holonomy. But only NN gives flat integrand.
This proves flat is a lattice-scale coupling property, not topological.

### test_null_mechanism.py — Not T-matrix, not resonance

| Hypothesis | Test | Expected | Old file |
|------------|------|----------|----------|
| Scalar T-matrix explains σ_bond | |DK·G| at all α | ≤ 0.21 → Born (FAILS) | 52 |
| Vectorial T-matrix explains σ_xx | |DK·G|_eig | ≤ 0.24 → Born (FAILS) | 54 |
| Resonance (|λ|→1) | |λ_max| of VG | 0.3-0.6 (far from 1) | 59 |
| Compensation σ_xx+σ_xy | σ_xx ≈ const, σ_xy grows 50× | No compensation (FAILS) | 53 |

Single-site T-matrix (scalar or vectorial) cannot explain the non-Born behavior.
The effect is collective multiple scattering, not resonance.

---

**Total negative tests: ~20 asserts across 5 files.**

Each rules out a specific alternative explanation.
A reviewer who reads these cannot argue "it might be an artifact" —
every plausible artifact hypothesis has been explicitly tested and disproved.

---

## Execution plan

**Phase 1 — helpers/ + data/** (foundation)
  Write config, lattice, geometry, stats, born, ms.
  Write data files with stored FDTD values.
  ~500 lines total. No tests yet, just infrastructure.

**Phase 2 — analytic + negative tests** (fast, verify immediately)
  test_s6_born_perbond.py — per-bond Born (0s)
  test_s6_born_disk.py — Born -5/2 (seconds)
  test_s6_multiple_scattering.py — MS (seconds)
  test_s6_assembly.py — assembly (0s)
  test_null_coupling.py — strain null, weak coupling not flat (0s, uses data)
  test_null_mechanism.py — T-matrix fails, not resonance (seconds)
  test_null_gauging.py — NNN not flat, AB fails (0s, uses data)
  All run in < 15 seconds combined.

**Phase 3 — infrastructure + pipeline null tests** (mix fast + FDTD)
  test_s2_infrastructure.py — dispersion, PML, gauge construction
  test_null_pipeline.py — α=0, uz=0, recording completeness (FDTD)
  Fast parts first, then FDTD parts.

**Phase 4 — results + systematics + geometry nulls** (FDTD-heavy)
  test_s4_spectrum.py — requires FDTD runs (~15 min)
  test_s4_flat_integrand.py — mostly stored data
  test_s4_kappa.py — stored data
  test_s5_systematics.py — 1 FDTD + stored data
  test_null_geometry.py — sphere, line, annulus NOT flat (mix: data + FDTD)

**Phase 5 — figure generation**
  test_figures.py or make_figures.py — produces all paper figures.
  Not assertions, just output.

---

## Naming convention

- `test_s2_` = §2 Model
- `test_s4_` = §4 Results
- `test_s5_` = §5 Systematics
- `test_s6_` = §6 Mechanism
- `test_null_` = Negative tests (disprove alternative hypotheses)
- helpers: snake_case, descriptive
- data: module name = what it stores

---

## Counts

| Category | Old files | New files | Claims |
|----------|-----------|-----------|--------|
| Infrastructure | ~10 (files 1-5, 8-9, 24, 33) | 1 | 10 |
| Spectrum + κ | ~15 (files 6-7, 10, 14-23, 28) | 3 | 14 |
| Systematics | ~5 (files 25-26, 29-32) | 1 | 4 |
| Mechanism | ~20 (files 34-62) | 4 | 22 |
| **Negative / null** | **scattered across many** | **5** | **~20** |
| **Total** | **~50 relevant** | **14 test files** | **~70 asserts** |

Plus ~6 helper modules + ~4 data modules. Total new code: ~2500 lines
(vs ~8000 in old tests, much of it duplicated).

---

*Mar 2026*
