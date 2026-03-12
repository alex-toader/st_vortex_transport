# Mechanism Story — Simple Version

**Mar 2026**

---

## What happens physically

1. Lattice (foam). Vibrates thermally — phonons.

2. Topological defect: disclination loop. The loop cuts the lattice on a disk
   (Dirac disk). On the disk, bonds are rotated by angle 2πα.

3. A phonon (wave) arrives. Hits the disk. Each rotated bond deflects
   the wave a little. This is **scattering** — the "bombardment".

4. If each bond scattered **independently** (Born), you get σ_tr ~ k^{-5/2}.
   This gives an integrand that diverges — some wavelengths dominate,
   others contribute nothing. κ would be controlled by a single regime.

5. But bonds do NOT scatter independently. A phonon deflected by bond A
   propagates (Green function ~ 1/r) to bond B, deflects again, reaches C, etc.
   This is **multiple scattering** — collective cooperation.

6. The cooperation changes the exponent: -5/2 → -2.
   With -2, the integrand sin²(k)·σ_tr becomes flat.
   **All** wavelengths contribute equally. κ = O(1).

---

## Two results

### Result A — the computation

Phonon transport cross-section of a disclination loop, full BZ.
sin²(k)·σ_tr ≈ const → κ = O(1).
Nobody has done this before (literature has point defects, dislocations,
superfluid vortices — not disclination loops in discrete elastic lattice).

### Result B — the deeper physics

Multiple scattering in a 2D defect cluster renormalizes the Born transport
exponent from -5/2 to ≈ -2.

This is a **collective property** of the scatterers. It does NOT depend on:
- disclination specifics
- Peierls gauge details
- foam lattice structure

It depends on:
- **geometry**: 2D cluster of scatterers
- **embedding**: in 3D medium
- **kernel**: Green function G ~ 1/r (long-range)

### Test evidence for Result B

| Test | What it shows |
|------|--------------|
| TestRandomDisk | Random positions → same shift. Not lattice-specific. |
| TestSeedIndependence | Multiple random seeds → same enhancement. Robust. |
| TestPositionalNoise | ±0.1a noise → same result. Not position-sensitive. |
| TestGeometryDependence | disk > line > annulus > single bond. 2D filling matters. |
| test_disk_interior_matters | R=9: disk >> annulus. Interior drives shift, not perimeter. |
| TestShiftDecomposition | 80% from inter-bond 1/r, 9% from self-energy. Cooperation. |
| TestPropagationRange | Near-field r≤3 dominates 60%+. Short-range cooperation. |
| TestPhaseRemoval | Static 1/r > physical exp(ikr)/r. Phase reduces cooperation. |
| TestUniversality | Lattice disk, random disk, square disk → similar exponent. |

---

## Connection to ST_11

In ST_11: phonons bombard defect → drag → diffusion → if κ=1 → D = ℏ/(2m) → Schrödinger emerges.

This paper shows the **mechanism** by which κ becomes O(1): collective cooperation
in the 2D disk makes all phonon modes contribute equally to drag.
Not fine-tuning — geometric consequence of 2D object in 3D medium.

---

## Reviewer's claim to verify

The reviewer says this is analogous to:
- electromagnetic cooperative scattering / superradiance
- multiple impurity scattering / cross-section renormalization
- collective scattering in random media

**Status: VERIFIED (Mar 2026).** Results:

### 1. Superradiance analogy: WRONG DIRECTION

Superradiance: N emitters → N² radiation (cooperative enhancement).
Our case: MS actually **suppresses** scattering at low k.

Data (R=5):

| k | MS/Born total | MS/Born forward | Born forward fraction |
|---|--------------|----------------|----------------------|
| 0.3 | 0.61 | 0.54 | 54% |
| 0.9 | 0.99 | 0.99 | 2% |
| 1.5 | 1.18 | 1.63 | 0.4% |

At low k: Born has strong forward coherence (54% of power in forward cone).
MS **breaks** this coherence → forward drops to 54% of Born, total to 61%.

At high k: Born has no forward coherence (phases random).
MS adds modestly incoherent (+18%).

**The exponent shift is from DECOHERENCE at low k, not ENHANCEMENT.**

MS flattens N_eff(k) by reducing the low-k peak (where Born forward
coherence is strongest), not by amplifying anything.

Verdict: NOT superradiance. Opposite mechanism — **forward decoherence**.
The reviewer is right that it's a collective effect, but wrong about the
direction: it's suppression at low k, not cooperative enhancement.

### 2. Is -5/2 → -2 specific to 2D-in-3D?

The Born exponent -5/2 requires:
- 2D disk geometry (cone area ~ k^{-3/2})
- transport weight (adds -1)

MS shift requires:
- multiple scatterers (N > 1)
- inter-bond Green function G ~ 1/r
- forward coherence to break (only exists at kR < 1)

1D line: shift saturates (tested, spread < 0.15). NOT the same.
Annulus: shift smaller than disk. Interior matters (tested R=9).

The mechanism is specific to **filled 2D disk** in 3D. Not general.

### 3. Is -2 universal for any 2D cluster with 1/r kernel?

- Random disk → same shift as lattice disk (tested). Position-independent.
- Square disk → similar exponent (tested). Shape-independent.
- C = 0.31-0.35 depends on UV cutoff. Magnitude is NOT universal.
- Direction (positive shift) IS robust for any 2D cluster with 1/r kernel,
  because forward coherence always exists at low k for any filled 2D shape.

### 4. Physical mechanism summary

The mechanism is: **forward decoherence by inter-bond re-scattering**.

At low k, a filled 2D disk has strong forward coherence (all phases ≈ 1).
N_eff_Born(k=0.3) is huge because |Σ exp(ikdx)|² ≈ N².
Inter-bond propagation (T-matrix) mixes the phases, breaking forward
coherence. This reduces N_eff at low k more than at high k → positive
exponent shift → flatter integrand.

The shift magnitude depends on coupling strength |V| and UV cutoff (C ≈ 0.3).
The shift direction is geometric — any filled 2D disk will have forward
coherence at low k that inter-bond re-scattering can break.

---

## What to put in the paper

Suggested Discussion paragraph (revised after verification):

> The flattening of the Boltzmann integrand originates from the interplay
> between forward coherence and inter-bond re-scattering. At low k, the
> Born approximation gives a strongly coherent forward cone (N_eff ~ N²)
> that produces a steep exponent −5/2. Multiple scattering through the
> inter-bond Green function partially breaks this forward coherence,
> reducing N_eff at low k more than at high k. The net effect is a
> positive exponent shift of approximately +1/2, flattening the integrand
> sin²(k)·σ_tr to a nearly constant function across the Brillouin zone.

**Our assessment:** More accurate than reviewer's original formulation.
Key correction: it's forward DECOHERENCE, not cooperative ENHANCEMENT.
The reviewer's "cooperative scattering" language suggests amplification,
but the data shows suppression at low k (MS/Born = 0.61 at k=0.3).
