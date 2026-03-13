# st_vortex_transport

Phonon transport cross-section of a disclination loop in a 3D elastic lattice.

**Author:** Alexandru Toader (toader_alexandru@yahoo.com)

## Result

First computation of the phonon transport cross-section sigma_tr(k) for a disclination loop (vortex ring of Peierls-gauged bonds) in a cubic lattice with NN springs. The Boltzmann integrand sin^2(k) * sigma_tr is nearly constant across the Brillouin zone (CV = 5.6% at alpha = 0.30), producing a transport coefficient kappa of order unity without fine-tuning.

The mechanism: Born scattering from a filled disk of N ~ piR^2 bonds gives N_eff ~ (kR)^{-5/2}. Multiple scattering shifts this to ~(kR)^{-2.0} through forward decoherence. An algebraic cancellation between sin^2(k) and 1/v_g^2 removes the remaining k-dependence.

## Paper

Submitted to Physical Review Letters (Mar 2026).

- `paper/main.tex` -- manuscript (revtex4-2, 4 pages)
- `paper/sm.tex` -- supplemental material
- `paper/make_figures.py` -- generates all 3 figures from test data

## Tests

192 tests across 10 files (~13s analytic + 6 min FDTD). Each test file corresponds to one step of the mechanism chain in the paper.

See `tests/tests_map.md` for the complete inventory with per-test descriptions and paper mappings.

```
tests/
├── test_1_born_perbond.py          (21 tests)  Per-bond Born: sigma = C0*V/v_g^2
├── test_2_born_disk.py             (11 tests)  Born exponent -5/2 from disk geometry
├── test_3_multiple_scattering.py   (31 tests)  MS shifts -5/2 to -2.0
├── test_4_assembly.py              (14 tests)  Flat integrand from algebraic cancellation
├── test_5_coupling_requirements.py (12 tests)  Why strong coupling + NN gauging required
├── test_6_mechanism_elimination.py (25 tests)  Rules out T-matrix, resonance, compensation
├── test_7_null_gauging.py          (13 tests)  NNN and AB fail (negative tests)
├── test_8_fdtd_convergence.py      (11 tests)  Box size, PML, bandwidth convergence
├── test_9_forward_mechanism.py     (42 tests)  Forward decoherence decomposition
├── test_10_forward_p2.py           (12 tests)  Scaling, consistency, kappa validation
├── helpers/                        Analytic helpers (Born, MS, geometry, lattice)
└── data/                           FDTD-generated sigma tables
```

## Source code

```
src/
├── elastic_3d.py      Scalar Laplacian, PML damping, FDTD loop
├── scattering_3d.py   Wave packets, sphere measurement, sigma integration
├── gauge_3d.py        Peierls vortex ring (NN + NNN gauging)
└── parallel_fdtd.py   Parallel FDTD utility
```

## Running tests

```bash
# Analytic tests (~13s)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -m pytest tests/test_1*.py tests/test_2*.py tests/test_3*.py tests/test_4*.py tests/test_5*.py tests/test_6*.py tests/test_7*.py tests/test_9*.py tests/test_10*.py -v

# FDTD convergence (~6 min)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -m pytest tests/test_8_fdtd_convergence.py -v
```

## Requirements

- Python 3.9+
- NumPy, SciPy

## License

MIT
