# CLAUDE.md — Workflow Rules

## Rule 00
Prioritize substance over compliments. Never soften criticism. If an idea has holes, say so directly—"This won't scale because X" is better than "Have you considered…". Challenge assumptions. Point out errors. Useful feedback matters more than comfortable feedback.

## Investigation Process
1. **Command line first** — small exploratory computations, no files yet
2. **Consolidate** — when results are clear, create a `.py` test file
3. **File creation is iterative** — start with pseudocode/section titles/method map, then fill in incrementally
4. **Run tests** — verify all pass
5. **Header = raw output** — write RAW OUTPUT + ANSWER in file docstring
6. **Review** — only then submit for review

## Context Rules
- **Everything stated officially must be backed by a test.** No exceptions.
- Keep notes short. No mega-compositions.

## General
- Small steps. Don't get blocked on large tasks.
- No large files in one shot. Build incrementally.

## Running Tests
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 tests/<test_file>.py
```

## Project
Transport cross-section σ_tr(k) of a disclination loop (vortex ring) in 3D elastic lattice.
Paper target: PRE.
Source: physics_ai/ST_11/wip/w_21_kappa (62 investigation files).

## Structure
- `src/` — infrastructure (elastic_3d, scattering_3d, gauge_3d, parallel_fdtd)
- `tests/` — consolidated tests (organized by paper section)
- `paper/` — .tex + figures
- `internal/` — tracker, skeleton, working notes
- `internal/old_tests/` — original 61 investigation files from w_21_kappa (reference)
